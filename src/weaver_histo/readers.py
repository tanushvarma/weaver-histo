"""Uniform access to slides at any resolution.

SlideReader.read_um() returns an RGB uint8 region at a requested um/px, no
matter whether the source is a whole-slide scan (.vsi, .svs, .ndpi, .czi, ...,
read through SlideIO, no Java needed) or an ordinary image (.tif/.png/.jpg).

All coordinates passed to read_um() are in microns from the top-left corner of
the selected scene, so the rest of the code never has to think about pyramid
levels or pixel sizes.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image

from .loading import _to_rgb_uint8

SLIDEIO_DRIVERS = {
    ".vsi": "VSI", ".svs": "SVS", ".ndpi": "NDPI", ".czi": "CZI", ".scn": "SCN",
    ".afi": "AFI", ".zvi": "ZVI", ".dcm": "DCM", ".qptiff": "QPTIFF",
}
FLAT_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


@dataclass
class SceneInfo:
    index: int
    name: str
    width_px: int
    height_px: int
    um_per_px: Optional[float]
    magnification: Optional[float] = None

    @property
    def megapixels(self) -> float:
        return self.width_px * self.height_px / 1e6

    def __str__(self):
        res = f"{self.um_per_px:.3f} um/px" if self.um_per_px else "unknown um/px"
        mag = f", {self.magnification:g}x" if self.magnification else ""
        return f"[{self.index}] {self.name or 'scene'}: {self.width_px}x{self.height_px} px, {res}{mag}"


class SlideReader:
    """Base class. width_px/height_px/um_per_px describe the full-resolution scene."""

    path: Path
    width_px: int
    height_px: int
    um_per_px: Optional[float]

    # -- subclasses implement
    def _read_px(self, x, y, w, h, out_w, out_h) -> np.ndarray:  # full-res px rect -> RGB
        raise NotImplementedError

    def close(self):
        pass

    # -- shared
    @property
    def width_um(self):
        return self.width_px * self.um_per_px

    @property
    def height_um(self):
        return self.height_px * self.um_per_px

    def read_um(self, x_um, y_um, w_um, h_um, target_um_per_px) -> np.ndarray:
        """Region [x_um, x_um+w_um) x [y_um, y_um+h_um) at target_um_per_px.
        Parts outside the scene are padded white. Output is RGB uint8."""
        if not self.um_per_px:
            raise ValueError("This image has no pixel size; set um/px first.")
        s = self.um_per_px
        out_w = max(1, int(round(w_um / target_um_per_px)))
        out_h = max(1, int(round(h_um / target_um_per_px)))
        x0, y0 = x_um / s, y_um / s
        w_px, h_px = w_um / s, h_um / s
        # clip to the scene, pad the rest
        cx0, cy0 = max(0.0, x0), max(0.0, y0)
        cx1, cy1 = min(float(self.width_px), x0 + w_px), min(float(self.height_px), y0 + h_px)
        canvas = np.full((out_h, out_w, 3), 255, np.uint8)
        if cx1 <= cx0 or cy1 <= cy0:
            return canvas
        # output pixel extent of the clipped part
        ox0 = int(round((cx0 - x0) / w_px * out_w))
        oy0 = int(round((cy0 - y0) / h_px * out_h))
        ox1 = int(round((cx1 - x0) / w_px * out_w))
        oy1 = int(round((cy1 - y0) / h_px * out_h))
        ow, oh = max(1, ox1 - ox0), max(1, oy1 - oy0)
        rx, ry = int(round(cx0)), int(round(cy0))
        rw, rh = max(1, int(round(cx1 - cx0))), max(1, int(round(cy1 - cy0)))
        block = self._read_px(rx, ry, rw, rh, ow, oh)
        canvas[oy0:oy0 + block.shape[0], ox0:ox0 + block.shape[1]] = block[:out_h - oy0, :out_w - ox0]
        return canvas

    def overview(self, max_side_px: int = 3000):
        """Whole scene as a small RGB image. Returns (rgb, um_per_px_of_that_image)."""
        long_um = max(self.width_um, self.height_um)
        target = max(self.um_per_px, long_um / max_side_px)
        return self.read_um(0, 0, self.width_um, self.height_um, target), target


class FlatReader(SlideReader):
    """Ordinary PNG/JPG/TIFF held in memory (fine up to a few hundred megapixels)."""

    def __init__(self, path, um_per_px: Optional[float] = None):
        self.path = Path(path)
        try:
            img = Image.open(self.path)
            arr = np.asarray(img.convert("RGB"))
        except Exception:
            import tifffile
            arr = _to_rgb_uint8(tifffile.imread(self.path))
        self._arr = arr
        self.height_px, self.width_px = arr.shape[:2]
        self.um_per_px = um_per_px

    def _read_px(self, x, y, w, h, out_w, out_h):
        crop = self._arr[y:y + h, x:x + w]
        if crop.shape[0] == out_h and crop.shape[1] == out_w:
            return crop
        return np.asarray(Image.fromarray(crop).resize((out_w, out_h), Image.LANCZOS if out_w < w else Image.BILINEAR))

    def scenes(self):
        return [SceneInfo(0, self.path.name, self.width_px, self.height_px, self.um_per_px)]


class SlideioReader(SlideReader):
    """Whole-slide formats through SlideIO (pyramid-aware, native VSI support)."""

    def __init__(self, path, scene_index: Optional[int] = None, um_per_px: Optional[float] = None,
                 driver: Optional[str] = None):
        self.path = Path(path)
        try:
            import slideio
        except ImportError as err:
            raise ImportError(
                f'Reading {self.path.suffix.lower()} files needs the optional dependency: pip install "weaver-histo[vsi]"'
            ) from err
        drv = driver or SLIDEIO_DRIVERS.get(self.path.suffix.lower(), "GDAL")
        self._slide = slideio.open_slide(str(self.path), drv)
        self._infos = []
        for i in range(self._slide.num_scenes):
            sc = self._slide.get_scene(i)
            w, h = sc.size
            res = sc.resolution  # metres per pixel (0 when unknown)
            um = float(res[0]) * 1e6 if res and res[0] else None
            mag = float(sc.magnification) if getattr(sc, "magnification", 0) else None
            self._infos.append(SceneInfo(i, sc.name, int(w), int(h), um, mag))
        if not self._infos:
            raise ValueError("No scenes found in this file.")
        if scene_index is None:  # default: the biggest scene is the main scan
            scene_index = max(range(len(self._infos)), key=lambda i: self._infos[i].megapixels)
        self.select_scene(scene_index, um_per_px)

    def select_scene(self, index: int, um_per_px: Optional[float] = None):
        info = self._infos[index]
        self.scene_index = index
        self._scene = self._slide.get_scene(index)
        self.width_px, self.height_px = info.width_px, info.height_px
        self.um_per_px = um_per_px or info.um_per_px
        self.scene_name = info.name

    def scenes(self):
        return list(self._infos)

    def _read_px(self, x, y, w, h, out_w, out_h):
        arr = self._scene.read_block(rect=(int(x), int(y), int(w), int(h)), size=(int(out_w), int(out_h)))
        return _to_rgb_uint8(arr)

    def close(self):
        try:
            self._slide.close()
        except Exception:
            pass


def open_slide(path, um_per_px: Optional[float] = None, scene_index: Optional[int] = None) -> SlideReader:
    """Open a slide or image. Whole-slide formats use SlideIO; plain images are loaded directly,
    except very large TIFFs, which also go through SlideIO so they are never fully decoded."""
    p = Path(path)
    ext = p.suffix.lower()
    if not p.exists():
        raise FileNotFoundError(f"File not found: {p}")
    if ext in SLIDEIO_DRIVERS:
        return SlideioReader(p, scene_index, um_per_px)
    if ext in (".tif", ".tiff"):
        try:
            import tifffile
            with tifffile.TiffFile(p) as t:
                shape = t.pages[0].shape
            big = shape[0] * shape[1] > 150e6
        except Exception:
            big = False
        if big:
            return SlideioReader(p, scene_index, um_per_px, driver="GDAL")
    if ext in FLAT_EXTS:
        return FlatReader(p, um_per_px)
    raise ValueError(f"Unsupported file type: {ext}")
