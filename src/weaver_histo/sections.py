"""Find the tissue sections on a slide overview.

Works on a low-resolution picture of the whole slide (a scanner's macro image,
or the main scan read at ~10-30 um/px). Sections are found by colour: H&E and
trichrome tissue is pink/purple/blue and saturated; the paper label, pen ink,
black slide holder and the glass are not.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage as ndi
from skimage import color, morphology


@dataclass
class Section:
    id: int
    bbox: tuple  # (x0, y0, x1, y1) in overview pixels, x1/y1 exclusive
    mask: np.ndarray  # bool, cropped to bbox
    area_px: int

    @property
    def centroid(self) -> tuple[float, float]:
        ys, xs = np.nonzero(self.mask)
        return (self.bbox[0] + xs.mean(), self.bbox[1] + ys.mean())


HUE_RANGES = {
    # H&E tissue is pink-to-purple (hue ~0.78-0.95); grey-lavender paper (~0.70) is excluded
    "he": ((0.74, 1.0), (0.0, 0.06)),
    # trichrome adds aniline-blue collagen (hue ~0.55-0.65) next to red cytoplasm
    "trichrome": ((0.50, 1.0), (0.0, 0.08)),
}


def tissue_colour_mask(rgb: np.ndarray, min_sat: float = 0.045, min_val: float = 0.30, stain: str = "he") -> np.ndarray:
    """Pixels that are coloured like stained tissue (not grey ink/paper, not black, not glass)."""
    hue_ranges = HUE_RANGES.get(stain, HUE_RANGES["he"])
    hsv = color.rgb2hsv(rgb)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    in_hue = np.zeros(h.shape, bool)
    for lo, hi in hue_ranges:
        in_hue |= (h >= lo) & (h <= hi)
    return (s >= min_sat) & (v >= min_val) & in_hue


def find_sections(
    rgb: np.ndarray,
    um_per_px: float | None = None,
    min_area_frac: float = 0.0015,
    merge_frac: float = 0.025,
    min_area_um2: float | None = None,
    min_sat: float = 0.045,
    stain: str = "he",
    min_texture: float = 0.0,
) -> list[Section]:
    """Detect tissue sections, sorted left-to-right then top-to-bottom."""
    # merge_frac is the gap, as a fraction of the long side, bridged so lacy tissue stays one section.
    long_side = max(rgb.shape[:2])
    m = tissue_colour_mask(rgb, min_sat=min_sat, stain=stain)
    m = ndi.binary_opening(m, structure=morphology.disk(1))  # drop single-pixel colour noise
    r = max(2, int(round(merge_frac * long_side)))
    merged = ndi.binary_closing(np.pad(m, r), structure=morphology.disk(r))[r:-r, r:-r]
    merged = ndi.binary_fill_holes(merged)

    lab, n = ndi.label(merged)
    if n == 0:
        return []
    if min_area_um2 is not None and um_per_px:
        min_px = min_area_um2 / (um_per_px**2)
    else:
        min_px = min_area_frac * rgb.shape[0] * rgb.shape[1]
    # optional: reject smooth tinted paper (min_texture > 0). Off by default: hue already separates
    # grey-lavender paper from H&E tissue, and smooth tissue (dense stroma, hydrogel) must not be dropped.
    v = color.rgb2gray(rgb)
    mu = ndi.uniform_filter(v, 5)
    texture = np.sqrt(np.maximum(ndi.uniform_filter(v * v, 5) - mu * mu, 0))
    objs = ndi.find_objects(lab)
    out = []
    for i, sl in enumerate(objs, start=1):
        comp = lab[sl] == i
        # report the area of actual tissue pixels inside the merged blob
        tissue_px = int((m[sl] & comp).sum())
        if tissue_px < min_px:
            continue
        if float(np.median(texture[sl][m[sl] & comp])) < min_texture * 0.8:  # smooth => not tissue
            continue
        y0, y1, x0, x1 = sl[0].start, sl[0].stop, sl[1].start, sl[1].stop
        out.append(Section(0, (x0, y0, x1, y1), comp, tissue_px))
    out.sort(key=lambda s: (round(s.centroid[0] / max(1, long_side * 0.05)), s.centroid[1]))
    for k, s in enumerate(out, start=1):
        s.id = k
    return out


def draw_sections(rgb: np.ndarray, sections: list[Section], width: int = 2) -> np.ndarray:
    """Overview with each section's outline and number."""
    import cv2

    out = rgb.copy()
    for s in sections:
        x0, y0, x1, y1 = s.bbox
        edge = s.mask & ~ndi.binary_erosion(s.mask)
        if width > 1:
            edge = ndi.binary_dilation(edge, iterations=width - 1)
        crop = out[y0:y1, x0:x1]
        crop[edge] = (255, 0, 255)
        cx, cy = s.centroid
        scale = max(0.5, min(rgb.shape[:2]) / 600)
        cv2.putText(
            out,
            str(s.id),
            (int(cx), int(cy)),
            cv2.FONT_HERSHEY_SIMPLEX,
            scale,
            (0, 140, 0),
            max(1, int(round(scale * 2))),
            cv2.LINE_AA,
        )
    return out
