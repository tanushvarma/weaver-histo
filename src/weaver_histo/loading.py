"""Image intake: PNG/JPG/TIFF (8/16-bit, grayscale or RGB) -> RGB uint8.

Named loading.py (not io.py) on purpose: a module called io.py can shadow the
standard library if a script is ever run from inside this folder.
"""
from __future__ import annotations

import numpy as np
from PIL import Image

Image.MAX_IMAGE_PIXELS = None  # we downscale ourselves


def _to_rgb_uint8(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr)
    while arr.ndim > 3:  # multi-page / stacks: take the first plane
        arr = arr[0]
    if arr.ndim == 2:
        arr = np.stack([arr] * 3, axis=-1)
    if arr.ndim == 3 and arr.shape[0] in (3, 4) and arr.shape[-1] not in (3, 4):
        arr = np.moveaxis(arr, 0, -1)  # channels-first -> channels-last
    if arr.ndim != 3 or arr.shape[-1] < 3:
        raise ValueError(f"Unsupported image shape {arr.shape}")
    arr = arr[..., :3]
    if arr.dtype == np.uint8:
        return arr
    if arr.dtype == np.uint16:
        return np.rint(arr / 257.0).astype(np.uint8)
    if np.issubdtype(arr.dtype, np.floating):
        a = arr.astype(np.float64)
        if a.max() <= 1.0:
            a = a * 255.0
        return np.clip(np.rint(a), 0, 255).astype(np.uint8)
    return np.clip(arr, 0, 255).astype(np.uint8)


def load_image(src, max_side: int = 3000):
    """Load an image from a path or file-like object.

    Returns (rgb_uint8, downscale) where downscale >= 1 is the factor the image
    was shrunk by to fit max_side. Multiply the microscope's um/px by it to get
    the effective um/px of the returned array.
    """
    arr = None
    try:
        img = Image.open(src)
        if img.mode in ("RGB", "RGBA", "L", "LA", "P", "CMYK"):
            arr = np.asarray(img.convert("RGB"))
    except Exception:
        arr = None
    if arr is None:
        import tifffile

        if hasattr(src, "seek"):
            src.seek(0)
        arr = _to_rgb_uint8(tifffile.imread(src))

    h, w = arr.shape[:2]
    downscale = 1.0
    if max(h, w) > max_side:
        f = max(h, w) / max_side
        new_w, new_h = max(1, round(w / f)), max(1, round(h / f))
        arr = np.asarray(Image.fromarray(arr).resize((new_w, new_h), Image.LANCZOS))
        downscale = w / new_w
    return arr, downscale
