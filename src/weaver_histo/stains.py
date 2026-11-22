"""Stain normalisation and colour deconvolution (native NumPy/scikit-image).

This is the interface a HistomicsTK backend could later replace; nothing here
needs HistomicsTK or any compiled extension beyond NumPy/SciPy/scikit-image.
"""

from __future__ import annotations

import numpy as np
from skimage import color

_EPS = 1e-8

# Macenko reference H&E optical-density vectors (columns: hematoxylin, eosin)
HE_REF_VECTORS = np.array([[0.5626, 0.2159], [0.7201, 0.8012], [0.4062, 0.5581]])
HE_REF_MAXC = np.array([1.9705, 1.0308])

# Masson's trichrome OD vectors (aniline-blue collagen, red/fuchsin cytoplasm),
# values as in the common Fiji colour-deconvolution preset. These are starting
# points: calibrate on your own slides via the reference colours in the app.
TRI_BLUE = np.array([0.7995107, 0.5913521, 0.10528667])
TRI_RED = np.array([0.09997159, 0.73738605, 0.6680326])


def rgb_to_od(rgb: np.ndarray) -> np.ndarray:
    return -np.log((rgb.astype(np.float32) + 1.0) / 256.0)


def od_to_rgb(od: np.ndarray) -> np.ndarray:
    return np.clip(np.rint(256.0 * np.exp(-od) - 1.0), 0, 255).astype(np.uint8)


def od_vector_from_rgb(rgb_triplet) -> np.ndarray:
    """Unit optical-density vector of a representative stain colour."""
    v = rgb_to_od(np.array(rgb_triplet, dtype=np.float32).reshape(1, 1, 3)).reshape(3)
    return v / max(float(np.linalg.norm(v)), _EPS)


def estimate_he_vectors(rgb, beta=0.15, alpha=1.0, min_pixels=2000) -> np.ndarray | None:
    """Macenko stain-vector estimate. Returns 3x2 [H, E] or None if unreliable."""
    od = rgb_to_od(rgb).reshape(-1, 3).astype(np.float64)
    od = od[(od > beta).all(axis=1)]
    if od.shape[0] < min_pixels:
        return None
    if od.shape[0] > 500_000:
        od = od[np.random.default_rng(0).choice(od.shape[0], 500_000, replace=False)]
    _, eigvecs = np.linalg.eigh(np.cov(od.T))
    plane = eigvecs[:, 1:3]
    proj = od @ plane
    phi = np.arctan2(proj[:, 1], proj[:, 0])
    lo, hi = np.percentile(phi, [alpha, 100 - alpha])
    v1 = plane @ np.array([np.cos(lo), np.sin(lo)])
    v2 = plane @ np.array([np.cos(hi), np.sin(hi)])
    if v1.sum() < 0:
        v1 = -v1
    if v2.sum() < 0:
        v2 = -v2
    h, e = (v1, v2) if v1[0] > v2[0] else (v2, v1)  # H absorbs more red
    h, e = h / np.linalg.norm(h), e / np.linalg.norm(e)
    # Sanity: must resemble a real H&E pair, else the caller falls back to the
    # reference. The H/E pair is intrinsically close (~22 deg apart), so a
    # near-collinear estimate makes the 2-stain unmixing ill-conditioned: very
    # dark nuclei then leak into the eosin channel and read too low in H.
    ref_h = HE_REF_VECTORS[:, 0] / np.linalg.norm(HE_REF_VECTORS[:, 0])
    ref_e = HE_REF_VECTORS[:, 1] / np.linalg.norm(HE_REF_VECTORS[:, 1])
    if float(h @ ref_h) < 0.95 or float(e @ ref_e) < 0.95 or float(h @ e) > 0.96:
        return None
    return np.stack([h, e], axis=1)


def deconvolve(rgb: np.ndarray, W: np.ndarray, chunk_rows: int = 512) -> np.ndarray:
    """Least-squares unmixing. W is 3 x k (unit OD vectors).
    Returns H x W x k float32 concentrations clipped at 0."""
    pinv = np.linalg.pinv(W).astype(np.float32)  # k x 3
    h, w, _ = rgb.shape
    out = np.empty((h, w, W.shape[1]), dtype=np.float32)
    for r0 in range(0, h, chunk_rows):
        od = rgb_to_od(rgb[r0 : r0 + chunk_rows])
        out[r0 : r0 + chunk_rows] = np.clip(od @ pinv.T, 0, None)
    return out


def dark_score(rgb: np.ndarray) -> np.ndarray:
    """Trichrome nuclear signal: the optical density of the least-absorbed channel."""
    # Blue-black nuclei absorb all three channels, while blue collagen and red cytoplasm each pass one.
    # As an OD it keeps dark pyknotic nuclei clearly above normal ones; a linear value compresses them.
    return rgb_to_od(rgb).min(axis=2)


def macenko_normalize(rgb, tissue=None, vectors=None, chunk_rows: int = 512) -> np.ndarray:
    """Macenko normalisation to the reference H&E appearance (H&E only)."""
    W = vectors if vectors is not None else estimate_he_vectors(rgb)
    if W is None:
        W = HE_REF_VECTORS
    conc = deconvolve(rgb, W)
    sel = tissue if (tissue is not None and tissue.any()) else np.ones(rgb.shape[:2], bool)
    max_c = np.percentile(conc[sel].reshape(-1, 2), 99, axis=0)
    scale = (HE_REF_MAXC / np.maximum(max_c, 1e-3)).astype(np.float32)
    ref = HE_REF_VECTORS.astype(np.float32)
    out = np.empty_like(rgb)
    for r0 in range(0, rgb.shape[0], chunk_rows):
        od = (conc[r0 : r0 + chunk_rows] * scale) @ ref.T
        out[r0 : r0 + chunk_rows] = od_to_rgb(od)
    return out


def lab_stats(rgb, mask=None) -> tuple[np.ndarray, np.ndarray]:
    lab = color.rgb2lab(rgb)
    px = lab[mask] if (mask is not None and mask.any()) else lab.reshape(-1, 3)
    return px.mean(axis=0), px.std(axis=0)


def reinhard_normalize(rgb, target_stats, mask=None) -> np.ndarray:
    """Reinhard LAB mean/std matching to a reference image's stats (any stain)."""
    lab = color.rgb2lab(rgb)
    m, s = lab_stats(rgb, mask)
    t_m, t_s = target_stats
    out = (lab - m) / np.maximum(s, 1e-6) * t_s + t_m
    return np.clip(np.rint(color.lab2rgb(out) * 255.0), 0, 255).astype(np.uint8)


def separate(
    rgb, stain, tri_blue_rgb=None, tri_red_rgb=None, he_vectors=None
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Stain separation for slide mode. Returns (nuclear, second, matrix_signal, nuclei_floor)."""
    # H&E: hematoxylin, eosin, eosin, 0.3. Trichrome: neutral OD, collagen, collagen + red, 0.6.
    if stain == "he":
        W = he_vectors if he_vectors is not None else HE_REF_VECTORS
        conc = deconvolve(rgb, W)
        return conc[..., 0], conc[..., 1], conc[..., 1], 0.3
    blue = od_vector_from_rgb(tri_blue_rgb) if tri_blue_rgb else TRI_BLUE
    red = od_vector_from_rgb(tri_red_rgb) if tri_red_rgb else TRI_RED
    conc = deconvolve(rgb, np.stack([blue, red], axis=1))
    return dark_score(rgb), conc[..., 0], conc[..., 0] + conc[..., 1], 0.6
