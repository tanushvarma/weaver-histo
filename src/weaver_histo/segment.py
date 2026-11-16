"""Masks and nuclei segmentation."""
from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi
from skimage import filters, morphology, segmentation

from .stains import rgb_to_od


def remove_small(mask: np.ndarray, min_px: float) -> np.ndarray:
    lab, n = ndi.label(mask)
    if n == 0:
        return mask
    keep = np.bincount(lab.ravel()) >= min_px
    keep[0] = False
    return keep[lab]


def fill_small_holes(mask: np.ndarray, max_hole_px: float) -> np.ndarray:
    lab, n = ndi.label(~mask)
    if n == 0:
        return mask
    small = np.bincount(lab.ravel()) <= max_hole_px
    small[0] = False  # label 0 = the mask itself
    return mask | small[lab]


def tissue_mask(rgb, threshold=None, s=1.0, min_area_um2=500.0, fill_hole_um2=200.0):
    """Everything that is not (near-)white background, including pale hydrogel.
    s = um per pixel (1.0 when uncalibrated, i.e. lengths are then pixels)."""
    min_area_px, fill_hole_px = min_area_um2 / (s * s), fill_hole_um2 / (s * s)
    od = filters.gaussian(rgb_to_od(rgb).sum(axis=2), sigma=max(1.0, 1.0 / s), preserve_range=True)
    if threshold is None:
        thr = float(np.clip(filters.threshold_triangle(od), 0.08, 0.6))
    else:
        thr = float(threshold)
    mask = remove_small(od > thr, min_area_px)
    mask = fill_small_holes(mask, fill_hole_px)
    return mask, thr


def segment_nuclei(signal, tissue, thresh_scale=1.0, floor=0.3, s=1.0, min_object_um2=6.0,
                   smooth_um=1.0, split=True, split_h_um=1.0, fill_hole_um2=30.0, threshold=None):
    """Threshold the nuclear signal inside tissue, then split touching nuclei.
    All length-like settings are in um (px when s == 1), so results do not depend
    on the image resolution. If `threshold` is given it is used as-is (slide mode keeps one
    cutoff for every tile). Returns (labels int32, threshold used)."""
    sigma = max(0.5, smooth_um / s)
    split_h = max(0.7, split_h_um / s)
    open_r = max(1, int(round(0.5 / s)))
    min_area_px = max(3.0, min_object_um2 / (s * s))
    sm = filters.gaussian(signal, sigma=sigma, preserve_range=True)
    if threshold is not None:
        thr = float(threshold)
    else:
        vals = sm[tissue]
        if vals.size < 100:
            return np.zeros(signal.shape, np.int32), float(floor)
        try:
            otsu = float(filters.threshold_otsu(vals))
        except ValueError:
            otsu = floor
        thr = max(otsu * thresh_scale, floor)
    binary = (sm > thr) & tissue
    binary = ndi.binary_opening(binary, structure=morphology.disk(open_r))
    binary = fill_small_holes(binary, fill_hole_um2 / (s * s))
    binary = remove_small(binary, min_area_px)
    comp, n = ndi.label(binary)
    if n == 0 or not split:
        return comp.astype(np.int32), thr

    dist = filters.gaussian(ndi.distance_transform_edt(binary), sigma=sigma, preserve_range=True)
    # h_maxima plateaus are 8-connected; label them as such, otherwise a peak
    # made of two diagonal pixels becomes two markers and the nucleus is cut in half.
    markers, _ = ndi.label(morphology.h_maxima(dist, split_h) & binary,
                           structure=np.ones((3, 3), dtype=int))
    # every component must get at least one marker
    has = np.zeros(n + 1, bool)
    has[comp[markers > 0]] = True
    missing = np.flatnonzero(~has[1:]) + 1
    if missing.size:
        pos = ndi.maximum_position(dist, comp, index=missing)
        nxt = markers.max() + 1
        for (y, x) in pos:
            markers[y, x] = nxt
            nxt += 1
    labels = segmentation.watershed(-dist, markers, mask=binary)
    labels, _, _ = segmentation.relabel_sequential(labels)
    return labels.astype(np.int32), thr


def matrix_from_signal(signal, tissue, nuclei, smooth_px, thresh=None,
                       open_px=3, min_area_px=3000, hole_px=600):
    """Hydrogel/matrix mask from stain *intensity*, not cell density, so that
    cells invading the matrix do not erase it. Pale, homogeneous material
    (low stain signal once nuclei are ignored) is matrix; strongly stained host
    tissue is not. Returns (mask, threshold or None)."""
    w = (tissue & ~ndi.binary_dilation(nuclei, iterations=1)).astype(np.float32)
    num = filters.gaussian(signal * w, sigma=smooth_px, preserve_range=True)
    den = filters.gaussian(w, sigma=smooth_px, preserve_range=True)
    sm = num / np.maximum(den, 1e-3)
    valid = tissue & (den > 0.02)
    vals = sm[valid]
    if thresh is None:
        if vals.size < 100 or float(vals.std()) < 1e-6:
            return np.zeros(signal.shape, bool), None
        thresh = float(filters.threshold_otsu(vals))
    m = valid & (sm < thresh)
    m = ndi.binary_opening(m, structure=morphology.disk(max(1, int(open_px))))
    m = fill_small_holes(m, hole_px)
    m = remove_small(m, min_area_px)
    return m, float(thresh)


def collagen_mask(collagen, tissue, nuclei, thresh_scale=1.0, floor=0.3, s=1.0,
                  smooth_um=1.0, min_area_um2=30.0):
    min_area_px = max(3.0, min_area_um2 / (s * s))
    sm = filters.gaussian(collagen, sigma=max(0.5, smooth_um / s), preserve_range=True)
    vals = sm[tissue]
    if vals.size < 100:
        return np.zeros(collagen.shape, bool), float(floor)
    try:
        otsu = float(filters.threshold_otsu(vals))
    except ValueError:
        otsu = floor
    thr = max(otsu * thresh_scale, floor)
    m = (sm > thr) & tissue & ~ndi.binary_dilation(nuclei, iterations=1)
    return remove_small(m, min_area_px), thr


def refine_edge(signal, mask, tissue, nuclei, s, band_um=8.0, smooth_um=1.5, min_area_px=30.0):
    """Move a region's boundary to where the stain signal is half-way between the region's
    interior and the surrounding tissue (the usual half-maximum edge). A trained or smoothed
    mask is usually a few pixels off because it never saw an example of the boundary itself.

    Only pixels within band_um of the current edge may change; nuclei are ignored when measuring.
    Returns the mask unchanged if the interior or surroundings cannot be measured."""
    if not mask.any():
        return mask
    signal = signal.astype(np.float32)
    if nuclei is not None and nuclei.any() and nuclei.mean() < 0.5:
        # nuclei are dark blobs on top of the material being measured: replace their signal by
        # that of their surroundings (normalised convolution) instead of freezing the old decision
        nuc = ndi.binary_dilation(nuclei, iterations=1)
        w = (~nuc).astype(np.float32)
        sg = max(1.5, 4.0 / s)
        fill = ndi.gaussian_filter(signal * w, sg) / np.maximum(ndi.gaussian_filter(w, sg), 1e-3)
        signal = np.where(nuc, fill, signal)
    sm = ndi.gaussian_filter(signal, max(0.7, smooth_um / s))
    d_in = ndi.distance_transform_edt(mask)
    d_out = ndi.distance_transform_edt(~mask)
    band = max(2.0, band_um / s)
    core_in = mask & (d_in > band)
    core_out = tissue & ~mask & (d_out > band) & (d_out < 4 * band)
    if core_in.sum() < 50 or core_out.sum() < 50:
        return mask
    m_in, m_out = float(np.median(sm[core_in])), float(np.median(sm[core_out]))
    if abs(m_in - m_out) < 1e-4:
        return mask
    mid = 0.5 * (m_in + m_out)
    like_inside = (sm < mid) if m_in < m_out else (sm > mid)
    near = (mask & (d_in <= band)) | (~mask & (d_out <= band))   # edt of a mask is 0 outside it
    new = np.where(near, like_inside, mask)
    new = ndi.binary_opening(new, structure=morphology.disk(1))
    return remove_small(new, min_area_px)
