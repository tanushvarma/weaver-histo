"""Quantification: per-nucleus features, nuclear-state classes, infiltration,
capsule/fibrosis, construct geometry and spatial statistics."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import ndimage as ndi
from scipy.spatial import cKDTree
from skimage import measure, morphology

CLASSES = ("intact", "pyknotic", "pale", "fragment")


# ---------------------------------------------------------------- per nucleus
def nucleus_table(labels: np.ndarray, signal: np.ndarray) -> pd.DataFrame:
    cols = ["label", "cy", "cx", "area_px", "core_area_px", "perimeter_px", "major_axis_px",
            "minor_axis_px", "eccentricity", "solidity", "circularity",
            "mean_signal", "core_mean_signal", "max_signal", "std_signal"]
    if labels.max() == 0:
        return pd.DataFrame(columns=cols)

    def std_signal(regionmask, intensity):
        return float(np.std(intensity[regionmask]))

    # "Core" = pixels at >= half of the object's own peak (95th percentile).
    # A fixed global threshold gives very dark small nuclei a wide halo, which
    # inflates their area and dilutes their mean; the core is robust to that.
    def core_area(regionmask, intensity):
        v = intensity[regionmask]
        return float((v >= 0.5 * np.percentile(v, 95)).sum())

    def core_mean_signal(regionmask, intensity):
        v = intensity[regionmask]
        return float(v[v >= 0.5 * np.percentile(v, 95)].mean())

    t = measure.regionprops_table(
        labels, intensity_image=signal,
        properties=("label", "area", "centroid", "perimeter", "major_axis_length",
                    "minor_axis_length", "eccentricity", "solidity",
                    "intensity_mean", "intensity_max"),
        extra_properties=(std_signal, core_area, core_mean_signal),
    )
    df = pd.DataFrame({
        "label": t["label"], "cy": t["centroid-0"], "cx": t["centroid-1"],
        "area_px": t["area"], "core_area_px": t["core_area"], "perimeter_px": t["perimeter"],
        "major_axis_px": t["major_axis_length"], "minor_axis_px": t["minor_axis_length"],
        "eccentricity": t["eccentricity"], "solidity": t["solidity"],
        "mean_signal": t["intensity_mean"], "core_mean_signal": t["core_mean_signal"],
        "max_signal": t["intensity_max"],
        "std_signal": t["std_signal"],
    })
    per = np.maximum(df["perimeter_px"].to_numpy(), 1e-6)
    df["circularity"] = np.clip(4 * np.pi * df["area_px"].to_numpy() / per ** 2, 0, 1)
    return df[cols]


def classify_nuclei(df: pd.DataFrame, p, ref_area_px=None) -> tuple[pd.DataFrame, dict]:
    """Nuclear-state classes from size, density, roundness (relative to the
    image's own typical nucleus, so no absolute calibration is needed).

    Size and darkness are measured on each nucleus's half-peak core.

    fragment : much smaller than a typical nucleus (karyorrhexis proxy)
    pyknotic : small, dark, round, compact (pyknosis proxy)
    pale     : normal size but faint (karyolysis / ghost-nucleus proxy)
    intact   : everything else
    """
    df = df.copy()
    if df.empty:
        df["rel_area"] = df["rel_dark"] = []
        df["state"] = []
        return df, {"ref_area_px": np.nan, "ref_signal": np.nan}
    area = df["core_area_px"].to_numpy(float)
    med_all = float(np.median(area))
    if ref_area_px:
        ref_area = float(ref_area_px)
    else:
        ref_area = float(np.median(area[area >= 0.5 * med_all]))
    big = area >= 0.5 * ref_area
    sig = df["core_mean_signal"].to_numpy(float)
    ref_sig = float(np.median(sig[big])) if big.any() else float(np.median(sig))
    rel_area, rel_dark = area / ref_area, sig / max(ref_sig, 1e-6)

    state = np.full(len(df), "intact", dtype=object)
    state[(rel_dark < p.pale_rel_dark) & (rel_area >= p.pyknotic_rel_area)] = "pale"
    pyk = ((rel_area < p.pyknotic_rel_area) & (rel_dark > p.pyknotic_rel_dark)
           & (df["solidity"].to_numpy() > p.pyknotic_min_solidity)
           & (df["eccentricity"].to_numpy() < p.pyknotic_max_ecc))
    state[pyk] = "pyknotic"
    state[rel_area < p.fragment_rel_area] = "fragment"
    df["rel_area"], df["rel_dark"], df["state"] = rel_area, rel_dark, state
    return df, {"ref_area_px": ref_area, "ref_signal": ref_sig}


def state_summary(df: pd.DataFrame, prefix: str = "") -> dict:
    """Counts and indices. Fragments are not whole cells, so they are reported
    per 100 whole nuclei instead of entering the denominator."""
    n = {c: int((df["state"] == c).sum()) for c in CLASSES} if len(df) else {c: 0 for c in CLASSES}
    whole = n["intact"] + n["pyknotic"] + n["pale"]
    nan = float("nan")
    return {
        f"{prefix}n_whole_nuclei": whole,
        f"{prefix}n_intact": n["intact"],
        f"{prefix}n_pyknotic": n["pyknotic"],
        f"{prefix}n_pale": n["pale"],
        f"{prefix}n_fragments": n["fragment"],
        f"{prefix}pyknosis_index_pct": 100 * n["pyknotic"] / whole if whole else nan,
        f"{prefix}pale_nuclei_pct": 100 * n["pale"] / whole if whole else nan,
        f"{prefix}fragments_per_100_nuclei": 100 * n["fragment"] / whole if whole else nan,
    }


# ------------------------------------------------------------------- geometry
def signed_distance(matrix: np.ndarray):
    """Pixels from the matrix edge: positive outside, negative inside."""
    if not matrix.any():
        return None
    return ndi.distance_transform_edt(~matrix) - ndi.distance_transform_edt(matrix)


def sample_at(img: np.ndarray, cy, cx) -> np.ndarray:
    y = np.clip(np.rint(cy).astype(int), 0, img.shape[0] - 1)
    x = np.clip(np.rint(cx).astype(int), 0, img.shape[1] - 1)
    return img[y, x]


def local_thickness(mask: np.ndarray) -> np.ndarray:
    """Local wall thickness (px) = 2 x distance-to-edge along the skeleton."""
    if not mask.any():
        return np.array([])
    edt = ndi.distance_transform_edt(mask)
    return 2.0 * edt[morphology.skeletonize(mask)]


def construct_geometry(matrix: np.ndarray, s: float, U: str) -> dict:
    out = {}
    if not matrix.any():
        return out
    area = float(matrix.sum())
    edge = matrix & ~ndi.binary_erosion(matrix)
    hull = morphology.convex_hull_image(edge)  # hull of the edge == hull of the shape
    hull_area = float(hull.sum())
    thk = local_thickness(matrix) * s
    n_parts = int(measure.label(matrix).max())
    out[f"matrix_area_{U}2"] = area * s * s
    out[f"matrix_perimeter_{U}"] = float(edge.sum()) * s
    out["matrix_n_pieces"] = n_parts
    out["matrix_void_fraction_in_hull_pct"] = 100 * (1 - area / hull_area) if hull_area else float("nan")
    out[f"matrix_wall_thickness_median_{U}"] = float(np.median(thk)) if thk.size else float("nan")
    out[f"matrix_wall_thickness_p90_{U}"] = float(np.percentile(thk, 90)) if thk.size else float("nan")
    return out


def shell_densities(cells, signed, tissue, matrix, s, U, outside_edges, inside_edges):
    """Nuclei per mm^2 (per Mpx^2 if uncalibrated) in shells outside/inside the matrix."""
    out = {}
    dens_unit = "per_mm2" if U == "um" else "per_Mpx2"
    d_px = cells["dist_to_matrix_edge_px"].to_numpy()
    for lo, hi in zip(outside_edges[:-1], outside_edges[1:]):
        area_px = float(((signed > lo / s) & (signed <= hi / s) & tissue).sum())
        n = int(((d_px > lo / s) & (d_px <= hi / s)).sum())
        out[f"density_outside_{lo}_{hi}{U}_{dens_unit}"] = n / (area_px * s * s) * 1e6 if area_px else float("nan")
    for lo, hi in zip(inside_edges[:-1], inside_edges[1:]):
        area_px = float(((-signed >= lo / s) & (-signed < hi / s) & matrix).sum())
        n = int(((-d_px >= lo / s) & (-d_px < hi / s) & (d_px < 0)).sum())
        out[f"density_inside_{lo}_{hi}{U}_{dens_unit}"] = n / (area_px * s * s) * 1e6 if area_px else float("nan")
    return out


def capsule_metrics(collagen, matrix, signed, tissue, s, U, search_um, touch_um=3.0):
    """Collagen capsule around the matrix.

    The capsule is the collagen that touches the matrix edge and lies within
    `search_um` of it. Thickness is the local width of that layer (2 x distance
    to the layer edge, sampled along its skeleton), so a thin sliver and a
    40 um capsule give different answers. It is capped by `search_um`: a capsule
    that continues into thick collagenous stroma reads as the search distance.
    Coverage is the share of the matrix edge that has collagen right against it."""
    out = {}
    tissue_area = float(tissue.sum())
    out["collagen_area_pct_of_tissue"] = 100 * float((collagen & tissue).sum()) / tissue_area if tissue_area else float("nan")
    if signed is None or not matrix.any():
        return out
    out["collagen_area_pct_inside_matrix"] = 100 * float((collagen & matrix).sum()) / float(matrix.sum())

    touch_px = max(2.0, touch_um / s)
    zero = {f"capsule_thickness_median_{U}": 0.0, f"capsule_thickness_mean_{U}": 0.0,
            f"capsule_thickness_p90_{U}": 0.0, "capsule_coverage_pct": 0.0}
    layer = collagen & (signed > 0) & (signed <= search_um / s)
    lab, n = ndi.label(layer)
    if n == 0:
        out.update(zero)
        return out
    touching = np.unique(lab[layer & (signed <= touch_px)])
    touching = touching[touching > 0]
    layer = np.isin(lab, touching)
    if not layer.any():
        out.update(zero)
        return out

    thk = local_thickness(layer) * s
    edge = matrix & ~ndi.binary_erosion(matrix)
    d_layer = ndi.distance_transform_edt(~layer)
    out[f"capsule_thickness_median_{U}"] = float(np.median(thk))
    out[f"capsule_thickness_mean_{U}"] = float(np.mean(thk))
    out[f"capsule_thickness_p90_{U}"] = float(np.percentile(thk, 90))
    out["capsule_coverage_pct"] = 100 * float((d_layer[edge] <= touch_px).mean()) if edge.any() else float("nan")
    return out


# -------------------------------------------------------------------- spatial
def clark_evans(cy, cx, area_px):
    """Clark-Evans nearest-neighbour ratio: <1 clustered, ~1 random, >1 regular."""
    n = len(cy)
    if n < 5 or area_px <= 0:
        return float("nan"), float("nan")
    pts = np.column_stack([cy, cx])
    d, _ = cKDTree(pts).query(pts, k=2)
    nn = d[:, 1]
    expected = 0.5 / np.sqrt(n / area_px)
    return float(nn.mean() / expected), float(nn.mean())
