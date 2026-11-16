"""End-to-end analysis of one H&E or Masson's trichrome image."""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Optional

import numpy as np
import pandas as pd

from . import metrics, segment, stains


@dataclass
class Params:
    stain: str = "he"                       # "he" | "trichrome"
    um_per_px: Optional[float] = None       # effective (after any downscale); None = uncalibrated
    normalization: str = "none"             # "none" | "macenko" | "reinhard"
    reinhard_target: Optional[tuple] = None  # (mean[3], std[3]) in LAB
    he_vector_source: str = "reference"     # "reference" | "estimate" (per-image Macenko, opt-in)

    tissue_threshold: Optional[float] = None
    nuclei_thresh_scale: float = 0.8        # x Otsu; <1 admits fainter ("pale") nuclei
    nuclei_floor: Optional[float] = None    # default: 0.3 (H&E) / 0.6 (trichrome)
    nucleus_split_h_um: float = 1.0         # watershed split depth
    min_object_um2: float = 6.0             # smallest object kept (fragments)

    ref_nucleus_area_um2: Optional[float] = None  # override typical nucleus core area (px^2 if uncalibrated)
    fragment_rel_area: float = 0.25
    pyknotic_rel_area: float = 0.60
    pyknotic_rel_dark: float = 1.15
    pyknotic_min_solidity: float = 0.90
    pyknotic_max_ecc: float = 0.85
    pale_rel_dark: float = 0.60

    matrix_thresh: Optional[float] = None
    matrix_smooth_um: float = 5.0
    matrix_open_um: float = 3.0
    matrix_min_area_um2: float = 3000.0
    matrix_hole_um2: float = 600.0

    collagen_thresh_scale: float = 1.0
    capsule_search_um: float = 150.0
    outside_shells_um: tuple = (0, 50, 100, 200)
    inside_shells_um: tuple = (0, 25, 50, 100)

    tri_blue_rgb: Optional[tuple] = None    # override aniline-blue reference colour
    tri_red_rgb: Optional[tuple] = None     # override red reference colour


@dataclass
class Result:
    params: Params
    summary: dict
    cells: pd.DataFrame
    rgb: np.ndarray                 # image actually analysed (after normalisation)
    tissue: np.ndarray
    labels: np.ndarray
    matrix: np.ndarray
    collagen: Optional[np.ndarray]
    signal: np.ndarray              # nuclear signal (H conc. or darkness)
    second: np.ndarray              # eosin (H&E) or collagen (trichrome) channel
    signed: Optional[np.ndarray]
    notes: list = field(default_factory=list)


def analyze(rgb: np.ndarray, params: Params, matrix_mask: Optional[np.ndarray] = None) -> Result:
    p = params
    cal = p.um_per_px is not None
    s = float(p.um_per_px) if cal else 1.0
    U = "um" if cal else "px"
    notes: list[str] = []
    if not cal:
        notes.append("Uncalibrated: lengths are in pixels. Enter um/px for micron units.")

    # ---- tissue + optional normalisation
    tissue, tissue_thr = segment.tissue_mask(rgb, p.tissue_threshold, s)
    rgb_n = rgb
    if p.normalization == "macenko":
        if p.stain == "he":
            rgb_n = stains.macenko_normalize(rgb, tissue)
        else:
            notes.append("Macenko normalisation is H&E-only; skipped for trichrome.")
    elif p.normalization == "reinhard":
        if p.reinhard_target is not None:
            rgb_n = stains.reinhard_normalize(rgb, p.reinhard_target, tissue)
        else:
            notes.append("Reinhard needs a reference image; skipped.")
    if rgb_n is not rgb:
        tissue, tissue_thr = segment.tissue_mask(rgb_n, p.tissue_threshold, s)
    if tissue.sum() < 1000:
        notes.append("Very little tissue detected; check the tissue threshold.")

    # ---- stain separation
    if p.stain == "he":
        W = stains.HE_REF_VECTORS
        if p.he_vector_source == "estimate" and p.normalization != "macenko":
            est = stains.estimate_he_vectors(rgb_n)
            if est is None:
                notes.append("Per-image stain estimate unreliable; used reference H&E vectors.")
            else:
                W = est
        conc = stains.deconvolve(rgb_n, W)
        signal, second = conc[..., 0], conc[..., 1]
        matrix_signal = second
        floor = 0.3 if p.nuclei_floor is None else p.nuclei_floor
    else:
        blue = stains.od_vector_from_rgb(p.tri_blue_rgb) if p.tri_blue_rgb else stains.TRI_BLUE
        red = stains.od_vector_from_rgb(p.tri_red_rgb) if p.tri_red_rgb else stains.TRI_RED
        conc = stains.deconvolve(rgb_n, np.stack([blue, red], axis=1))
        collagen_conc, red_conc = conc[..., 0], conc[..., 1]
        signal = stains.dark_score(rgb_n)
        second = collagen_conc
        matrix_signal = collagen_conc + red_conc
        floor = 0.6 if p.nuclei_floor is None else p.nuclei_floor

    # ---- nuclei
    labels, nuc_thr = segment.segment_nuclei(
        signal, tissue, p.nuclei_thresh_scale, floor, s,
        min_object_um2=p.min_object_um2, split=True, split_h_um=p.nucleus_split_h_um)
    cells = metrics.nucleus_table(labels, signal)
    ref_px = p.ref_nucleus_area_um2 / (s * s) if p.ref_nucleus_area_um2 else None
    cells, ref = metrics.classify_nuclei(cells, p, ref_area_px=ref_px)
    nuclei_bin = labels > 0

    # ---- matrix (hydrogel)
    matrix_thr = None
    if matrix_mask is not None:
        if matrix_mask.shape != tissue.shape:
            raise ValueError(f"Matrix mask shape {matrix_mask.shape} != image shape {tissue.shape}")
        matrix = matrix_mask.astype(bool)
    else:
        matrix, matrix_thr = segment.matrix_from_signal(
            matrix_signal, tissue, nuclei_bin,
            smooth_px=max(2.0, p.matrix_smooth_um / s),
            thresh=p.matrix_thresh,
            open_px=max(1, int(round(p.matrix_open_um / s))),
            min_area_px=max(300.0, p.matrix_min_area_um2 / (s * s)),
            hole_px=max(50.0, p.matrix_hole_um2 / (s * s)))
        if not matrix.any():
            notes.append("No matrix/hydrogel region found automatically; "
                         "infiltration and capsule metrics are blank. Adjust the matrix "
                         "threshold or upload a hand-drawn matrix mask.")

    # ---- collagen (trichrome)
    collagen = None
    collagen_thr = None
    if p.stain == "trichrome":
        collagen, collagen_thr = segment.collagen_mask(
            second, tissue, nuclei_bin, p.collagen_thresh_scale, s=s)

    # ---- distances, regions
    signed = metrics.signed_distance(matrix)
    if len(cells):
        if signed is not None:
            cells["dist_to_matrix_edge_px"] = metrics.sample_at(signed, cells["cy"], cells["cx"])
            cells["region"] = np.where(cells["dist_to_matrix_edge_px"] < 0, "inside_matrix", "outside")
        else:
            cells["dist_to_matrix_edge_px"] = np.nan
            cells["region"] = "no_matrix"
    else:
        cells["dist_to_matrix_edge_px"] = []
        cells["region"] = []

    # ---- tidy per-cell table in physical units (built explicitly: in pixel mode a
    # converted name such as perimeter_px equals its source name)
    cells_out = pd.DataFrame({
        "label": cells["label"].to_numpy(),
        "state": cells["state"].to_numpy(),
        "region": cells["region"].to_numpy(),
        f"x_{U}": cells["cx"].to_numpy() * s,
        f"y_{U}": cells["cy"].to_numpy() * s,
        f"area_{U}2": cells["area_px"].to_numpy() * s * s,
        f"core_area_{U}2": cells["core_area_px"].to_numpy() * s * s,
        f"dist_to_matrix_edge_{U}": cells["dist_to_matrix_edge_px"].to_numpy(float) * s,
        f"perimeter_{U}": cells["perimeter_px"].to_numpy() * s,
        f"major_axis_{U}": cells["major_axis_px"].to_numpy() * s,
        f"minor_axis_{U}": cells["minor_axis_px"].to_numpy() * s,
        "eccentricity": cells["eccentricity"].to_numpy(),
        "solidity": cells["solidity"].to_numpy(),
        "circularity": cells["circularity"].to_numpy(),
        "rel_area": cells["rel_area"].to_numpy(),
        "rel_dark": cells["rel_dark"].to_numpy(),
        "mean_signal": cells["mean_signal"].to_numpy(),
        "core_mean_signal": cells["core_mean_signal"].to_numpy(),
        "max_signal": cells["max_signal"].to_numpy(),
        "std_signal": cells["std_signal"].to_numpy(),
    })

    # ---- summary
    dens_unit = "per_mm2" if cal else "per_Mpx2"
    tissue_area_px = float(tissue.sum())
    S: dict = {
        "stain": p.stain, "normalization": p.normalization,
        "um_per_px": p.um_per_px if cal else float("nan"),
        "tissue_threshold": tissue_thr, "nuclei_threshold": nuc_thr,
        "matrix_threshold": matrix_thr if matrix_thr is not None else float("nan"),
        "matrix_source": "user_mask" if matrix_mask is not None else "auto",
        f"tissue_area_{U}2": tissue_area_px * s * s,
        "n_objects": int(len(cells)),
        f"nuclei_{dens_unit}": len(cells) / (tissue_area_px * s * s) * 1e6 if tissue_area_px else float("nan"),
        "nuclear_area_pct_of_tissue": 100 * float(nuclei_bin.sum()) / tissue_area_px if tissue_area_px else float("nan"),
        f"ref_nucleus_area_{U}2": ref["ref_area_px"] * s * s,
    }
    S.update(metrics.state_summary(cells))
    if signed is not None and len(cells):
        inside = cells["region"] == "inside_matrix"
        S.update(metrics.state_summary(cells[inside], "inside_"))
        S.update(metrics.state_summary(cells[~inside], "outside_"))
        whole = cells["state"] != "fragment"
        n_in, n_out = int((inside & whole).sum()), int((~inside & whole).sum())
        S["n_whole_inside_matrix"] = n_in
        S["n_whole_outside_matrix"] = n_out
        S["infiltration_pct_cells_inside"] = 100 * n_in / (n_in + n_out) if (n_in + n_out) else float("nan")
        depth = -cells.loc[inside & whole, "dist_to_matrix_edge_px"].to_numpy() * s
        S[f"infiltration_depth_median_{U}"] = float(np.median(depth)) if depth.size else float("nan")
        S[f"infiltration_depth_p90_{U}"] = float(np.percentile(depth, 90)) if depth.size else float("nan")
        S[f"infiltration_depth_max_{U}"] = float(depth.max()) if depth.size else float("nan")
        S.update(metrics.shell_densities(cells, signed, tissue, matrix, s, U,
                                         p.outside_shells_um, p.inside_shells_um))
    if signed is not None:
        S.update(metrics.construct_geometry(matrix, s, U))
        S["matrix_area_pct_of_tissue"] = 100 * float(matrix.sum()) / tissue_area_px if tissue_area_px else float("nan")
    if p.stain == "trichrome":
        S["collagen_threshold"] = collagen_thr
        S.update(metrics.capsule_metrics(collagen, matrix, signed, tissue, s, U, p.capsule_search_um))
    ce, nn = metrics.clark_evans(cells["cy"].to_numpy(), cells["cx"].to_numpy(), tissue_area_px)
    S["clark_evans_R"] = ce
    S[f"nn_distance_mean_{U}"] = nn * s if np.isfinite(nn) else float("nan")

    return Result(params=p, summary=S, cells=cells_out, rgb=rgb_n, tissue=tissue,
                  labels=labels, matrix=matrix, collagen=collagen, signal=signal,
                  second=second, signed=signed, notes=notes)
