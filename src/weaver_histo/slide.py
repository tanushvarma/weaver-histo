"""Whole-slide analysis in two scales.

  coarse (~2 um/px)  per section: tissue, hydrogel/matrix (rules or a trained model),
                     collagen, distance-to-matrix map, geometry, capsule
  fine   (~0.5 um/px) per tile: nuclei segmentation and per-nucleus features

Tiles overlap by a margin and a nucleus is kept only by the tile that contains its
centre, so nothing is counted twice or cut in half. Nuclear states are decided on all
nuclei of the slide together (relative to the typical nucleus), then every nucleus is
placed against the coarse matrix map to get inside/outside, depth and shell densities.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import ndimage as ndi
from skimage import filters

from . import metrics, segment, stains
from .pipeline import Params
from .readers import SlideReader
from .sections import Section, find_sections

U = "um"


@dataclass
class SlideParams:
    stain: str = "he"  # "he" | "trichrome"
    overview_max_side: int = 3000  # px, for finding sections
    analysis_um_per_px: float = 0.5  # nuclei scale (use ~0.25-0.5)
    region_um_per_px: float = 2.0  # tissue / matrix / collagen scale
    region_max_side: int = 2400  # px cap on one section's coarse image
    tile_um: float = 1000.0  # tile edge (core, without margin)
    margin_um: float = 20.0  # overlap so cut nuclei are seen whole
    roi_pad_frac: float = 0.08  # pad around each detected section
    max_tiles_per_section: int | None = 24  # None = analyse every tile
    min_tile_tissue_frac: float = 0.03  # skip tiles that are (almost) all glass
    section_merge_frac: float = 0.025  # bridge gaps this wide when grouping tissue
    refine_matrix_edge: bool = True  # snap matrix edge to the half-max of the stain signal
    seed: int = 0
    params: Params = field(default_factory=Params)  # nuclei / state / matrix / capsule settings


@dataclass
class SectionResult:
    id: int
    roi_um: tuple  # (x0, y0, x1, y1) in slide microns
    region_um_per_px: float
    roi_rgb: np.ndarray  # coarse image of the section
    tissue: np.ndarray
    matrix: np.ndarray
    collagen: np.ndarray | None
    signed: np.ndarray | None  # px at region scale, + outside / - inside matrix
    tiles: list  # analysed tile footprints (x_um, y_um, w_um, h_um)
    summary: dict
    cells: pd.DataFrame
    notes: list = field(default_factory=list)


@dataclass
class SlideResult:
    params: SlideParams
    scene_info: str
    overview_rgb: np.ndarray
    overview_um_per_px: float
    sections: list  # [SectionResult]
    section_outlines: list  # [Section] on the overview
    summary: pd.DataFrame
    cells: pd.DataFrame
    notes: list = field(default_factory=list)


Progress = Callable[[float, str], None] | None


def _px(v_um: float, s: float) -> int:
    return int(round(v_um / s))


def _proxy_nuclei(signal, tissue, floor, scale, s):
    # Coarse nuclei mask, only used to keep nuclei from biasing the matrix and collagen masks.
    sm = ndi.gaussian_filter(signal, max(0.7, 1.0 / s))
    v = sm[tissue]
    if v.size < 100:
        return np.zeros(signal.shape, bool)
    try:
        thr = max(float(filters.threshold_otsu(v)) * scale, floor)
    except ValueError:
        thr = floor
    return (sm > thr) & tissue


def _region_masks(roi, s, sp: SlideParams, region_model, notes):
    p = sp.params
    tissue, tissue_thr = segment.tissue_mask(roi, p.tissue_threshold, s)
    sig, second, matrix_sig, floor = stains.separate(roi, sp.stain, p.tri_blue_rgb, p.tri_red_rgb)
    nuc = _proxy_nuclei(sig, tissue, floor, p.nuclei_thresh_scale, s)

    matrix_thr, source = None, "auto"
    if region_model is not None and region_model.is_trained:
        cls = region_model.predict(roi)
        mnum = region_model.class_number("matrix")
        if mnum is None:
            raise ValueError("The region model has no class called 'matrix'.")
        matrix = cls == mnum
        bnum = region_model.class_number("background")
        if bnum is not None:
            tissue = segment.remove_small(cls != bnum, 500.0 / (s * s))
        matrix = segment.remove_small(matrix, max(30.0, p.matrix_min_area_um2 / (s * s) / 4))
        # no hole filling here: nuclei are painted out before classification, and filling would
        # also fill the host-tissue gaps between hydrogel arms
        source = "trained_model"
    else:
        matrix, matrix_thr = segment.matrix_from_signal(
            matrix_sig,
            tissue,
            nuc,
            smooth_px=max(2.0, p.matrix_smooth_um / s),
            thresh=p.matrix_thresh,
            open_px=max(1, int(round(p.matrix_open_um / s))),
            min_area_px=max(300.0, p.matrix_min_area_um2 / (s * s)),
            hole_px=max(50.0, p.matrix_hole_um2 / (s * s)),
        )
        if not matrix.any():
            notes.append(
                "No hydrogel/matrix found automatically in this section; infiltration and capsule "
                "metrics are blank. Teach the region model or adjust the matrix threshold."
            )
    if sp.refine_matrix_edge and matrix.any():
        for _ in range(4):  # several passes: a pass can move the edge by one band width
            matrix = segment.refine_edge(
                matrix_sig, matrix, tissue, nuc, s, min_area_px=max(30.0, p.matrix_min_area_um2 / (s * s) / 4)
            )
    collagen, collagen_thr = None, None
    if sp.stain == "trichrome":
        collagen, collagen_thr = segment.collagen_mask(second, tissue, nuc, p.collagen_thresh_scale, s=s)
    return dict(
        tissue=tissue,
        tissue_thr=tissue_thr,
        matrix=matrix,
        matrix_thr=matrix_thr,
        matrix_source=source,
        collagen=collagen,
        collagen_thr=collagen_thr,
    )


def _select_tiles(roi_um, s, tissue, sp: SlideParams, section_id: int):
    x0, y0, x1, y1 = roi_um
    tiles = []
    ty = y0
    while ty < y1 - 1e-6:
        h = min(sp.tile_um, y1 - ty)
        tx = x0
        while tx < x1 - 1e-6:
            w = min(sp.tile_um, x1 - tx)
            ry0, ry1 = int((ty - y0) / s), max(int((ty - y0) / s) + 1, int(np.ceil((ty - y0 + h) / s)))
            rx0, rx1 = int((tx - x0) / s), max(int((tx - x0) / s) + 1, int(np.ceil((tx - x0 + w) / s)))
            cov = float(tissue[ry0:ry1, rx0:rx1].mean()) if tissue[ry0:ry1, rx0:rx1].size else 0.0
            if cov >= sp.min_tile_tissue_frac:
                tiles.append((tx, ty, w, h))
            tx += sp.tile_um
        ty += sp.tile_um
    n_total = len(tiles)
    if sp.max_tiles_per_section and n_total > sp.max_tiles_per_section:
        rng = np.random.default_rng(sp.seed + section_id)
        keep = np.sort(rng.choice(n_total, sp.max_tiles_per_section, replace=False))
        tiles = [tiles[i] for i in keep]
    return tiles, n_total


def _read_tile(reader: SlideReader, tile, sp: SlideParams):
    tx, ty, w, h = tile
    m = sp.margin_um
    return reader.read_um(tx - m, ty - m, w + 2 * m, h + 2 * m, sp.analysis_um_per_px)


def _calibrate_nuclei_threshold(tile_imgs, tissue_thr, sp: SlideParams) -> float:
    # One cutoff for the whole slide: Otsu on nuclear signal pooled from sample tiles.
    p = sp.params
    s = sp.analysis_um_per_px
    sigma = max(0.5, 1.0 / s)
    pool, floor = [], 0.3
    rng = np.random.default_rng(sp.seed)
    for rgb in tile_imgs:
        sig, _, _, floor = stains.separate(rgb, sp.stain, p.tri_blue_rgb, p.tri_red_rgb)
        tissue, _ = segment.tissue_mask(rgb, tissue_thr, s)
        sm = ndi.gaussian_filter(sig, sigma)
        v = sm[tissue]
        if v.size:
            pool.append(v[rng.choice(v.size, min(v.size, 200_000), replace=False)])
    floor = p.nuclei_floor if p.nuclei_floor is not None else floor
    if not pool:
        return floor
    vals = np.concatenate(pool)
    try:
        otsu = float(filters.threshold_otsu(vals))
    except ValueError:
        otsu = floor
    return max(otsu * p.nuclei_thresh_scale, floor)


def _analyze_tile(rgb, tile, sp: SlideParams, tissue_thr, nuc_thr):
    # Returns (cells with slide-micron centres, core tissue area in um2).
    p = sp.params
    s = sp.analysis_um_per_px
    tx, ty, w, h = tile
    m_px = _px(sp.margin_um, s)
    sig, _, _, floor = stains.separate(rgb, sp.stain, p.tri_blue_rgb, p.tri_red_rgb)
    tissue, _ = segment.tissue_mask(rgb, tissue_thr, s)
    labels, _ = segment.segment_nuclei(
        sig, tissue, 1.0, floor, s, p.min_object_um2, split=True, split_h_um=p.nucleus_split_h_um, threshold=nuc_thr
    )
    cells = metrics.nucleus_table(labels, sig)
    cw, ch = _px(w, s), _px(h, s)
    core_tissue = float(tissue[m_px : m_px + ch, m_px : m_px + cw].sum()) * s * s
    if len(cells):
        inside = (cells["cx"] >= m_px) & (cells["cx"] < m_px + cw) & (cells["cy"] >= m_px) & (cells["cy"] < m_px + ch)
        cells = cells[inside].copy()
        cells["x_um"] = tx + (cells["cx"] - m_px + 0.5) * s
        cells["y_um"] = ty + (cells["cy"] - m_px + 0.5) * s
    else:
        cells["x_um"], cells["y_um"] = [], []
    return cells.drop(columns=["label", "cy", "cx"], errors="ignore"), core_tissue


def analyze_slide(
    reader: SlideReader, sp: SlideParams, region_model=None, state_model=None, progress: Progress = None
) -> SlideResult:
    if not reader.um_per_px:
        raise ValueError("Pixel size unknown: set microns-per-pixel for this slide.")
    p = sp.params
    notes: list[str] = []

    def prog(f, msg) -> None:
        if progress:
            progress(float(min(max(f, 0.0), 1.0)), msg)

    prog(0.0, "Reading overview and finding tissue sections...")
    ov, ov_um = reader.overview(sp.overview_max_side)
    sections = find_sections(ov, ov_um, merge_frac=sp.section_merge_frac, stain=sp.stain)
    if not sections:
        raise ValueError("No tissue sections found on the overview. Check the stain choice, or crop to the tissue.")
    s_an = sp.analysis_um_per_px

    plans = []
    for k, sec in enumerate(sections):
        prog(0.05 + 0.15 * k / len(sections), f"Section {sec.id}: tissue, matrix and collagen maps...")
        bx0, by0, bx1, by1 = sec.bbox
        x0, y0, x1, y1 = bx0 * ov_um, by0 * ov_um, bx1 * ov_um, by1 * ov_um
        pad = sp.roi_pad_frac * max(x1 - x0, y1 - y0)
        roi = (max(0.0, x0 - pad), max(0.0, y0 - pad), min(reader.width_um, x1 + pad), min(reader.height_um, y1 + pad))
        long_um = max(roi[2] - roi[0], roi[3] - roi[1])
        s_r = max(
            region_model.um_per_px if (region_model is not None and region_model.is_trained) else sp.region_um_per_px,
            long_um / sp.region_max_side,
        )
        sec_notes: list[str] = []
        if region_model is not None and region_model.is_trained and s_r > region_model.um_per_px * 1.05:
            sec_notes.append(
                f"Section is large, so the coarse map was read at {s_r:.2f} um/px "
                f"instead of the model's {region_model.um_per_px:.2f}; accuracy may drop."
            )
        roi_rgb = reader.read_um(roi[0], roi[1], roi[2] - roi[0], roi[3] - roi[1], s_r)
        masks = _region_masks(roi_rgb, s_r, sp, region_model, sec_notes)
        tiles, n_total = _select_tiles(roi, s_r, masks["tissue"], sp, sec.id)
        plans.append(
            dict(sec=sec, roi=roi, s_r=s_r, rgb=roi_rgb, masks=masks, tiles=tiles, n_total=n_total, notes=sec_notes)
        )

    prog(0.2, "Calibrating the nuclei threshold...")
    cal_imgs, cache = [], {}
    for pi, pl in enumerate(plans):
        for ti, tile in enumerate(pl["tiles"][:3]):
            if len(cal_imgs) < 12:
                rgb = _read_tile(reader, tile, sp)
                cache[(pi, ti)] = rgb
                cal_imgs.append(rgb)
    tissue_thr = float(np.median([pl["masks"]["tissue_thr"] for pl in plans]))
    nuc_thr = _calibrate_nuclei_threshold(cal_imgs, tissue_thr, sp)
    del cal_imgs

    total_tiles = max(1, sum(len(pl["tiles"]) for pl in plans))
    done = 0
    for pi, pl in enumerate(plans):
        tables, core_area = [], 0.0
        for ti, tile in enumerate(pl["tiles"]):
            prog(0.25 + 0.65 * done / total_tiles, f"Section {pl['sec'].id}: tile {ti + 1}/{len(pl['tiles'])}...")
            rgb = cache.pop((pi, ti), None)
            if rgb is None:
                rgb = _read_tile(reader, tile, sp)
            cells, a = _analyze_tile(rgb, tile, sp, tissue_thr, nuc_thr)
            cells["tile"] = ti
            tables.append(cells)
            core_area += a
            done += 1
        pl["cells_px"] = pd.concat(tables, ignore_index=True) if tables else pd.DataFrame()
        if len(pl["cells_px"]):
            pl["cells_px"]["_sec"] = pi
        pl["core_tissue_um2"] = core_area

    prog(0.92, "Classifying nuclei and computing metrics...")
    all_px = [pl["cells_px"] for pl in plans if len(pl["cells_px"])]
    state_notes = []
    pooled = pd.DataFrame()
    ref = {"ref_area_px": float("nan"), "ref_signal": float("nan")}
    if all_px:
        pooled = pd.concat(all_px, ignore_index=True)
        ref_px = p.ref_nucleus_area_um2 / (s_an * s_an) if p.ref_nucleus_area_um2 else None
        # one reference ("typical") nucleus for the whole slide, so sections are judged by the same yardstick
        pooled, ref = metrics.classify_nuclei(pooled, p, ref_area_px=ref_px)
        if state_model is not None and state_model.is_trained:
            pooled["state"], pooled["state_confidence"] = state_model.predict(pooled)
    else:
        state_notes.append("No nuclei were found. Check the analysis resolution and stain choice.")
    used_model = state_model is not None and state_model.is_trained
    empty_cols = metrics.nucleus_table(np.zeros((2, 2), np.int32), np.zeros((2, 2))).columns

    results, summaries, all_cells = [], [], []
    for pi, pl in enumerate(plans):
        if len(pooled):
            cells = pooled[pooled["_sec"] == pi].reset_index(drop=True)
        else:
            cells = pd.DataFrame(columns=[*empty_cols, "x_um", "y_um"])
        if not len(cells):
            cells = metrics.classify_nuclei(pd.DataFrame(columns=empty_cols), p)[0]
            cells["x_um"], cells["y_um"] = [], []
        res = _finish_section(pl["sec"], pl, cells, ref, sp, used_model, pl["notes"] + list(state_notes))
        results.append(res)
        summaries.append(res.summary)
        if len(res.cells):
            all_cells.append(res.cells)
    prog(1.0, "Done")

    scene = getattr(reader, "scene_name", "") or reader.path.name
    return SlideResult(
        params=sp,
        scene_info=f"{scene}: {reader.width_px}x{reader.height_px} px at {reader.um_per_px:.3f} um/px",
        overview_rgb=ov,
        overview_um_per_px=ov_um,
        sections=results,
        section_outlines=sections,
        summary=pd.DataFrame(summaries),
        cells=pd.concat(all_cells, ignore_index=True) if all_cells else pd.DataFrame(),
        notes=notes,
    )


def _finish_section(sec: Section, pl, cells: pd.DataFrame, ref, sp: SlideParams, used_state_model: bool, notes):
    p = sp.params
    s_r, s_an = pl["s_r"], sp.analysis_um_per_px
    roi, m = pl["roi"], pl["masks"]
    tissue, matrix, collagen = m["tissue"], m["matrix"], m["collagen"]

    # footprint of analysed tiles at coarse scale -> area terms refer to the sampled part only
    sampled = np.zeros(tissue.shape, bool)
    for tx, ty, w, h in pl["tiles"]:
        ry0, rx0 = int((ty - roi[1]) / s_r), int((tx - roi[0]) / s_r)
        ry1, rx1 = int(np.ceil((ty - roi[1] + h) / s_r)), int(np.ceil((tx - roi[0] + w) / s_r))
        sampled[ry0:ry1, rx0:rx1] = True
    tissue_eff, matrix_eff = tissue & sampled, matrix & sampled
    sampled_tissue_um2 = float(tissue_eff.sum()) * s_r * s_r

    signed = metrics.signed_distance(matrix)
    if len(cells):
        if signed is not None:
            d_px = metrics.sample_at(signed, (cells["y_um"] - roi[1]) / s_r, (cells["x_um"] - roi[0]) / s_r)
            cells["dist_to_matrix_edge_px"] = d_px
            cells["region"] = np.where(d_px < 0, "inside_matrix", "outside")
        else:
            cells["dist_to_matrix_edge_px"] = np.nan
            cells["region"] = "no_matrix"
    else:
        cells["dist_to_matrix_edge_px"], cells["region"] = [], []

    out = pd.DataFrame(
        {
            "section_id": sec.id,
            "tile": cells.get("tile", pd.Series(dtype=int)).to_numpy(),
            "state": cells["state"].to_numpy(),
            "region": cells["region"].to_numpy(),
            "x_um": cells["x_um"].to_numpy(),
            "y_um": cells["y_um"].to_numpy(),
            "dist_to_matrix_edge_um": cells["dist_to_matrix_edge_px"].to_numpy(float) * s_r,
            "area_um2": cells["area_px"].to_numpy() * s_an * s_an,
            "core_area_um2": cells["core_area_px"].to_numpy() * s_an * s_an,
            "perimeter_um": cells["perimeter_px"].to_numpy() * s_an,
            "major_axis_um": cells["major_axis_px"].to_numpy() * s_an,
            "minor_axis_um": cells["minor_axis_px"].to_numpy() * s_an,
            "eccentricity": cells["eccentricity"].to_numpy(),
            "solidity": cells["solidity"].to_numpy(),
            "circularity": cells["circularity"].to_numpy(),
            "rel_area": cells["rel_area"].to_numpy(),
            "rel_dark": cells["rel_dark"].to_numpy(),
            "mean_signal": cells["mean_signal"].to_numpy(),
            "core_mean_signal": cells["core_mean_signal"].to_numpy(),
            "max_signal": cells["max_signal"].to_numpy(),
            "std_signal": cells["std_signal"].to_numpy(),
        }
    )
    if "state_confidence" in cells:
        out["state_confidence"] = cells["state_confidence"].to_numpy()

    S: dict = {
        "section_id": sec.id,
        "stain": sp.stain,
        "state_source": "trained_model" if used_state_model else "rules",
        "analysis_um_per_px": s_an,
        "region_um_per_px": s_r,
        "matrix_source": m["matrix_source"],
        "tissue_threshold": m["tissue_thr"],
        "matrix_threshold": m["matrix_thr"] if m["matrix_thr"] is not None else float("nan"),
        "roi_x_um": roi[0],
        "roi_y_um": roi[1],
        "roi_w_um": roi[2] - roi[0],
        "roi_h_um": roi[3] - roi[1],
        "tiles_total": pl["n_total"],
        "tiles_analyzed": len(pl["tiles"]),
        "pct_tissue_analyzed": 100 * float((tissue & sampled).sum()) / max(1, int(tissue.sum())),
        "tissue_area_um2": float(tissue.sum()) * s_r * s_r,
        "tissue_area_analyzed_um2": sampled_tissue_um2,
        "n_objects": int(len(out)),
        "nuclei_per_mm2": len(out) / sampled_tissue_um2 * 1e6 if sampled_tissue_um2 else float("nan"),
        "nuclear_area_pct_of_tissue": 100 * float(out["area_um2"].sum()) / sampled_tissue_um2
        if sampled_tissue_um2
        else float("nan"),
        "ref_nucleus_area_um2": ref["ref_area_px"] * s_an * s_an,
    }
    S.update(metrics.state_summary(out))
    if signed is not None and len(out):
        inside = out["region"] == "inside_matrix"
        S.update(metrics.state_summary(out[inside], "inside_"))
        S.update(metrics.state_summary(out[~inside], "outside_"))
        whole = out["state"] != "fragment"
        n_in, n_out = int((inside & whole).sum()), int((~inside & whole).sum())
        S["n_whole_inside_matrix"], S["n_whole_outside_matrix"] = n_in, n_out
        S["infiltration_pct_cells_inside"] = 100 * n_in / (n_in + n_out) if (n_in + n_out) else float("nan")
        depth = -out.loc[inside & whole, "dist_to_matrix_edge_um"].to_numpy()
        S["infiltration_depth_median_um"] = float(np.median(depth)) if depth.size else float("nan")
        S["infiltration_depth_p90_um"] = float(np.percentile(depth, 90)) if depth.size else float("nan")
        S["infiltration_depth_max_um"] = float(depth.max()) if depth.size else float("nan")
        S.update(
            metrics.shell_densities(
                cells, signed, tissue_eff, matrix_eff, s_r, U, p.outside_shells_um, p.inside_shells_um
            )
        )
    if signed is not None:
        S.update(metrics.construct_geometry(matrix, s_r, U))
        S["matrix_area_pct_of_tissue"] = 100 * float(matrix.sum()) / max(1, int(tissue.sum()))
    if sp.stain == "trichrome":
        S["collagen_threshold"] = m["collagen_thr"]
        S.update(metrics.capsule_metrics(collagen, matrix, signed, tissue, s_r, U, p.capsule_search_um))
    ce, nn = metrics.clark_evans(out["y_um"].to_numpy(), out["x_um"].to_numpy(), sampled_tissue_um2)
    S["clark_evans_R"], S["nn_distance_mean_um"] = ce, (nn if np.isfinite(nn) else float("nan"))

    if len(pl["tiles"]) < pl["n_total"]:
        notes.append(
            f"Sampled {len(pl['tiles'])} of {pl['n_total']} tiles. Counts, densities and percentages "
            "describe the sampled tiles; geometry and capsule use the whole section."
        )
    return SectionResult(
        id=sec.id,
        roi_um=roi,
        region_um_per_px=s_r,
        roi_rgb=pl["rgb"],
        tissue=tissue,
        matrix=matrix,
        collagen=collagen,
        signed=signed,
        tiles=pl["tiles"],
        summary=S,
        cells=out,
        notes=notes,
    )
