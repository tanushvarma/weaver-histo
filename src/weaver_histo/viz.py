"""Overlays and figures."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy import ndimage as ndi  # noqa: E402
from skimage import filters, morphology  # noqa: E402

STATE_COLOURS = {          # RGB
    "intact": (0, 200, 0),
    "pyknotic": (255, 0, 0),
    "pale": (0, 120, 255),
    "fragment": (255, 220, 0),
}
STATE_IDS = {name: i for i, name in enumerate(STATE_COLOURS)}


def _outline(mask: np.ndarray, width: int) -> np.ndarray:
    edge = mask & ~ndi.binary_erosion(mask)
    if width > 1:
        edge = ndi.binary_dilation(edge, structure=morphology.disk(width - 1))
    return edge


def make_overlay(result, line_width: int = 1, show_matrix=True, show_collagen=True,
                 show_nuclei=True) -> np.ndarray:
    out = result.rgb.astype(np.float32).copy()
    if show_collagen and result.collagen is not None:
        m = result.collagen
        out[m] = 0.6 * out[m] + 0.4 * np.array([0, 255, 255], np.float32)
    out = out.astype(np.uint8)

    if show_matrix and result.matrix.any():
        out[_outline(result.matrix, line_width + 1)] = (255, 0, 255)

    if show_nuclei and result.labels.max() > 0 and len(result.cells):
        lookup = np.full(result.labels.max() + 1, -1, np.int16)
        lookup[result.cells["label"].to_numpy()] = [STATE_IDS[s] for s in result.cells["state"]]
        state_img = lookup[result.labels]
        for name, sid in STATE_IDS.items():
            m = state_img == sid
            if m.any():
                out[_outline(m, line_width)] = STATE_COLOURS[name]
    return out


def density_heatmap(result, sigma_px: float = 30.0, alpha: float = 0.55) -> np.ndarray:
    dens = filters.gaussian((result.labels > 0).astype(np.float32), sigma=sigma_px, preserve_range=True)
    if dens.max() > 0:
        dens = dens / dens.max()
    heat = (plt.get_cmap("magma")(dens)[..., :3] * 255).astype(np.float32)
    base = result.rgb.astype(np.float32)
    out = base.copy()
    t = result.tissue
    out[t] = (1 - alpha) * base[t] + alpha * heat[t]
    return out.astype(np.uint8)


def channel_image(arr: np.ndarray, tissue: np.ndarray | None = None) -> np.ndarray:
    a = arr.astype(np.float32)
    ref = a[tissue] if tissue is not None and tissue.any() else a
    hi = float(np.percentile(ref, 99.5)) or 1.0
    return (255 * (1 - np.clip(a / hi, 0, 1))).astype(np.uint8)  # dark = strong stain


def distance_histogram(result):
    fig, ax = plt.subplots(figsize=(5, 3))
    cells = result.cells
    unit = "um" if result.params.um_per_px else "px"
    col = f"dist_to_matrix_edge_{unit}"
    if len(cells) and cells[col].notna().any():
        whole = cells[cells["state"] != "fragment"]
        ax.hist(whole[col].dropna(), bins=40, color="#4c78a8")
        ax.axvline(0, color="m", lw=1.5)
        ax.set_xlabel(f"distance from matrix edge ({unit}); <0 = inside matrix")
        ax.set_ylabel("nuclei")
    else:
        ax.text(0.5, 0.5, "No matrix / no nuclei", ha="center", va="center")
        ax.set_axis_off()
    fig.tight_layout()
    return fig


def state_bars(result):
    fig, ax = plt.subplots(figsize=(4, 3))
    counts = result.cells["state"].value_counts() if len(result.cells) else {}
    names = list(STATE_COLOURS)
    ax.bar(names, [int(counts.get(n, 0)) for n in names],
           color=[np.array(STATE_COLOURS[n]) / 255 for n in names])
    ax.set_ylabel("objects")
    fig.tight_layout()
    return fig


# ----------------------------------------------------------------- slide mode
def section_overlay(sr, dot_px: int = 2, show_tiles: bool = True, show_matrix: bool = True,
                    show_cells: bool = True, show_collagen: bool = True, only_states=None) -> np.ndarray:
    """Coarse picture of one section with matrix edge, collagen tint, analysed tiles and a dot per
    nucleus coloured by state (same colours as the single-image overlay)."""
    import cv2

    s = sr.region_um_per_px
    out = sr.roi_rgb.astype(np.float32).copy()
    if show_collagen and sr.collagen is not None:
        out[sr.collagen] = 0.6 * out[sr.collagen] + 0.4 * np.array([0, 255, 255], np.float32)
    out = np.ascontiguousarray(out.astype(np.uint8))
    if show_matrix and sr.matrix.any():
        out[_outline(sr.matrix, 2)] = (255, 0, 255)
    if show_tiles:
        for (tx, ty, w, h) in sr.tiles:
            p0 = (int((tx - sr.roi_um[0]) / s), int((ty - sr.roi_um[1]) / s))
            p1 = (int((tx - sr.roi_um[0] + w) / s), int((ty - sr.roi_um[1] + h) / s))
            cv2.rectangle(out, p0, p1, (60, 60, 60), 1)
    if show_cells and len(sr.cells):
        for state, col in STATE_COLOURS.items():
            if only_states is not None and state not in only_states:
                continue
            sub = sr.cells[sr.cells["state"] == state]
            for x, y in zip(((sub["x_um"] - sr.roi_um[0]) / s).to_numpy(), ((sub["y_um"] - sr.roi_um[1]) / s).to_numpy()):
                cv2.circle(out, (int(round(x)), int(round(y))), dot_px, col, -1, cv2.LINE_AA)
    return out


def slide_overview(res, width: int = 2) -> np.ndarray:
    from .sections import draw_sections
    return draw_sections(res.overview_rgb, res.section_outlines, width)


def section_distance_plot(sr):
    fig, ax = plt.subplots(figsize=(5, 3))
    c = sr.cells
    if len(c) and c["dist_to_matrix_edge_um"].notna().any():
        whole = c[c["state"] != "fragment"]
        ax.hist(whole["dist_to_matrix_edge_um"].dropna(), bins=40, color="#4c78a8")
        ax.axvline(0, color="m", lw=1.5)
        ax.set_xlabel("distance from matrix edge (um); <0 = inside matrix")
        ax.set_ylabel("nuclei")
    else:
        ax.text(0.5, 0.5, "No matrix / no nuclei", ha="center", va="center")
        ax.set_axis_off()
    fig.tight_layout()
    return fig
