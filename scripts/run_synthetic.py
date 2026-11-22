"""Run the pipeline on the synthetic scenes and compare against the planted ground truth.

Run from the repository root:  python -m scripts.run_synthetic
"""

import logging
import time

import numpy as np
from tests.synthetic import build_scene

from weaver_histo import Params, analyze

logger = logging.getLogger(__name__)

KEYS = [
    "pyknosis_index_pct",
    "pale_nuclei_pct",
    "fragments_per_100_nuclei",
    "inside_pyknosis_index_pct",
    "outside_pyknosis_index_pct",
    "infiltration_pct_cells_inside",
    "infiltration_depth_median_um",
    "matrix_wall_thickness_median_um",
    "matrix_void_fraction_in_hull_pct",
    "clark_evans_R",
    "nuclei_per_mm2",
]


def iou(a: np.ndarray, b: np.ndarray) -> float:
    return float((a & b).sum() / max((a | b).sum(), 1))


def run(kind: str) -> None:
    img, truth = build_scene(kind)
    res = analyze(img, Params(stain=kind, um_per_px=1.0))
    s = res.summary
    logger.info("===== %s =====", kind)
    for note in res.notes:
        logger.info("note: %s", note)
    logger.info(
        "truth   intact=%d pyk=%d pale=%d frag=%d inside_whole=%d",
        truth["n_intact"],
        truth["n_pyknotic"],
        truth["n_pale"],
        truth["n_fragment"],
        truth["n_inside"],
    )
    logger.info(
        "found   intact=%d pyk=%d pale=%d frag=%d inside_whole=%s",
        s["n_intact"],
        s["n_pyknotic"],
        s["n_pale"],
        s["n_fragments"],
        s.get("n_whole_inside_matrix"),
    )
    logger.info("matrix IoU = %.3f", iou(res.matrix, truth["matrix"]))
    if kind == "trichrome":
        nuclei = res.labels > 0
        logger.info(
            "capsule thickness median/mean/p90 (truth %d): %.1f / %.1f / %.1f  coverage=%.0f%%",
            truth["capsule_px"],
            s["capsule_thickness_median_um"],
            s["capsule_thickness_mean_um"],
            s["capsule_thickness_p90_um"],
            s["capsule_coverage_pct"],
        )
        logger.info("capsule IoU = %.3f", iou(res.collagen & ~nuclei, truth["capsule"] & ~nuclei))
    for key in KEYS:
        if key in s:
            logger.info("  %-36s %.3f", key, s[key])


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    t0 = time.perf_counter()
    run("he")
    run("trichrome")
    logger.info("elapsed %.1fs", time.perf_counter() - t0)


if __name__ == "__main__":
    main()
