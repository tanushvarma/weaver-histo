"""Run the pipeline on synthetic scenes and compare against ground truth."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))

import numpy as np  # noqa: E402

from make_synthetic import build_scene  # noqa: E402
from weaver_histo import Params, analyze  # noqa: E402


def iou(a, b):
    return float((a & b).sum() / max((a | b).sum(), 1))


def run(kind, **kw):
    img, truth = build_scene(kind, **kw)
    res = analyze(img, Params(stain=kind, um_per_px=1.0))
    S = res.summary
    print(f"\n===== {kind} =====")
    for n in res.notes:
        print("note:", n)
    print(f"truth   intact={truth['n_intact']} pyk={truth['n_pyknotic']} "
          f"pale={truth['n_pale']} frag={truth['n_fragment']} inside_whole={truth['n_inside']}")
    print(f"found   intact={S['n_intact']} pyk={S['n_pyknotic']} pale={S['n_pale']} "
          f"frag={S['n_fragments']} inside_whole={S.get('n_whole_inside_matrix')}")
    print(f"matrix IoU = {iou(res.matrix, truth['matrix']):.3f}")
    if kind == "trichrome":
        print(f"capsule thickness median/mean/p90 (truth {truth['capsule_px']}): "
              f"{S.get('capsule_thickness_median_um'):.1f} / {S.get('capsule_thickness_mean_um'):.1f} / "
              f"{S.get('capsule_thickness_p90_um'):.1f}  coverage={S.get('capsule_coverage_pct'):.0f}%")
        print(f"capsule IoU = {iou(res.collagen & ~(res.labels > 0), truth['capsule'] & ~(res.labels > 0)):.3f}")
    keys = ["pyknosis_index_pct", "pale_nuclei_pct", "fragments_per_100_nuclei",
            "inside_pyknosis_index_pct", "outside_pyknosis_index_pct",
            "infiltration_pct_cells_inside", "infiltration_depth_median_um",
            "matrix_wall_thickness_median_um", "matrix_void_fraction_in_hull_pct",
            "clark_evans_R", "nuclei_per_mm2"]
    for k in keys:
        if k in S:
            print(f"  {k:36s} {S[k]:.3f}")
    return res, truth


if __name__ == "__main__":
    import time
    t0 = time.time()
    run("he")
    run("trichrome")
    print(f"\nelapsed {time.time() - t0:.1f}s")
