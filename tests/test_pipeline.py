"""Regression tests on synthetic scenes with known ground truth.

Run:  python -m pytest tests -q      (or)      python tests/test_pipeline.py
Tolerances reflect measured behaviour; they are not tuned to be loose.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))

import numpy as np  # noqa: E402
import pytest  # noqa: E402
from PIL import Image  # noqa: E402

from make_synthetic import build_scene  # noqa: E402
from weaver_histo import Params, analyze  # noqa: E402
from weaver_histo.loading import load_image  # noqa: E402


def iou(a, b):
    return float((a & b).sum() / max((a | b).sum(), 1))


@pytest.fixture(scope="module")
def he():
    img, truth = build_scene("he")
    return img, truth, analyze(img, Params(stain="he", um_per_px=1.0))


@pytest.fixture(scope="module")
def tri():
    img, truth = build_scene("trichrome")
    return img, truth, analyze(img, Params(stain="trichrome", um_per_px=1.0))


def test_he_nuclear_states(he):
    _, t, r = he
    S = r.summary
    assert abs(S["n_intact"] - t["n_intact"]) <= 5          # pale ones may land here
    assert abs(S["n_pyknotic"] - t["n_pyknotic"]) <= 1
    assert abs(S["n_fragments"] - t["n_fragment"]) <= 2
    assert S["n_pale"] >= 0.6 * t["n_pale"]                  # pale detection is the weakest class
    assert abs(S["n_whole_inside_matrix"] - t["n_inside"]) <= 3


def test_he_matrix_and_geometry(he):
    _, t, r = he
    assert iou(r.matrix, t["matrix"]) > 0.85
    wall = r.summary["matrix_wall_thickness_median_um"]
    assert 20 <= wall <= 28                                  # true wall = 22


def test_he_regional_pyknosis(he):
    _, t, r = he
    # 5 of 25 whole nuclei inside the matrix are pyknotic (20%), 10 of 130 outside (7.7%)
    assert abs(r.summary["inside_pyknosis_index_pct"] - 20) <= 5
    assert abs(r.summary["outside_pyknosis_index_pct"] - 7.7) <= 2


def test_trichrome_nuclear_states(tri):
    _, t, r = tri
    S = r.summary
    assert abs(S["n_intact"] - t["n_intact"]) <= 3
    assert abs(S["n_pyknotic"] - t["n_pyknotic"]) <= 1
    assert abs(S["n_fragments"] - t["n_fragment"]) <= 1
    assert abs(S["n_whole_inside_matrix"] - t["n_inside"]) <= 2


def test_trichrome_capsule(tri):
    _, t, r = tri
    nuc = r.labels > 0
    assert iou(r.collagen & ~nuc, t["capsule"] & ~nuc) > 0.95
    # 18 px capsule; spiral gaps cap it at ~16 and edge placement costs ~1px/side
    assert 12 <= r.summary["capsule_thickness_median_um"] <= 18
    assert r.summary["capsule_coverage_pct"] > 90


def test_uncalibrated_uses_pixels():
    img, _ = build_scene("he")
    r = analyze(img, Params(stain="he"))
    assert any("Uncalibrated" in n for n in r.notes)
    assert "matrix_area_px2" in r.summary and "matrix_area_um2" not in r.summary
    assert "nuclei_per_Mpx2" in r.summary and "nuclei_per_mm2" not in r.summary
    assert np.isnan(r.summary["um_per_px"])


def test_resolution_invariance_in_microns():
    """Same specimen sampled twice as finely (2x upsample, 0.5 um/px) must give
    the same physical answers: lengths/areas are in um, not pixels."""
    import cv2
    img, _ = build_scene("he")
    up = cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTER_LINEAR)
    a = analyze(img, Params(stain="he", um_per_px=1.0)).summary
    b = analyze(up, Params(stain="he", um_per_px=0.5)).summary
    assert b["matrix_area_um2"] == pytest.approx(a["matrix_area_um2"], rel=0.05)
    assert b["matrix_wall_thickness_median_um"] == pytest.approx(
        a["matrix_wall_thickness_median_um"], rel=0.10)
    assert b["tissue_area_um2"] == pytest.approx(a["tissue_area_um2"], rel=0.03)
    assert abs(b["n_whole_nuclei"] - a["n_whole_nuclei"]) <= 0.05 * a["n_whole_nuclei"]
    assert abs(b["n_pyknotic"] - a["n_pyknotic"]) <= 2


def test_no_nuclei_does_not_crash():
    img = np.full((300, 300, 3), (246, 243, 246), np.uint8)
    yy, xx = np.mgrid[:300, :300]
    img[((yy - 150) ** 2 + (xx - 150) ** 2) < 100 ** 2] = (226, 165, 200)
    for stain in ("he", "trichrome"):
        r = analyze(img, Params(stain=stain, um_per_px=1.0))
        assert r.summary["n_objects"] == 0
        assert np.isnan(r.summary["pyknosis_index_pct"])


def test_user_mask_and_shape_check(he):
    img, t, _ = he
    r = analyze(img, Params(stain="he", um_per_px=1.0), matrix_mask=t["matrix"])
    assert r.summary["matrix_source"] == "user_mask"
    assert abs(r.summary["n_whole_inside_matrix"] - t["n_inside"]) <= 2
    with pytest.raises(ValueError):
        analyze(img, Params(stain="he"), matrix_mask=t["matrix"][:-1])


@pytest.mark.parametrize("norm", ["macenko", "reinhard"])
def test_normalization_paths_run(norm):
    img, _ = build_scene("he")
    target = None
    if norm == "reinhard":
        from weaver_histo import stains
        target = stains.lab_stats(img)
    r = analyze(img, Params(stain="he", um_per_px=1.0, normalization=norm, reinhard_target=target))
    assert r.summary["n_whole_nuclei"] > 100
    assert r.rgb.shape == img.shape


def test_loading_formats_and_downscale(tmp_path):
    rng = np.random.default_rng(0)
    rgb = rng.integers(0, 255, (60, 80, 3), dtype=np.uint8)
    Image.fromarray(rgb).save(tmp_path / "a.png")
    out, ds = load_image(tmp_path / "a.png")
    assert out.shape == (60, 80, 3) and out.dtype == np.uint8 and ds == 1.0

    import tifffile
    g16 = rng.integers(0, 65535, (50, 70), dtype=np.uint16)
    tifffile.imwrite(tmp_path / "g.tif", g16)
    out, _ = load_image(tmp_path / "g.tif")
    assert out.shape == (50, 70, 3) and out.dtype == np.uint8

    rgb16 = rng.integers(0, 65535, (40, 40, 3), dtype=np.uint16)
    tifffile.imwrite(tmp_path / "c.tif", rgb16)
    out, _ = load_image(tmp_path / "c.tif")
    assert out.shape == (40, 40, 3) and out.dtype == np.uint8

    big = rng.integers(0, 255, (400, 800, 3), dtype=np.uint8)
    Image.fromarray(big).save(tmp_path / "big.png")
    out, ds = load_image(tmp_path / "big.png", max_side=200)
    assert max(out.shape[:2]) == 200 and ds == pytest.approx(4.0, rel=0.01)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
