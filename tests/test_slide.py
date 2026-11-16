"""Whole-slide mode: readers, section finder, tile pipeline, trainable models.

All on synthetic slides with known ground truth (planted nuclei, spiral hydrogel).
Run:  python -m pytest tests -q        (the slide tests take a few minutes)
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402
from PIL import Image  # noqa: E402
from scipy import ndimage as ndi  # noqa: E402
from scipy.spatial import cKDTree  # noqa: E402

from make_synthetic import build_slide  # noqa: E402
from weaver_histo import ml, segment  # noqa: E402
from weaver_histo.readers import FlatReader, open_slide  # noqa: E402
from weaver_histo.sections import find_sections  # noqa: E402
from weaver_histo.slide import SlideParams, _region_masks, analyze_slide  # noqa: E402

UM = 0.5


@pytest.fixture(scope="module")
def slide(tmp_path_factory):
    img, truth = build_slide("he", n_sections=2)
    path = tmp_path_factory.mktemp("slides") / "slide_he.tif"
    Image.fromarray(img).save(path)
    reader = FlatReader(path, UM)
    res = analyze_slide(reader, SlideParams(stain="he", tile_um=450.0, max_tiles_per_section=None))
    return path, img, truth, reader, res


def truth_matrix_roi(truth, reader, roi_um, shape):
    full = np.zeros((reader.height_px, reader.width_px), np.uint8)
    for t in truth["sections"]:
        m, x, y0 = t["matrix_full"]
        full[y0:y0 + m.shape[0], x:x + m.shape[1]] |= m.astype(np.uint8)
    x0, y0, x1, y1 = [v / UM for v in roi_um]
    crop = full[int(y0):int(y1), int(x0):int(x1)]
    return cv2.resize(crop, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST) > 0


# ------------------------------------------------------------------- readers
def test_flat_reader_geometry_and_padding(slide):
    _, img, _, reader, _ = slide
    # 1:1 read equals the array slice
    got = reader.read_um(100, 50, 200, 100, UM)
    assert got.shape == (200, 400, 3)
    np.testing.assert_array_equal(got, img[100:300, 200:600])
    # 4x downscale keeps the physical field of view
    assert reader.read_um(100, 50, 200, 100, 2.0).shape == (50, 100, 3)
    # off-slide parts are white glass
    pad = reader.read_um(-50, -50, 100, 100, UM)
    assert pad.shape == (200, 200, 3) and (pad[:90, :90] == 255).all()
    # no pixel size -> clear error
    with pytest.raises(ValueError):
        FlatReader(slide[0]).read_um(0, 0, 10, 10, 1.0)


def test_slideio_gdal_reader_matches_flat(slide):
    pytest.importorskip("slideio")
    from weaver_histo.readers import SlideioReader
    path, _, _, flat, _ = slide
    rd = SlideioReader(path, um_per_px=UM, driver="GDAL")
    assert (rd.width_px, rd.height_px) == (flat.width_px, flat.height_px)
    a = rd.read_um(200, 100, 600, 400, 1.0).astype(float)
    b = flat.read_um(200, 100, 600, 400, 1.0).astype(float)
    assert a.shape == b.shape and np.abs(a - b).mean() < 3.0
    assert open_slide(path, um_per_px=UM).width_px == flat.width_px


# ------------------------------------------------------------------ sections
def test_sections_found_on_slide(slide):
    _, _, truth, _, res = slide
    assert len(res.sections) == 2
    for sr, t in zip(res.sections, truth["sections"]):
        m, mx, my = t["matrix_full"]                       # the hydrogel must lie inside the section's ROI
        ys, xs = np.nonzero(m)
        hx0, hx1 = (mx + xs.min()) * UM, (mx + xs.max()) * UM
        hy0, hy1 = (my + ys.min()) * UM, (my + ys.max()) * UM
        x0, y0, x1, y1 = sr.roi_um
        assert x0 <= hx0 and x1 >= hx1 and y0 <= hy0 and y1 >= hy1


def test_label_card_smudge_is_not_a_section():
    img = np.full((375, 991, 3), 241, np.uint8)
    img[20:340, 10:215] = (200, 198, 205)                 # grey-lavender paper, hue ~0.70
    img[40:300, 400:520] = (226, 165, 200)                # real H&E-coloured tissue
    secs = find_sections(img, stain="he")
    assert len(secs) == 1 and secs[0].bbox[0] >= 380


# ---------------------------------------------------------- tile pipeline
def test_nuclei_counts_across_tile_seams(slide):
    _, _, truth, _, res = slide
    for sr, t in zip(res.sections, truth["sections"]):
        S = sr.summary
        assert S["tiles_analyzed"] == S["tiles_total"] > 1               # really tiled
        assert abs(S["n_whole_nuclei"] - t["n_whole"]) <= 3              # no double counts, none lost
        assert abs(S["n_pyknotic"] - t["n_pyknotic"]) <= 1
        assert abs(S["n_fragments"] - t["n_fragment"]) <= 1
        assert S["n_pale"] >= 0.6 * t["n_pale"]                          # pale = weakest class
        assert abs(S["pyknosis_index_pct"] - t["pyk_idx_pct"]) <= 1.5


def test_infiltration_and_geometry(slide):
    _, _, truth, _, res = slide
    for sr, t in zip(res.sections, truth["sections"]):
        S = sr.summary
        assert abs(S["infiltration_pct_cells_inside"] - t["infiltration_pct"]) <= 6   # rule-based mask
        assert abs(S["matrix_area_um2"] / t["matrix_area_um2"] - 1) <= 0.12
        assert 20 <= S["matrix_wall_thickness_median_um"] <= 26                      # true 23.4
        assert 0 < S["nuclei_per_mm2"] < 5000 and S["pct_tissue_analyzed"] > 90


def test_slide_wide_reference_nucleus(slide):
    """Both sections are judged against the same typical nucleus."""
    _, _, _, _, res = slide
    refs = res.summary["ref_nucleus_area_um2"].to_numpy()
    assert np.ptp(refs) == 0 and refs[0] > 0


def test_tile_size_does_not_change_results(tmp_path):
    img, truth = build_slide("he", n_sections=1)
    p = tmp_path / "one.tif"
    Image.fromarray(img).save(p)
    rd = FlatReader(p, UM)
    a = analyze_slide(rd, SlideParams(stain="he", tile_um=400.0, max_tiles_per_section=None)).summary.iloc[0]
    b = analyze_slide(rd, SlideParams(stain="he", tile_um=1500.0, max_tiles_per_section=None)).summary.iloc[0]
    assert a["tiles_analyzed"] > b["tiles_analyzed"]
    assert abs(a["n_whole_nuclei"] - b["n_whole_nuclei"]) <= 2
    assert abs(a["n_pyknotic"] - b["n_pyknotic"]) <= 1


def test_sampling_reports_what_it_did(tmp_path):
    img, _ = build_slide("he", n_sections=1)
    p = tmp_path / "one.tif"
    Image.fromarray(img).save(p)
    res = analyze_slide(FlatReader(p, UM), SlideParams(stain="he", tile_um=400.0, max_tiles_per_section=3))
    S = res.sections[0].summary
    assert S["tiles_analyzed"] == 3 < S["tiles_total"] and S["pct_tissue_analyzed"] < 100
    assert any("Sampled 3 of" in n for n in res.sections[0].notes)


def test_errors_are_clear(tmp_path):
    img, _ = build_slide("he", n_sections=1)
    p = tmp_path / "one.tif"
    Image.fromarray(img).save(p)
    with pytest.raises(ValueError, match="Pixel size unknown"):
        analyze_slide(FlatReader(p), SlideParams())
    blank = tmp_path / "blank.png"
    Image.fromarray(np.full((600, 900, 3), 245, np.uint8)).save(blank)
    with pytest.raises(ValueError, match="No tissue"):
        analyze_slide(FlatReader(blank, 1.0), SlideParams())


# ------------------------------------------------------------ trainable models
def _label_by_truth(cells, truth):
    out = np.array([None] * len(cells), dtype=object)
    for sid, t in enumerate(truth["sections"], 1):
        pts = np.array([(x, y) for _, x, y in t["nuclei_um"]])
        kinds = [k for k, _, _ in t["nuclei_um"]]
        idx = np.flatnonzero(cells["section_id"].to_numpy() == sid)
        d, j = cKDTree(pts).query(cells.loc[idx, ["x_um", "y_um"]].to_numpy())
        for i, dd, jj in zip(idx, d, j):
            if dd < 8:
                out[i] = kinds[jj]
    return out


def test_state_classifier_learns_and_persists(slide, tmp_path):
    _, _, truth, _, res = slide
    cells = res.cells.reset_index(drop=True)
    cells["true"] = _label_by_truth(cells, truth)
    cells = cells[cells["true"].notna()]
    rng = np.random.default_rng(0)
    train_idx = []
    for k in ("intact", "pyknotic", "pale", "fragment"):
        pool = np.flatnonzero((cells["section_id"] == 1) & (cells["true"] == k))
        train_idx += list(rng.choice(pool, min(15, len(pool)), replace=False))
    train, test = cells.iloc[train_idx], cells[cells["section_id"] == 2]
    sc = ml.StateClassifier().fit(train, train["true"].to_numpy())
    pred, conf = sc.predict(test)
    assert (pred == test["true"].to_numpy()).mean() >= 0.97          # on a section it never saw
    assert ((conf > 0) & (conf <= 1)).all()
    sc.save(tmp_path / "s.joblib")
    sc2 = ml.StateClassifier.load(tmp_path / "s.joblib")
    assert (sc2.predict(test)[0] == pred).all()
    n_before = len(sc2.train_y)
    sc2.fit(test.head(30), test.head(30)["true"].to_numpy(), keep_previous=True)   # teaching adds, never resets
    assert len(sc2.train_y) == n_before + 30
    with pytest.raises(ValueError):
        ml.StateClassifier().fit(train.head(12), ["intact"] * 12)    # needs two different states


def test_pixel_classifier_learns_hydrogel_from_clicks(slide, tmp_path):
    path, _, truth, reader, res = slide
    s1 = res.sections[0]
    tm1 = truth_matrix_roi(truth, reader, s1.roi_um, s1.roi_rgb.shape[:2])
    rng = np.random.default_rng(0)
    pts = []
    for k, reg in ((1, ~ndi.binary_dilation(s1.tissue, iterations=8)),
                   (2, s1.tissue & ~ndi.binary_dilation(tm1, iterations=3)),
                   (3, ndi.binary_erosion(tm1, iterations=2))):
        ys, xs = np.nonzero(reg)
        sel = rng.choice(len(ys), 12, replace=False)
        pts += [(int(xs[i]), int(ys[i]), k) for i in sel]                  # 36 "clicks"
    pc = ml.PixelClassifier(um_per_px=s1.region_um_per_px, stain="he")
    pc.fit([(s1.roi_rgb, ml.annotations_from_points(s1.roi_rgb.shape, pts, 5))])
    assert pc.oob_accuracy_ > 0.9

    pc.save(tmp_path / "p.joblib")
    pc = ml.PixelClassifier.load(tmp_path / "p.joblib")
    s2 = res.sections[1]                                                    # never seen
    tm2 = truth_matrix_roi(truth, reader, s2.roi_um, s2.roi_rgb.shape[:2])
    m = _region_masks(s2.roi_rgb, s2.region_um_per_px, SlideParams(stain="he"), pc, [])["matrix"]
    iou = (m & tm2).sum() / (m | tm2).sum()
    assert iou > 0.88 and 0.9 < m.sum() / tm2.sum() < 1.12


def test_region_model_is_used_by_the_pipeline(slide):
    path, _, truth, reader, res = slide
    s1 = res.sections[0]
    tm1 = truth_matrix_roi(truth, reader, s1.roi_um, s1.roi_rgb.shape[:2])
    rng = np.random.default_rng(1)
    pts = []
    for k, reg in ((1, ~ndi.binary_dilation(s1.tissue, iterations=8)),
                   (2, s1.tissue & ~ndi.binary_dilation(tm1, iterations=3)),
                   (3, ndi.binary_erosion(tm1, iterations=2))):
        ys, xs = np.nonzero(reg)
        pts += [(int(xs[i]), int(ys[i]), k) for i in rng.choice(len(ys), 12, replace=False)]
    pc = ml.PixelClassifier(um_per_px=s1.region_um_per_px, stain="he")
    pc.fit([(s1.roi_rgb, ml.annotations_from_points(s1.roi_rgb.shape, pts, 5))])
    out = analyze_slide(reader, SlideParams(stain="he", tile_um=900.0, max_tiles_per_section=4), region_model=pc)
    assert (out.summary["matrix_source"] == "trained_model").all()
    assert all(abs(r["infiltration_pct_cells_inside"] - t["infiltration_pct"]) <= 4
               for (_, r), t in zip(out.summary.iterrows(), truth["sections"]))


def test_refine_edge_moves_boundary_to_half_max():
    yy, xx = np.mgrid[:200, :200]
    true = (xx > 70) & (xx < 130)                                           # a 59 px wide band
    sig = ndi.gaussian_filter(np.where(true, 0.2, 1.0).astype(np.float32), 1.5)   # pale inside, strong outside
    tissue = np.ones_like(true)
    fat = (xx > 62) & (xx < 138)                                            # mask 8 px too wide per side
    fixed = segment.refine_edge(sig, fat, tissue, np.zeros_like(true), s=1.0, band_um=12.0)
    assert abs(int(fixed.sum()) - int(true.sum())) < abs(int(fat.sum()) - int(true.sum())) * 0.25
    assert segment.refine_edge(sig, np.zeros_like(true), tissue, np.zeros_like(true), 1.0).sum() == 0
