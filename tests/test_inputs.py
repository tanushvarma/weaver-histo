import sys

import numpy as np
import pytest
import tifffile
from PIL import Image

from weaver_histo import Params, analyze
from weaver_histo.loading import load_image
from weaver_histo.readers import open_slide


def test_16bit_tiff(tmp_path):
    rgb16 = np.zeros((40, 40, 3), np.uint16)
    rgb16[..., 0] = 65535
    tifffile.imwrite(tmp_path / "a.tif", rgb16)
    out, _ = load_image(tmp_path / "a.tif")
    assert out.dtype == np.uint8
    assert (out[..., 0] == 255).all() and (out[..., 1:] == 0).all()


def test_rgba_png(tmp_path):
    rgba = np.zeros((30, 30, 4), np.uint8)
    rgba[..., 2] = 200
    rgba[..., 3] = 128
    Image.fromarray(rgba, "RGBA").save(tmp_path / "a.png")
    out, _ = load_image(tmp_path / "a.png")
    assert out.shape == (30, 30, 3)
    assert (out[..., 2] == 200).all()


def test_grayscale_png(tmp_path):
    Image.fromarray(np.full((20, 25), 90, np.uint8), "L").save(tmp_path / "g.png")
    out, _ = load_image(tmp_path / "g.png")
    assert out.shape == (20, 25, 3)
    assert (out == 90).all()


@pytest.mark.parametrize("stain", ["he", "trichrome"])
def test_no_tissue(stain):
    r = analyze(np.full((300, 300, 3), 255, np.uint8), Params(stain=stain, um_per_px=1.0))
    assert r.summary["n_objects"] == 0
    assert any("little tissue" in n for n in r.notes)


def test_vsi_without_slideio(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "slideio", None)
    (tmp_path / "s.vsi").write_bytes(b"")
    with pytest.raises(ImportError, match=r'pip install "weaver-histo\[vsi\]"'):
        open_slide(tmp_path / "s.vsi")
