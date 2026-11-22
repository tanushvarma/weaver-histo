import pandas as pd
import pytest
from PIL import Image

from synthetic import build_scene
from weaver_histo import __version__
from weaver_histo.cli import main


def test_analyze_writes_outputs(tmp_path):
    img, _ = build_scene("he")
    Image.fromarray(img).save(tmp_path / "scene.png")
    out = tmp_path / "out"
    assert main(["analyze", str(tmp_path / "scene.png"), "--stain", "he", "--um-per-px", "1.0", "--out", str(out)]) == 0
    assert (out / "nuclei.csv").exists()
    assert (out / "scene_overlay.png").exists()
    assert len(pd.read_csv(out / "summary.csv")) == 1


def test_missing_file_fails_cleanly(tmp_path, caplog):
    assert main(["analyze", str(tmp_path / "nope.png"), "--stain", "he", "--out", str(tmp_path)]) == 1
    assert "nope.png" in caplog.text


def test_version(capsys):
    with pytest.raises(SystemExit) as exc:
        main(["--version"])
    assert exc.value.code == 0
    assert __version__ in capsys.readouterr().out
