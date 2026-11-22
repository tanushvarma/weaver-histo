"""Command-line interface: batch analysis without the Streamlit app."""

from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from . import __version__, viz
from .loading import load_image
from .pipeline import Params, analyze
from .readers import open_slide
from .slide import SlideParams, analyze_slide

logger = logging.getLogger("weaver_histo")


def _save_png(arr: np.ndarray, path: Path) -> None:
    Image.fromarray(arr).save(path)


def _load_mask(path: Path, shape: tuple[int, int]) -> np.ndarray:
    mask = Image.open(path).convert("L")
    if mask.size != (shape[1], shape[0]):
        logger.info("Resizing mask %s to %dx%d", path.name, shape[1], shape[0])
        mask = mask.resize((shape[1], shape[0]), Image.NEAREST)
    return np.asarray(mask) > 127


def run_analyze(args: argparse.Namespace) -> None:
    """Analyse single images and write summary.csv, nuclei.csv and one overlay PNG per image."""
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rows, cells = [], []
    for image in args.images:
        path = Path(image)
        logger.info("Analysing %s", path.name)
        rgb, down = load_image(path)
        um = args.um_per_px * down if args.um_per_px else None
        mask = _load_mask(Path(args.mask), rgb.shape[:2]) if args.mask else None
        res = analyze(rgb, Params(stain=args.stain, um_per_px=um), matrix_mask=mask)
        for note in res.notes:
            logger.warning("%s: %s", path.name, note)
        rows.append({"image": path.name, "downscale": down, **res.summary})
        if len(res.cells):
            cells.append(res.cells.assign(image=path.name))
        _save_png(viz.make_overlay(res), out / f"{path.stem}_overlay.png")
    pd.DataFrame(rows).to_csv(out / "summary.csv", index=False)
    (pd.concat(cells, ignore_index=True) if cells else pd.DataFrame()).to_csv(out / "nuclei.csv", index=False)
    logger.info("Wrote %d result(s) to %s", len(rows), out)


def run_slide(args: argparse.Namespace) -> None:
    """Analyse a whole slide and write summary.csv, nuclei.csv, an overview and one overlay per section."""
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    reader = open_slide(args.slide, um_per_px=args.um_per_px, scene_index=args.scene)
    try:
        sp = SlideParams(stain=args.stain, params=Params(stain=args.stain))
        res = analyze_slide(reader, sp, progress=lambda f, msg: logger.info("%3.0f%% %s", 100 * f, msg))
    finally:
        reader.close()
    for note in res.notes:
        logger.warning("%s", note)
    res.summary.to_csv(out / "summary.csv", index=False)
    res.cells.to_csv(out / "nuclei.csv", index=False)
    _save_png(viz.slide_overview(res), out / "overview.png")
    for sr in res.sections:
        _save_png(viz.section_overlay(sr), out / f"section_{sr.id}_overlay.png")
    logger.info("Wrote %d section(s) to %s", len(res.sections), out)


def build_parser() -> argparse.ArgumentParser:
    """Argument parser for the weaver-histo command."""
    parser = argparse.ArgumentParser(prog="weaver-histo", description=__doc__)
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    parser.add_argument("-v", "--verbose", action="store_true", help="log progress and per-image notes")
    sub = parser.add_subparsers(dest="command", required=True)

    a = sub.add_parser("analyze", help="analyse PNG, JPG or TIFF images")
    a.add_argument("images", nargs="+", metavar="IMAGE")
    a.add_argument("--stain", choices=["he", "trichrome"], required=True)
    a.add_argument("--um-per-px", type=float, help="microns per pixel of the input images")
    a.add_argument("--mask", help="hand-drawn hydrogel mask (white = matrix) applied to every image")
    a.add_argument("--out", required=True, help="output directory")
    a.set_defaults(func=run_analyze)

    s = sub.add_parser("slide", help="analyse a whole slide (.vsi, .svs, large TIFF)")
    s.add_argument("slide", metavar="SLIDE")
    s.add_argument("--out", required=True, help="output directory")
    s.add_argument("--stain", choices=["he", "trichrome"], default="he")
    s.add_argument("--um-per-px", type=float, help="override the pixel size stored in the file")
    s.add_argument("--scene", type=int, help="scene index inside the file (default: largest)")
    s.set_defaults(func=run_slide)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point for the weaver-histo command. Returns the process exit code."""
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING, format="%(levelname)s: %(message)s")
    try:
        args.func(args)
    except (OSError, ValueError, RuntimeError, ImportError) as err:
        logger.error("%s", err, exc_info=args.verbose)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
