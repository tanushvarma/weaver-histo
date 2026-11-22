"""Write the synthetic H&E and trichrome test scenes as PNGs.

Run from the repository root:  python -m scripts.make_synthetic [OUT_DIR]
"""

import argparse
import logging
from pathlib import Path

from PIL import Image
from tests.synthetic import build_scene

logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out", nargs="?", default=".", type=Path)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args.out.mkdir(parents=True, exist_ok=True)
    for kind in ("he", "trichrome"):
        img, truth = build_scene(kind)
        path = args.out / f"synthetic_{kind}.png"
        Image.fromarray(img).save(path)
        counts = {k: v for k, v in truth.items() if k.startswith("n_")}
        logger.info("%s: %s %s", kind, path, counts)


if __name__ == "__main__":
    main()
