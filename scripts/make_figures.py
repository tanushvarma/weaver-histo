"""Render docs/images/overview.png from a synthetic trichrome scene (no lab data).

Run from the repository root:  python -m scripts.make_figures
"""

import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch
from PIL import Image
from tests.synthetic import build_scene

from weaver_histo import Params, analyze, viz

logger = logging.getLogger(__name__)

OUT = Path("docs/images/overview.png")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    img, _ = build_scene("trichrome")
    res = analyze(img, Params(stain="trichrome", um_per_px=1.0))
    overlay = viz.make_overlay(res, line_width=2)
    ys, xs = np.nonzero(res.tissue)
    pad = 15
    crop = np.s_[max(ys.min() - pad, 0) : ys.max() + pad, max(xs.min() - pad, 0) : xs.max() + pad]

    fig, axes = plt.subplots(1, 2, figsize=(10, 5.6), dpi=110)
    for ax, panel, title in zip(axes, (img, overlay), ("Synthetic Masson's trichrome", "Detected"), strict=True):
        ax.imshow(panel[crop])
        ax.set_title(title, fontsize=11)
        ax.set_axis_off()
    present = set(res.cells["state"])
    legend = [
        Patch(color=[c / 255 for c in rgb], label=f"{name} nucleus")
        for name, rgb in viz.STATE_COLOURS.items()
        if name in present
    ]
    legend += [Patch(color=(1, 0, 1), label="hydrogel edge"), Patch(color=(0, 1, 1), label="collagen capsule")]
    fig.legend(handles=legend, loc="lower center", ncol=len(legend), frameon=False, fontsize=9)
    fig.tight_layout(rect=(0, 0.07, 1, 0.98))

    OUT.parent.mkdir(parents=True, exist_ok=True)
    tmp = OUT.with_suffix(".tmp.png")
    fig.savefig(tmp)
    plt.close(fig)
    Image.open(tmp).convert("RGB").quantize(colors=128, method=Image.Quantize.MEDIANCUT).save(OUT, optimize=True)
    tmp.unlink()
    logger.info("Wrote %s (%.0f KB)", OUT, OUT.stat().st_size / 1024)


if __name__ == "__main__":
    main()
