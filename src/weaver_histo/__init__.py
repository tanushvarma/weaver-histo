"""Weaver Lab histology analysis: stain separation, nuclei segmentation,
cell-state, infiltration, fibrosis and construct-geometry metrics."""

__version__ = "0.1.0"

from .pipeline import Params, Result, analyze  # noqa: E402,F401
