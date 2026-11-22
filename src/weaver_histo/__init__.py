"""Weaver Lab histology analysis: stain separation, nuclei segmentation,
cell-state, infiltration, fibrosis and construct-geometry metrics."""

from importlib.metadata import PackageNotFoundError, version

from .pipeline import Params, Result, analyze

try:
    __version__ = version("weaver-histo")
except PackageNotFoundError:
    __version__ = "0.0.0"

__all__ = ["Params", "Result", "__version__", "analyze"]
