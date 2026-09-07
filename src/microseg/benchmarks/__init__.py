"""Comparisons against external segmentation methods.

Both modules here are optional: they import their heavy dependency lazily and
report a clear "not installed" status rather than failing, so the core pipeline
and its tests never depend on them.
"""

from microseg.benchmarks.cellpose_compare import (
    CELLPOSE_AVAILABLE,
    cellpose_segment,
    run_cellpose_benchmark,
)
from microseg.benchmarks.sam_experiment import (
    SAM_AVAILABLE,
    run_sam_experiment,
    sam_automatic_masks,
    sam_prompted_masks,
)

__all__ = [
    "CELLPOSE_AVAILABLE",
    "SAM_AVAILABLE",
    "cellpose_segment",
    "run_cellpose_benchmark",
    "run_sam_experiment",
    "sam_automatic_masks",
    "sam_prompted_masks",
]
