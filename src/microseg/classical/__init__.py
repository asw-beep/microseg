"""Classical (non-learned) segmentation baseline."""

from microseg.classical.segment import (
    ClassicalParams,
    classical_instances,
    classical_segment,
    morphological_cleanup,
    threshold_image,
    watershed_split,
)

__all__ = [
    "ClassicalParams",
    "classical_instances",
    "classical_segment",
    "morphological_cleanup",
    "threshold_image",
    "watershed_split",
]
