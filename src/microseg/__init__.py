"""microseg -- microscopy image analysis & nuclei/cell instance segmentation.

An end-to-end pipeline: raw microscopy image -> preprocessing -> segmentation
(classical or U-Net) -> instance labels -> quantitative morphology, intensity
and texture features.
"""

__version__ = "0.1.0"

from microseg.channels import ChannelSpec, MicroscopyImage

__all__ = ["ChannelSpec", "MicroscopyImage", "__version__"]
