"""Per-object shape features.

These are the numbers the segmentation exists to produce.  Each is included
because it answers a biological question, not because scikit-image offers it:

* **area** -- nuclear size; tracks ploidy and cell-cycle stage (S/G2 nuclei are
  measurably larger than G1).
* **perimeter** and **circularity** -- membrane irregularity.  Malignant nuclei
  are characteristically lobulated, so circularity drops well below 1.
* **eccentricity** and **aspect ratio** -- elongation, which distinguishes
  fibroblast-like from epithelial morphology.
* **solidity** (area / convex area) -- concavity.  A dip in solidity is the
  classic signature of two nuclei that the segmentation merged, so it doubles as
  a quality-control flag.
* **equivalent diameter** -- an interpretable size in the same units a
  microscopist reads off an eyepiece.

Perimeter deserves a note: the naive pixel-boundary count overestimates by up to
~11% on a digitised circle, because a staircase is longer than the curve it
approximates.  scikit-image's Crofton perimeter is used instead, which corrects
this, and circularity is computed from it -- otherwise perfectly round nuclei
would score ~0.8 rather than ~1.0.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
from skimage.measure import regionprops_table

# Shape properties read straight from regionprops.
_BASE_PROPS = (
    "label",
    "centroid",
    "area",
    "area_convex",
    "area_filled",
    "perimeter",
    "perimeter_crofton",
    "axis_major_length",
    "axis_minor_length",
    "eccentricity",
    "orientation",
    "solidity",
    "extent",
    "euler_number",
    "equivalent_diameter_area",
    "bbox",
)


def morphology_features(
    labels: np.ndarray, pixel_size_um: float | None = None
) -> pd.DataFrame:
    """One row of shape features per instance.

    When ``pixel_size_um`` is known, physical-unit columns are added alongside
    the pixel ones rather than replacing them, so results stay comparable across
    datasets with different calibration.
    """
    labels = np.asarray(labels, dtype=np.int32)
    if labels.max() == 0:
        return pd.DataFrame(columns=["label"])

    table = regionprops_table(labels, properties=_BASE_PROPS)
    df = pd.DataFrame(table).rename(
        columns={
            "centroid-0": "centroid_y",
            "centroid-1": "centroid_x",
            "area": "area_px",
            "area_convex": "convex_area_px",
            "area_filled": "filled_area_px",
            "perimeter": "perimeter_px",
            "perimeter_crofton": "perimeter_crofton_px",
            "axis_major_length": "major_axis_px",
            "axis_minor_length": "minor_axis_px",
            "equivalent_diameter_area": "equivalent_diameter_px",
            "bbox-0": "bbox_min_row",
            "bbox-1": "bbox_min_col",
            "bbox-2": "bbox_max_row",
            "bbox-3": "bbox_max_col",
        }
    )

    # Circularity: 1.0 for a perfect circle, lower for irregular outlines.
    perimeter = df["perimeter_crofton_px"].to_numpy(dtype=float)
    area = df["area_px"].to_numpy(dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        circularity = np.where(perimeter > 0, 4.0 * math.pi * area / perimeter**2, 0.0)
    # Discretisation can nudge a near-circle just past 1; clip so the column
    # stays interpretable as "fraction of ideal roundness".
    df["circularity"] = np.clip(circularity, 0.0, 1.0)

    major = df["major_axis_px"].to_numpy(dtype=float)
    minor = df["minor_axis_px"].to_numpy(dtype=float)
    df["aspect_ratio"] = np.where(minor > 0, major / np.maximum(minor, 1e-8), 0.0)
    df["orientation_deg"] = np.degrees(df["orientation"].to_numpy(dtype=float))

    # A filled area larger than the measured area means the object had internal
    # holes -- usually nucleoli, occasionally a segmentation defect.
    df["hole_area_px"] = df["filled_area_px"] - df["area_px"]

    df["touches_border"] = (
        (df["bbox_min_row"] == 0)
        | (df["bbox_min_col"] == 0)
        | (df["bbox_max_row"] == labels.shape[0])
        | (df["bbox_max_col"] == labels.shape[1])
    )

    if pixel_size_um:
        scale = float(pixel_size_um)
        df["area_um2"] = df["area_px"] * scale**2
        df["perimeter_um"] = df["perimeter_crofton_px"] * scale
        df["equivalent_diameter_um"] = df["equivalent_diameter_px"] * scale
        df["major_axis_um"] = df["major_axis_px"] * scale
        df["minor_axis_um"] = df["minor_axis_px"] * scale

    return df
