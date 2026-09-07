"""Assembles the per-object feature table: morphology + intensity + texture.

One correctness point drives the API here.  **Intensity and texture must be
measured on the raw image, not the preprocessed one.**  CLAHE, gamma correction
and percentile normalisation all deliberately destroy the relationship between
pixel value and photon count -- that is what makes them good for segmentation and
useless for quantification.  A "mean intensity" read off a CLAHE-enhanced image
is not a measurement of stain concentration; it is a measurement of local
contrast.

So the pipeline segments on the preprocessed image and measures on the raw one,
and :func:`extract_features` takes the measurement image explicitly rather than
defaulting to whatever was most recently computed.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import ndimage as ndi

from microseg.analysis.morphology import morphology_features
from microseg.analysis.texture import object_glcm_features
from microseg.channels import MicroscopyImage
from microseg.config import AnalysisConfig


def intensity_features(
    labels: np.ndarray, intensity: np.ndarray, prefix: str = ""
) -> pd.DataFrame:
    """Per-object intensity statistics for one channel.

    ``integrated`` (the sum) is the one to reach for when quantifying total
    stain per nucleus -- e.g. DNA content for cell-cycle analysis -- because it
    scales with both concentration and nuclear volume, whereas the mean divides
    the volume back out.
    """
    labels = np.asarray(labels, dtype=np.int32)
    intensity = np.asarray(intensity, dtype=np.float32)
    if labels.max() == 0:
        return pd.DataFrame(columns=["label"])

    # Index the labels that are actually present rather than 1..max.  The two
    # agree for a sequentially relabelled map, but asking scipy about an id with
    # no pixels divides by a zero count and yields NaN features.
    index = np.unique(labels)
    index = index[index != 0]

    # scipy's labelled-statistics helpers internally divide by a per-bin count
    # that is zero for the "not in index" bin whenever the requested labels cover
    # every pixel (a whole-image object).  The returned values for the real
    # labels are correct; only that unused bin produces a 0/0, so the warning is
    # suppressed rather than worked around.
    with np.errstate(invalid="ignore", divide="ignore"):
        means = ndi.mean(intensity, labels, index)
        stds = ndi.standard_deviation(intensity, labels, index)
        mins = ndi.minimum(intensity, labels, index)
        maxs = ndi.maximum(intensity, labels, index)
        sums = ndi.sum_labels(intensity, labels, index)
        medians = ndi.median(intensity, labels, index)

    return pd.DataFrame(
        {
            "label": index,
            f"{prefix}mean_intensity": np.asarray(means, dtype=float),
            f"{prefix}median_intensity": np.asarray(medians, dtype=float),
            f"{prefix}std_intensity": np.asarray(stds, dtype=float),
            f"{prefix}min_intensity": np.asarray(mins, dtype=float),
            f"{prefix}max_intensity": np.asarray(maxs, dtype=float),
            f"{prefix}integrated_intensity": np.asarray(sums, dtype=float),
        }
    )


def extract_features(
    labels: np.ndarray,
    image: MicroscopyImage,
    cfg: AnalysisConfig | None = None,
    image_id: str | None = None,
) -> pd.DataFrame:
    """Full per-object feature table for one field of view.

    Parameters
    ----------
    labels:
        Instance label image (0 = background).
    image:
        The image to *measure* on -- pass the raw/unenhanced one (see module
        docstring).  Every channel is quantified against the same labels, which
        is the point of segmenting one channel in a multiplexed assay.
    cfg:
        Controls texture computation and border-object handling.
    """
    cfg = cfg or AnalysisConfig()
    labels = np.asarray(labels, dtype=np.int32)

    df = morphology_features(labels, image.pixel_size_um)
    if df.empty:
        return df

    for spec in image.channels:
        channel = image.channel(spec.name)
        prefix = f"{spec.name}_" if image.n_channels > 1 else ""
        df = df.merge(intensity_features(labels, channel, prefix), on="label", how="left")

        if cfg.compute_texture:
            texture = pd.DataFrame(
                object_glcm_features(
                    labels,
                    channel,
                    cfg.glcm_distances,
                    cfg.glcm_angles_deg,
                    cfg.glcm_levels,
                )
            )
            if not texture.empty:
                texture = texture.rename(
                    columns={c: f"{prefix}{c}" for c in texture.columns if c != "label"}
                )
                df = df.merge(texture, on="label", how="left")

    if cfg.exclude_border_objects:
        df = df[~df["touches_border"]].reset_index(drop=True)

    df.insert(0, "image_id", image_id or image.name)
    return df


def summarize_image(df: pd.DataFrame, image_id: str | None = None) -> dict[str, float]:
    """Collapse a per-object table into one row of image-level statistics.

    This is the output an assay actually consumes: a count and a few
    distribution summaries per field, ready to join against plate metadata.
    """
    if df is None or df.empty:
        return {"image_id": image_id, "n_objects": 0}

    summary: dict[str, float] = {
        "image_id": image_id or (df["image_id"].iloc[0] if "image_id" in df else None),
        "n_objects": int(len(df)),
    }
    for column in ("area_px", "perimeter_crofton_px", "circularity", "eccentricity", "solidity"):
        if column in df:
            values = df[column].to_numpy(dtype=float)
            summary[f"{column}_mean"] = float(np.mean(values))
            summary[f"{column}_std"] = float(np.std(values))
            summary[f"{column}_median"] = float(np.median(values))

    for column in df.columns:
        if column.endswith("mean_intensity") or column.endswith("glcm_contrast"):
            summary[f"{column}_mean"] = float(df[column].to_numpy(dtype=float).mean())

    if "touches_border" in df:
        summary["n_border_objects"] = int(df["touches_border"].sum())
    return summary


def summarize_population(frames: list[pd.DataFrame]) -> pd.DataFrame:
    """Stack per-image summaries into one table, one row per field."""
    rows = [summarize_image(df) for df in frames if df is not None]
    return pd.DataFrame(rows)
