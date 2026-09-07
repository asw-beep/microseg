"""The end-to-end pipeline: raw microscopy image -> quantitative biology.

This is the object that ties the project together, and the one both the CLI and
the Gradio app drive.  It runs the same five stages regardless of which
segmentation backend is selected, so the classical baseline and the U-Net are
directly comparable -- they differ in exactly one step and share the other four:

    raw image
      -> preprocess          (normalise, denoise, flatten, enhance)
      -> semantic segment    (classical threshold  OR  U-Net probabilities)
      -> instance segment    (watershed from seeds)
      -> measure             (morphology, intensity, texture per nucleus)
      -> report              (per-object table + per-image summary)

Timing is captured per stage, because "how long does it take" is a deployment
question that gets asked immediately after "how accurate is it".
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from microseg.analysis import extract_features, summarize_image
from microseg.channels import MicroscopyImage
from microseg.classical import ClassicalParams, classical_instances
from microseg.config import ExperimentConfig
from microseg.instance import probabilities_to_instances, remove_border_objects
from microseg.preprocessing import preprocess
from microseg.qc import QCConfig, QCReport, assess


@dataclass
class PipelineResult:
    """Everything one image produced, from pixels to a feature table."""

    image_id: str
    raw: MicroscopyImage
    preprocessed: MicroscopyImage
    instances: np.ndarray
    features: pd.DataFrame
    summary: dict
    probabilities: np.ndarray | None = None
    # The predicted distance-to-edge map, when the model has a distance head.
    # Kept on the result because it is the most diagnostic thing to look at when
    # a field comes back over- or under-segmented.
    distance: np.ndarray | None = None
    stages: dict[str, np.ndarray] = field(default_factory=dict)
    timings: dict[str, float] = field(default_factory=dict)
    qc: QCReport | None = None

    @property
    def n_objects(self) -> int:
        return int(self.instances.max())

    @property
    def foreground(self) -> np.ndarray:
        return self.instances > 0

    @property
    def trustworthy(self) -> bool:
        """Whether QC passed.  ``True`` when QC was not run."""
        return True if self.qc is None else self.qc.passed


class SegmentationPipeline:
    """Runs the full analysis with a selectable segmentation backend.

    Parameters
    ----------
    cfg:
        The experiment config; supplies preprocessing, post-processing and
        analysis settings.
    backend:
        ``"unet"`` (needs ``model``) or ``"classical"``.
    model:
        A trained U-Net.  Ignored for the classical backend.
    """

    def __init__(
        self,
        cfg: ExperimentConfig | None = None,
        backend: str = "classical",
        model=None,
        device: str = "auto",
        classical_params: ClassicalParams | None = None,
        tta: bool = False,
        run_qc: bool = True,
        qc_config: QCConfig | None = None,
    ) -> None:
        self.cfg = cfg or ExperimentConfig()
        self.backend = backend
        self.model = model
        self.device = device
        self.classical_params = classical_params or ClassicalParams(
            min_object_area=self.cfg.postprocess.min_object_area,
            peak_min_distance=self.cfg.postprocess.peak_min_distance,
        )
        self.tta = tta
        self.run_qc = run_qc
        self.qc_config = qc_config or QCConfig()

        if backend == "unet" and model is None:
            raise ValueError("backend='unet' requires a trained model; pass model=...")
        if backend not in ("unet", "classical"):
            raise ValueError(f"unknown backend {backend!r}; expected unet|classical")

    # ------------------------------------------------------------- constructors
    @classmethod
    def from_checkpoint(
        cls, path: str | Path, device: str = "auto", tta: bool = False
    ) -> SegmentationPipeline:
        """Build a U-Net pipeline from a saved checkpoint.

        The checkpoint's own config is used, so inference preprocessing matches
        training preprocessing exactly -- a mismatch there is a silent and
        expensive source of degraded accuracy.
        """
        from microseg.inference import load_checkpoint

        model, cfg, _ = load_checkpoint(path, device)
        return cls(cfg, backend="unet", model=model, device=device, tta=tta)

    # ------------------------------------------------------------------ stages
    def segment(
        self, image: MicroscopyImage
    ) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None]:
        """Preprocessed image -> (instance labels, class probabilities, distance)."""
        nuclei = image.nuclei()

        if self.backend == "classical":
            return classical_instances(nuclei, self.classical_params), None, None

        from microseg.inference import predict_maps

        prediction = predict_maps(self.model, nuclei, self.device, tta=self.tta)
        instances = probabilities_to_instances(
            prediction.probs, self.cfg.postprocess, distance=prediction.distance
        )
        return instances, prediction.probs, prediction.distance

    def run(
        self,
        image: MicroscopyImage,
        collect_stages: bool = False,
        measure_channels=None,
    ) -> PipelineResult:
        """Run every stage on one image.

        ``measure_channels`` selects which channels are quantified; the default
        (all of them) is what a multiplexed assay wants.  Note that features are
        computed against ``image`` -- the *raw* input -- while segmentation uses
        the preprocessed copy, so intensity values stay physically meaningful.
        """
        timings: dict[str, float] = {}

        start = time.perf_counter()
        pre = preprocess(image, self.cfg.preprocess, channels="nuclei", collect_stages=collect_stages)
        timings["preprocess_s"] = time.perf_counter() - start

        start = time.perf_counter()
        instances, probs, distance = self.segment(pre.image)
        timings["segment_s"] = time.perf_counter() - start

        if self.cfg.analysis.exclude_border_objects:
            instances = remove_border_objects(instances)

        start = time.perf_counter()
        measure_on = image
        if measure_channels is not None:
            indices = image.resolve(measure_channels)
            measure_on = MicroscopyImage(
                image.data[:, :, indices],
                [image.channels[i] for i in indices],
                pixel_size_um=image.pixel_size_um,
                name=image.name,
            )
        features = extract_features(instances, measure_on, self.cfg.analysis, image.name)
        timings["analysis_s"] = time.perf_counter() - start
        timings["total_s"] = sum(timings.values())

        # QC runs on the finished result, so it sees exactly what the caller
        # would act on.  It is cheap (microseconds) and reuses the feature table
        # that was computed anyway.
        report = assess(instances, features, probs, self.qc_config) if self.run_qc else None

        summary = summarize_image(features, image.name)
        summary["backend"] = self.backend
        summary.update({k: round(v, 4) for k, v in timings.items()})
        if report is not None:
            summary.update(report.as_dict())

        return PipelineResult(
            image_id=image.name,
            raw=image,
            preprocessed=pre.image,
            instances=instances,
            features=features,
            summary=summary,
            probabilities=probs,
            distance=distance,
            stages=pre.stages,
            timings=timings,
            qc=report,
        )

    def run_batch(self, images: list[MicroscopyImage]) -> list[PipelineResult]:
        """Run the pipeline over several fields."""
        return [self.run(image) for image in images]


def results_to_tables(results: list[PipelineResult]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Concatenate results into (per-object table, per-image summary table)."""
    objects = [r.features for r in results if r.features is not None and not r.features.empty]
    per_object = pd.concat(objects, ignore_index=True) if objects else pd.DataFrame()
    per_image = pd.DataFrame([r.summary for r in results])
    return per_object, per_image
