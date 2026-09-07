"""Per-image quality control: is this segmentation trustworthy?

The gap this closes is the one that keeps the pipeline out of unattended use.
On the BBBC038 test split the U-Net returns a usable result on ~95% of fields
and a badly wrong one on the rest -- including fields where it reports dozens of
objects that are not there.  A tool that returns "27 nuclei" with the same
outward confidence whether it is right or catastrophically wrong cannot be run
without a human looking at every output.

So this module scores each result **without ground truth** and either passes it
or flags it for review.  Every signal is a statement about what a plausible
field of nuclei looks like:

* **foreground fraction** -- nuclei occupy a minority of a field.  If half the
  image is labelled nucleus, the threshold or the model has latched onto
  background texture.  This is the single most diagnostic signal, and it is what
  catches the pathological over-segmentation cases.
* **object density** -- objects per megapixel has a plausible range; hundreds of
  thousands is debris, zero is a missed field.
* **size dispersion** -- real nuclei in one field are broadly similar in size.
  A very high coefficient of variation of area means the result mixes whole
  nuclei with fragments.
* **tiny-object fraction** -- many objects far below the median size is the
  signature of fragmentation, whatever caused it.
* **solidity** -- a merged pair of nuclei is concave, so low median solidity
  indicates under-segmentation.  (This is the QC use of solidity noted in
  ``analysis/morphology.py``.)
* **circularity** -- nuclei are roughly round; a field of very irregular objects
  is usually a segmentation artifact rather than biology.
* **predictive confidence** -- for the U-Net only, the mean probability the model
  assigned to the class it chose.  Genuinely uncertain predictions are worth
  flagging even when the geometry looks plausible.

Each signal contributes a penalty in ``[0, 1]`` measuring how far outside its
plausible band the value sits, and the confidence is the product of
``(1 - penalty)``.  A product rather than a sum, so that one clearly broken
signal is enough to sink the score and several mildly odd ones compound.

Thresholds are tuned on the **validation** split and reported on **test**; see
``docs/quality_control.md`` for the measured catch rate.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class QCConfig:
    """Plausible band for each QC signal, as a ``(limit, width)`` pair.

    ``width`` is expressed in the signal's own units and sets how fast the
    penalty ramps from 0 at the limit to 1 at ``limit +/- width``.  An explicit
    width matters because these signals live on very different scales: solidity
    is meaningful over a 0.04-wide band near 1.0, while object density spans
    four orders of magnitude.  Scaling the ramp relative to the limit -- the
    obvious shortcut -- makes the solidity check useless.

    Limits were tuned on the BBBC038 **validation** split and are reported on
    **test**; see ``docs/quality_control.md``.
    """

    # Shape signals.  These are the two that separate genuine failures most
    # cleanly, and both have a direct interpretation: a merged pair of nuclei is
    # concave (low solidity), and a segmentation artifact is irregular (low
    # circularity).  Real nuclei are convex and round.
    min_median_solidity: float = 0.88
    solidity_width: float = 0.04
    min_median_circularity: float = 0.60
    circularity_width: float = 0.12

    # Size dispersion: whole nuclei mixed with fragments.
    max_area_cv: float = 1.0
    area_cv_width: float = 0.6
    tiny_area_ratio: float = 0.15
    max_tiny_fraction: float = 0.35
    tiny_width: float = 0.2

    # Sanity guards for pathological output.  Wide on purpose -- these exist to
    # catch "half the image is a nucleus", not to police normal variation.
    max_foreground_fraction: float = 0.45
    foreground_width: float = 0.15
    min_foreground_fraction: float = 0.0005
    max_object_density: float = 20000.0
    density_width: float = 10000.0

    # Mean probability assigned to the winning class over foreground pixels.
    # A weak signal here (it overlaps between good and bad results), so its band
    # is set low enough to fire only on genuinely uncertain predictions.
    min_predictive_confidence: float = 0.72
    predictive_width: float = 0.06

    # Below this overall score the result is flagged for review.
    pass_threshold: float = 0.55


@dataclass
class QCReport:
    """QC verdict for one image."""

    confidence: float
    passed: bool
    flags: list[str] = field(default_factory=list)
    signals: dict[str, float] = field(default_factory=dict)

    def as_dict(self) -> dict:
        out: dict = {
            "qc_confidence": round(float(self.confidence), 4),
            "qc_passed": bool(self.passed),
            "qc_flags": "; ".join(self.flags),
            "qc_n_flags": len(self.flags),
        }
        out.update({f"qc_{k}": round(float(v), 5) for k, v in self.signals.items()})
        return out


def _penalty_above(value: float, limit: float, width: float) -> float:
    """Penalty for exceeding ``limit``, reaching 1.0 at ``limit + width``.

    A ramp rather than a step, so a value just over the line reads as slightly
    suspicious rather than definitively broken.
    """
    if width <= 0 or value <= limit:
        return 0.0
    return float(np.clip((value - limit) / width, 0.0, 1.0))


def _penalty_below(value: float, limit: float, width: float) -> float:
    """Penalty for falling below ``limit``, reaching 1.0 at ``limit - width``."""
    if width <= 0 or value >= limit:
        return 0.0
    return float(np.clip((limit - value) / width, 0.0, 1.0))


def predictive_confidence(probs: np.ndarray, labels: np.ndarray | None = None) -> float:
    """Mean probability the model assigned to the class it actually chose.

    Restricted to predicted-foreground pixels when labels are available: a field
    that is 95% background would otherwise score high simply because background
    is easy, which tells you nothing about the objects.
    """
    probs = np.asarray(probs, dtype=np.float32)
    winning = probs.max(axis=0)
    if labels is not None:
        foreground = np.asarray(labels) > 0
        if foreground.any():
            return float(winning[foreground].mean())
    return float(winning.mean())


def compute_signals(
    labels: np.ndarray,
    features: pd.DataFrame | None = None,
    probs: np.ndarray | None = None,
    cfg: QCConfig | None = None,
) -> dict[str, float]:
    """Compute the raw QC signals for one segmented image."""
    cfg = cfg or QCConfig()
    labels = np.asarray(labels, dtype=np.int32)
    n_pixels = int(labels.size)
    n_objects = int(labels.max())

    signals: dict[str, float] = {
        "n_objects": float(n_objects),
        "foreground_fraction": float(np.count_nonzero(labels) / max(n_pixels, 1)),
        "object_density_per_mp": float(n_objects / max(n_pixels / 1e6, 1e-9)),
    }

    if features is not None and not features.empty and "area_px" in features:
        areas = features["area_px"].to_numpy(dtype=float)
        median_area = float(np.median(areas)) if areas.size else 0.0
        signals["median_area_px"] = median_area
        signals["area_cv"] = (
            float(np.std(areas) / np.mean(areas)) if areas.size and np.mean(areas) > 0 else 0.0
        )
        signals["tiny_fraction"] = (
            float(np.mean(areas < cfg.tiny_area_ratio * median_area)) if median_area > 0 else 0.0
        )
        if "solidity" in features:
            signals["median_solidity"] = float(np.median(features["solidity"].to_numpy(float)))
        if "circularity" in features:
            signals["median_circularity"] = float(
                np.median(features["circularity"].to_numpy(float))
            )
        if "touches_border" in features:
            signals["border_fraction"] = float(features["touches_border"].mean())

    if probs is not None:
        signals["predictive_confidence"] = predictive_confidence(probs, labels)

    return signals


def assess(
    labels: np.ndarray,
    features: pd.DataFrame | None = None,
    probs: np.ndarray | None = None,
    cfg: QCConfig | None = None,
) -> QCReport:
    """Score one segmentation result and flag it if implausible.

    Returns a :class:`QCReport` whose ``confidence`` is in ``[0, 1]`` and whose
    ``flags`` name, in plain language, every check that failed.
    """
    cfg = cfg or QCConfig()
    signals = compute_signals(labels, features, probs, cfg)

    penalties: dict[str, float] = {}
    flags: list[str] = []

    # An empty result is not "low confidence", it is a distinct outcome: there
    # is nothing to be uncertain about, and the caller needs to know why.
    if signals["n_objects"] == 0:
        return QCReport(
            confidence=0.0,
            passed=False,
            flags=["no objects detected"],
            signals=signals,
        )

    fg = signals["foreground_fraction"]
    penalties["foreground_high"] = _penalty_above(
        fg, cfg.max_foreground_fraction, cfg.foreground_width
    )
    if penalties["foreground_high"] > 0:
        flags.append(f"implausible foreground coverage ({fg:.0%} of the image)")

    penalties["foreground_low"] = _penalty_below(
        fg, cfg.min_foreground_fraction, cfg.min_foreground_fraction
    )
    if penalties["foreground_low"] > 0:
        flags.append(f"almost no foreground detected ({fg:.3%})")

    density = signals["object_density_per_mp"]
    penalties["density"] = _penalty_above(density, cfg.max_object_density, cfg.density_width)
    if penalties["density"] > 0:
        flags.append(f"implausible object density ({density:.0f} objects/megapixel)")

    if "area_cv" in signals:
        penalties["area_cv"] = _penalty_above(
            signals["area_cv"], cfg.max_area_cv, cfg.area_cv_width
        )
        if penalties["area_cv"] > 0:
            flags.append(
                f"object sizes vary implausibly (CV {signals['area_cv']:.2f}); "
                "likely fragmentation"
            )

    if "tiny_fraction" in signals:
        penalties["tiny"] = _penalty_above(
            signals["tiny_fraction"], cfg.max_tiny_fraction, cfg.tiny_width
        )
        if penalties["tiny"] > 0:
            flags.append(
                f"{signals['tiny_fraction']:.0%} of objects far below median size; "
                "likely fragmentation"
            )

    if "median_solidity" in signals:
        penalties["solidity"] = _penalty_below(
            signals["median_solidity"], cfg.min_median_solidity, cfg.solidity_width
        )
        if penalties["solidity"] > 0:
            flags.append(
                f"low median solidity ({signals['median_solidity']:.2f}); "
                "objects may be merged nuclei"
            )

    if "median_circularity" in signals:
        penalties["circularity"] = _penalty_below(
            signals["median_circularity"], cfg.min_median_circularity, cfg.circularity_width
        )
        if penalties["circularity"] > 0:
            flags.append(
                f"objects are unusually irregular (median circularity "
                f"{signals['median_circularity']:.2f})"
            )

    if "predictive_confidence" in signals:
        penalties["predictive"] = _penalty_below(
            signals["predictive_confidence"],
            cfg.min_predictive_confidence,
            cfg.predictive_width,
        )
        if penalties["predictive"] > 0:
            flags.append(
                f"model is uncertain (mean class probability "
                f"{signals['predictive_confidence']:.2f})"
            )

    # Product, not sum: one clearly broken signal should be enough on its own.
    confidence = float(np.prod([1.0 - p for p in penalties.values()])) if penalties else 1.0
    return QCReport(
        confidence=confidence,
        passed=confidence >= cfg.pass_threshold,
        flags=flags,
        signals={**signals, **{f"penalty_{k}": v for k, v in penalties.items() if v > 0}},
    )
