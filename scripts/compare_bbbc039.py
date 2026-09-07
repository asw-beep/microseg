"""Paired comparison of two backends on BBBC039.

    python scripts/evaluate_bbbc039.py --backend unet      --limit 100
    python scripts/evaluate_bbbc039.py --backend unet      --limit 100 --offset 100
    python scripts/evaluate_bbbc039.py --backend classical --limit 100 --config configs/classical_baseline.yaml
    python scripts/evaluate_bbbc039.py --backend classical --limit 100 --offset 100 --config configs/classical_baseline.yaml
    python scripts/compare_bbbc039.py

Both backends see identical images, so the comparison is *paired* -- the right
test is on per-image differences, not on the gap between two independent means.
That matters here: the difference in AP is small next to the spread across
fields, and an unpaired reading of the summary numbers would call a result that
a paired test cannot separate from zero.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from microseg.utils import LOGGER, setup_logging  # noqa: E402

ROOT = Path("outputs/bbbc039")


def load(backend: str) -> pd.DataFrame:
    """Every evaluated field for one backend, across all offset runs."""
    parts = []
    for d in sorted(ROOT.glob(f"{backend}*")):
        csv = d / "per_image_metrics.csv"
        if csv.exists():
            parts.append(pd.read_csv(csv))
    if not parts:
        raise FileNotFoundError(f"no results for {backend!r} under {ROOT}")
    return pd.concat(parts, ignore_index=True).drop_duplicates("image_id")


def paired_report(values: np.ndarray, name: str, lower_is_better: bool = False,
                  n_boot: int = 10000, seed: int = 0) -> None:
    rng = np.random.default_rng(seed)
    boot = np.array([rng.choice(values, values.size, replace=True).mean() for _ in range(n_boot)])
    lo, hi = np.percentile(boot, [2.5, 97.5])
    wins = int((values < 0).sum() if lower_is_better else (values > 0).sum())

    crosses_zero = lo <= 0 <= hi
    LOGGER.info("%s: mean %+.4f  median %+.4f  95%% CI [%+.4f, %+.4f]  first better on %d/%d%s",
                name, values.mean(), np.median(values), lo, hi, wins, values.size,
                "  <- CI includes zero" if crosses_zero else "")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--a", default="unet")
    parser.add_argument("--b", default="classical")
    args = parser.parse_args(argv)

    setup_logging()
    a, b = load(args.a), load(args.b)
    merged = a.merge(b, on="image_id", suffixes=("_a", "_b"))
    LOGGER.info("%d fields evaluated by both %s and %s", len(merged), args.a, args.b)

    paired_report(np.asarray(merged.ap_mean_a - merged.ap_mean_b), f"AP ({args.a} - {args.b})")
    paired_report(np.asarray(merged.f1_a - merged.f1_b), f"F1 ({args.a} - {args.b})")
    paired_report(np.asarray(merged.dice_a - merged.dice_b), f"Dice ({args.a} - {args.b})")
    paired_report(
        np.asarray(merged.count_abs_error_a - merged.count_abs_error_b),
        f"Count error ({args.a} - {args.b})", lower_is_better=True,
    )

    try:
        from scipy import stats
        p = stats.wilcoxon(merged.ap_mean_a - merged.ap_mean_b).pvalue
        LOGGER.info("Wilcoxon signed-rank on AP: p = %.3g", p)
    except Exception as exc:  # pragma: no cover - scipy is a hard dep, but be safe
        LOGGER.warning("skipped Wilcoxon: %s", exc)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
