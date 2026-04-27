"""
Train XGBoost on survey_long_ratings.csv: perceived_emotion from acoustics.

Prefer the full report: python analysis/run_analysis.py --use-xgb
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

# When executed as `python analysis\\run_xgb_perceived_and_intended.py`, the
# `analysis/` folder is on sys.path, so we import sibling modules directly.
from ml_emotion_xgb import run_xgb_pipeline


def main() -> None:
    root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description="XGBoost: perceived emotion from acoustics")
    parser.add_argument(
        "--csv",
        type=Path,
        default=root / "survey_long_ratings.csv",
        help="Long-format ratings CSV (must include acoustic + target columns)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=root / "results",
        help="Output directory for plots (default: results/)",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.25,
        help="Fraction of clips held out for testing (higher reduces missing-class confusion matrices).",
    )
    args = parser.parse_args()

    if not args.csv.is_file():
        raise SystemExit(f"Input not found: {args.csv}")

    df = pd.read_csv(args.csv)
    args.out.mkdir(parents=True, exist_ok=True)

    md_perceived = run_xgb_pipeline(
        df,
        args.out,
        random_state=args.seed,
        use_xgb=True,
        group_col="response_id",
        test_size=args.test_size,
    )

    (args.out / "xgb_summary.md").write_text(md_perceived, encoding="utf-8")
    print(f"Wrote summary: {args.out / 'xgb_summary.md'}")
    print(f"Plots saved under: {args.out}")


if __name__ == "__main__":
    main()

