#!/usr/bin/env python3
"""
Run mixed-effects models (design_cell → ratings) and XGBoost (acoustics → emotion).
Writes Markdown + HTML under results/ so you can open the report in a browser.

Usage (from repo root):
  python analysis/run_analysis.py
  python analysis/run_analysis.py --csv survey_long_ratings.csv --out results
"""
from __future__ import annotations

import argparse
import html as html_mod
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import pandas as pd

from analysis.descriptive_figures import write_descriptive_figures
from analysis.ml_emotion_xgb import run_xgb_pipeline
from analysis.stats_models import run_all_mixedlm


def _write_html(md_path: Path, html_path: Path, title: str) -> None:
    md_text = md_path.read_text(encoding="utf-8")
    simple = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <title>{html_mod.escape(title)}</title>
  <style>
    body {{ font-family: system-ui, sans-serif; max-width: 900px; margin: 2rem auto; padding: 0 1rem; line-height: 1.5; }}
    pre {{ background: #f4f4f4; padding: 1rem; overflow-x: auto; font-size: 0.85rem; }}
    img {{ max-width: 100%; height: auto; }}
    h1, h2, h3 {{ margin-top: 1.5rem; }}
  </style>
</head>
<body>
  <h1>{html_mod.escape(title)}</h1>
  <p><small>Generated {html_mod.escape(datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"))}</small></p>
  <p>For best formatting, open <code>summary.md</code> in your editor or viewer. Figures below reference files in this folder.</p>
  <hr/>
"""
    # Append markdown as readable pre
    simple += "<pre>" + html_mod.escape(md_text) + "</pre>\n"
    simple += "<h2>Figures</h2>\n"
    for src, alt in [
        ("fig_ratings_by_design_cell.png", "Mean ratings by design cell"),
        ("fig_perceived_by_design_cell.png", "Perceived emotion by design cell"),
        ("fig_intended_vs_perceived_heatmap.png", "Intended vs perceived heatmap"),
        ("fig_clips_per_participant.png", "Clips per participant"),
        ("xgb_confusion_matrix.png", "Perceived confusion matrix"),
        ("xgb_feature_importance.png", "Perceived feature importance"),
    ]:
        simple += f'<p><img src="{html_mod.escape(src)}" alt="{html_mod.escape(alt)}"/></p>\n'
    simple += "</body></html>"
    html_path.write_text(simple, encoding="utf-8")


def main() -> None:
    root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description="Run ANOVA-style mixed models + XGBoost report.")
    parser.add_argument(
        "--csv",
        type=Path,
        default=root / "survey_long_ratings.csv",
        help="Long-format CSV (with acoustic columns if available)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=root / "results",
        help="Output directory for reports and figures",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--use-xgb",
        action="store_true",
        help="Use XGBoost if importable; default is sklearn HistGradientBoostingClassifier",
    )
    args = parser.parse_args()

    warnings.filterwarnings(
        "ignore",
        message="Random effects covariance is singular",
        category=UserWarning,
        module="statsmodels.regression.mixed_linear_model",
    )

    if not args.csv.is_file():
        raise SystemExit(f"Input not found: {args.csv}")

    df = pd.read_csv(args.csv)
    args.out.mkdir(parents=True, exist_ok=True)

    sections: list[str] = []
    sections.append(f"# Analysis report\n\n**Input:** `{args.csv.name}`  \n**Rows:** {len(df)}\n")

    try:
        sections.append(write_descriptive_figures(df, args.out))
    except Exception as e:
        sections.append(f"## Descriptive figures (skipped)\n\n`{e}`\n")

    # Stats
    md_stats, _diag = run_all_mixedlm(df)
    sections.append(md_stats)

    # ML (may fail without features)
    ml_md = ""
    try:
        ml_md = run_xgb_pipeline(
            df,
            args.out,
            random_state=args.seed,
            use_xgb=args.use_xgb,
            group_col="response_id",
        )
    except Exception as e:
        ml_md = f"## XGBoost (skipped or failed)\n\n`{e}`\n"
    sections.append(ml_md)

    summary_md = "\n\n".join(sections)
    md_path = args.out / "summary.md"
    md_path.write_text(summary_md, encoding="utf-8")

    html_path = args.out / "summary.html"
    _write_html(md_path, html_path, "Voice emotion study — analysis")

    print(f"Wrote {md_path}")
    print(f"Wrote {html_path}")
    print(f"Open in browser: file://{html_path.resolve()}")


if __name__ == "__main__":
    main()
