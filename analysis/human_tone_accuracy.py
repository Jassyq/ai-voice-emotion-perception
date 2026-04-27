"""
Compute how often humans got the clip's intended tone right.

Definition used:
- intended tone = first word of the filename (already provided as `intended_tone`)
- human tone right = perceived_emotion == intended_tone

Reports:
- overall accuracy across all rating rows
- per-clip majority-vote accuracy (mode of perceived_emotion per `stem`)
"""

from __future__ import annotations

import pandas as pd


def mode_series(s: pd.Series):
    """Return mode label; ties resolved by pandas' default (first by freq then sort order)."""
    vc = s.value_counts()
    return vc.index[0] if len(vc) > 0 else None


def main() -> None:
    root = __file__.split("\\analysis\\")[0]
    survey_csv = f"{root}\\survey_long_ratings.csv"

    df = pd.read_csv(survey_csv)
    sub = df.dropna(subset=["intended_tone", "perceived_emotion"]).copy()
    sub["intended_tone"] = sub["intended_tone"].astype(str).str.lower().str.strip()
    sub["perceived_emotion"] = sub["perceived_emotion"].astype(str).str.lower().str.strip()
    sub = sub[(sub["intended_tone"] != "nan") & (sub["perceived_emotion"] != "nan")]

    overall_acc = (sub["perceived_emotion"] == sub["intended_tone"]).mean()

    clip = sub.groupby("stem").agg(
        intended_tone=("intended_tone", "first"),
        majority_perceived=("perceived_emotion", mode_series),
        n_ratings=("perceived_emotion", "size"),
    )
    per_clip_majority_acc = (clip["majority_perceived"] == clip["intended_tone"]).mean()

    breakdown = (
        clip.groupby("intended_tone")
        .apply(lambda g: (g["majority_perceived"] == g["intended_tone"]).mean())
        .reset_index(name="per_clip_majority_acc")
    )
    counts = clip["intended_tone"].value_counts().rename("n_clips").reset_index()
    counts = counts.rename(columns={"index": "intended_tone"})
    breakdown = breakdown.merge(counts, on="intended_tone").sort_values("intended_tone")

    print(f"Ratings rows used: {len(sub)}")
    print(f"Unique clips (stems): {sub['stem'].nunique()}")
    print(f"Human tone accuracy (per rating): {overall_acc:.4f} ({overall_acc*100:.1f}%)")
    print(
        f"Human tone accuracy (per clip, majority vote): {per_clip_majority_acc:.4f} "
        f"({per_clip_majority_acc*100:.1f}%)"
    )
    print("\nPer intended tone (per-clip majority vote):")
    print(breakdown.to_string(index=False))


if __name__ == "__main__":
    main()

