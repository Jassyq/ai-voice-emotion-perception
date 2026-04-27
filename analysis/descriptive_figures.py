"""
Exploratory figures for the long-format ratings table (design_cell, Likert outcomes, emotions).
Writes PNGs under results/; meant to complement mixed models (descriptive only).
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

_DESIGN_LABELS: dict[str, str] = {
    "tone_neutral_word_neutral": "Neutral tone\nNeutral words",
    "tone_neutral_word_emotional": "Neutral tone\nEmotional words",
    "tone_emotional_word_neutral": "Emotional tone\nNeutral words",
    "tone_emotional_word_emotional": "Emotional tone\nEmotional words",
}


def _cell_order(cells: list[str]) -> list[str]:
    preferred = [
        "tone_neutral_word_neutral",
        "tone_neutral_word_emotional",
        "tone_emotional_word_neutral",
        "tone_emotional_word_emotional",
    ]
    out = [c for c in preferred if c in cells]
    for c in sorted(cells):
        if c not in out:
            out.append(c)
    return out


def _display_cell(c: str) -> str:
    return _DESIGN_LABELS.get(c, c.replace("_", " "))


def plot_ratings_by_design_cell(df: pd.DataFrame, out_path: Path) -> None:
    """Mean Likert outcomes by design_cell with approximate 95% CI (SE × 1.96)."""
    dvs = ["naturalness", "trust", "emotion_strength", "tone_vs_words"]
    present = [c for c in dvs if c in df.columns]
    if not present:
        return

    sub = df.dropna(subset=["design_cell"]).copy()
    cells = _cell_order(sub["design_cell"].dropna().unique().tolist())
    x = np.arange(len(cells))
    fig, axes = plt.subplots(2, 2, figsize=(11, 9), sharex=True)
    axes_flat = axes.flatten()
    for ax, dv in zip(axes_flat, present):
        means, ses = [], []
        for c in cells:
            v = pd.to_numeric(sub.loc[sub["design_cell"] == c, dv], errors="coerce").dropna()
            n = len(v)
            means.append(float(v.mean()) if n else float("nan"))
            ses.append(float(v.std(ddof=1) / np.sqrt(n)) if n > 1 else 0.0)
        means = np.array(means, dtype=float)
        ses = np.array(ses, dtype=float)
        ci = 1.96 * ses
        colors = plt.cm.Set2(np.linspace(0, 1, len(cells)))
        ax.bar(x, means, yerr=ci, capsize=4, color=colors, edgecolor="0.3", linewidth=0.6)
        ax.set_ylabel(dv.replace("_", " ").title())
        ax.set_xticks(x)
        ax.set_xticklabels([_display_cell(c) for c in cells], fontsize=8)
        ax.set_ylim(0.5, 5.5)
        ax.axhline(3.0, color="0.75", linestyle="--", linewidth=0.8, zorder=0)
    for j in range(len(present), 4):
        axes_flat[j].set_visible(False)
    fig.suptitle("Mean ratings by design cell (1–5 Likert; error bars ≈ 95% CI of mean)", fontsize=12)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def plot_perceived_stacked_by_design_cell(df: pd.DataFrame, out_path: Path) -> None:
    """100% stacked bars: distribution of perceived_emotion within each design_cell."""
    if "perceived_emotion" not in df.columns or "design_cell" not in df.columns:
        return
    sub = df.dropna(subset=["design_cell", "perceived_emotion"]).copy()
    sub["perceived_emotion"] = sub["perceived_emotion"].astype(str).str.lower().str.strip()
    cells = _cell_order(sub["design_cell"].unique().tolist())
    emotions = sorted(sub["perceived_emotion"].unique().tolist())
    ct = pd.crosstab(sub["design_cell"], sub["perceived_emotion"])
    ct = ct.reindex(index=cells, columns=emotions, fill_value=0)
    prop = ct.div(ct.sum(axis=1), axis=0).fillna(0)

    fig, ax = plt.subplots(figsize=(10, 5.5))
    left = np.zeros(len(cells))
    cmap = plt.cm.tab20(np.linspace(0, 1, max(len(emotions), 1)))
    for i, emo in enumerate(emotions):
        vals = prop[emo].values
        ax.barh(np.arange(len(cells)), vals, left=left, label=emo, color=cmap[i % len(cmap)], edgecolor="white", linewidth=0.5)
        left = left + vals
    ax.set_yticks(np.arange(len(cells)))
    ax.set_yticklabels([_display_cell(c) for c in cells], fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("Proportion of ratings")
    ax.set_title("Perceived emotion by design cell (within-cell proportions)")
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8, title="Perceived")
    ax.set_xlim(0, 1)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def plot_intended_vs_perceived_heatmap(df: pd.DataFrame, out_path: Path) -> None:
    """Count heatmap: intended tone (stimulus) × perceived emotion (response)."""
    need = {"intended_tone", "perceived_emotion"}
    if not need.issubset(df.columns):
        return
    sub = df.dropna(subset=list(need)).copy()
    for c in need:
        sub[c] = sub[c].astype(str).str.lower().str.strip()
    ct = pd.crosstab(sub["intended_tone"], sub["perceived_emotion"])
    if ct.size == 0:
        return
    # consistent row/col order
    ct = ct.sort_index(axis=0).sort_index(axis=1)

    fig, ax = plt.subplots(figsize=(10, 6))
    arr = ct.values.astype(float)
    im = ax.imshow(arr, aspect="auto", cmap="Blues")
    ax.set_xticks(np.arange(ct.shape[1]))
    ax.set_yticks(np.arange(ct.shape[0]))
    ax.set_xticklabels(ct.columns, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(ct.index, fontsize=9)
    ax.set_xlabel("Perceived emotion")
    ax.set_ylabel("Intended tone (stimulus)")
    ax.set_title("Stimulus tone vs. perceived emotion (raw counts)")
    fig.colorbar(im, ax=ax, label="Count")
    thresh = arr.max() / 2.0 if arr.size else 0
    for i in range(ct.shape[0]):
        for j in range(ct.shape[1]):
            v = int(arr[i, j])
            if v == 0:
                continue
            ax.text(
                j,
                i,
                str(v),
                ha="center",
                va="center",
                color="white" if arr[i, j] > thresh else "black",
                fontsize=8,
            )
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def plot_clips_per_participant(df: pd.DataFrame, out_path: Path) -> None:
    """Histogram: how many clip ratings each participant contributed (shows partial completers)."""
    if "response_id" not in df.columns:
        return
    counts = df.groupby("response_id").size()
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(counts.values, bins=range(1, int(counts.max()) + 2), align="left", rwidth=0.85, color="steelblue", edgecolor="0.3")
    ax.set_xlabel("Number of clip ratings per participant")
    ax.set_ylabel("Participants")
    ax.set_title("Exposure: ratings per participant (max 5 if complete)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def write_descriptive_figures(df: pd.DataFrame, results_dir: Path) -> str:
    """
    Write PNGs and return a short markdown section with image links.
    """
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    plot_ratings_by_design_cell(df, results_dir / "fig_ratings_by_design_cell.png")
    plot_perceived_stacked_by_design_cell(df, results_dir / "fig_perceived_by_design_cell.png")
    plot_intended_vs_perceived_heatmap(df, results_dir / "fig_intended_vs_perceived_heatmap.png")
    plot_clips_per_participant(df, results_dir / "fig_clips_per_participant.png")

    lines = [
        "## Descriptive figures\n",
        "Exploratory plots (not adjusted for repeated measures; mixed models above are primary for inference).\n",
        "### Mean Likert ratings by design cell\n",
        "![Ratings by design cell](fig_ratings_by_design_cell.png)\n",
        "### Perceived emotion mix within each design cell\n",
        "![Perceived emotion by design cell](fig_perceived_by_design_cell.png)\n",
        "### Stimulus intended tone vs. participant perceived emotion\n",
        "![Intended vs perceived heatmap](fig_intended_vs_perceived_heatmap.png)\n",
        "### How many clips each participant rated\n",
        "![Clips per participant](fig_clips_per_participant.png)\n",
    ]
    return "\n".join(lines)
