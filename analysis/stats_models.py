"""
Linear mixed-effects models (participant random intercept) for Likert outcomes.
Tests whether design_cell (tone × words condition) predicts ratings, accounting for repeated measures.
"""
from __future__ import annotations

import io
from typing import Any

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy.stats import f_oneway, kruskal


def _clean_df(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d["design_cell"] = d["design_cell"].astype("category")
    for c in ("naturalness", "trust", "emotion_strength", "tone_vs_words"):
        if c in d.columns:
            d[c] = pd.to_numeric(d[c], errors="coerce")
    d = d.dropna(subset=["response_id", "design_cell"])
    return d


_DESIGN_CELL_LABEL = {
    "tone_neutral_word_neutral": "Neutral tone / Neutral words",
    "tone_neutral_word_emotional": "Neutral tone / Emotional words",
    "tone_emotional_word_neutral": "Emotional tone / Neutral words",
    "tone_emotional_word_emotional": "Emotional tone / Emotional words",
}


def _design_cell_row_order(cells_raw: list[str]) -> list[str]:
    preferred = [
        "tone_neutral_word_neutral",
        "tone_neutral_word_emotional",
        "tone_emotional_word_neutral",
        "tone_emotional_word_emotional",
    ]
    out = [c for c in preferred if c in cells_raw]
    for c in sorted(cells_raw):
        if c not in out:
            out.append(c)
    return out


def observed_means_by_design_cell_md(df: pd.DataFrame) -> str:
    """
    Markdown table: raw mean (SD) and n per design cell for each Likert DV.
    """
    d = _clean_df(df)
    dvs = [c for c in ("naturalness", "trust", "emotion_strength", "tone_vs_words") if c in d.columns]
    if not dvs:
        return ""

    cells = _design_cell_row_order(d["design_cell"].dropna().astype(str).unique().tolist())
    lines: list[str] = [
        "### Observed means by design cell (descriptive)\n",
        "Per cell: **mean (SD)** on the 1–5 scale and **n** clip ratings. "
        "These are **simple sample averages** (not mixed-model predictions).\n\n",
        "**About χ² in the models below:** The likelihood-ratio test compares how well the data support a model **with** vs **without** design-cell fixed effects (using **log-likelihood**), "
        "not a direct calculation on this table’s means. The model still relates to **differences across conditions**, "
        "while adjusting for repeated measures per participant.\n\n",
    ]
    head = (
        "| Design cell | "
        + " | ".join(dv.replace("_", " ").title() for dv in dvs)
        + " | n clips |\n"
    )
    sep = "| " + " | ".join(["---"] * (len(dvs) + 2)) + " |\n"
    lines.append(head)
    lines.append(sep)

    for cell in cells:
        sub = d.loc[d["design_cell"].astype(str) == cell]
        n_cell = len(sub)
        row_label = _DESIGN_CELL_LABEL.get(cell, cell.replace("_", " "))
        row_vals: list[str] = [row_label]
        for dv in dvs:
            v = sub[dv].dropna()
            if len(v) == 0:
                row_vals.append("—")
            else:
                m, s = float(v.mean()), float(v.std(ddof=1)) if len(v) > 1 else 0.0
                row_vals.append(f"{m:.2f} ({s:.2f})")
        row_vals.append(str(n_cell))
        lines.append("| " + " | ".join(row_vals) + " |\n")

    lines.append("\n")
    return "".join(lines)


def _fit_mixedlm_pair(null_m: Any, full_m: Any) -> tuple[Any, Any, dict[str, Any]]:
    """
    Fit null and full MixedLM with the same optimizer settings (required for LRT).
    Tries REML + L-BFGS first; falls back to ML + Nelder-Mead when numerics break.
    """
    configs: list[dict[str, Any]] = [
        {"reml": True, "method": "lbfgs", "maxiter": 500},
        {"reml": True, "method": "nm", "maxiter": 3000},
        {"reml": False, "method": "lbfgs", "maxiter": 500},
        {"reml": False, "method": "nm", "maxiter": 3000},
    ]
    last_err: Exception | None = None
    for cfg in configs:
        try:
            r0 = null_m.fit(**cfg)
            r1 = full_m.fit(**cfg)
            meta = {
                "fit_reml": cfg["reml"],
                "fit_method": cfg["method"],
                "fit_maxiter": cfg["maxiter"],
            }
            return r0, r1, meta
        except Exception as e:
            last_err = e
            continue
    raise RuntimeError(f"MixedLM fit failed for all optimizer configs: {last_err}")


def fit_mixedlm_outcome(
    df: pd.DataFrame,
    dv: str,
) -> tuple[Any, dict[str, Any]]:
    """
    Fit: DV ~ C(design_cell) with random intercept for response_id.
    Returns (fitted MixedLMResultsWrapper, diagnostics dict).
    """
    d = _clean_df(df)
    d = d.dropna(subset=[dv])
    if len(d) < 20:
        raise ValueError(f"Not enough rows with non-missing {dv}: {len(d)}")

    null = smf.mixedlm(f"{dv} ~ 1", d, groups=d["response_id"])
    full = smf.mixedlm(f"{dv} ~ C(design_cell)", d, groups=d["response_id"])
    res_null, res_full, fit_meta = _fit_mixedlm_pair(null, full)

    # Likelihood ratio test (fixed effects)
    df_diff = float(len(res_full.fe_params) - len(res_null.fe_params))
    lr_stat = 2.0 * (res_full.llf - res_null.llf)
    from scipy import stats as st

    p_lrt = float(1.0 - st.chi2.cdf(lr_stat, df_diff)) if df_diff > 0 else np.nan

    diag = {
        "n": len(d),
        "n_participants": d["response_id"].nunique(),
        "lr_statistic": float(lr_stat),
        "lr_df": float(df_diff),
        "lr_pvalue": p_lrt,
        "icc_approx": _approx_icc(res_full, d, dv),
        **fit_meta,
    }
    return res_full, diag


def _approx_icc(result: Any, d: pd.DataFrame, dv: str) -> float:
    """Rough ICC = var(random) / (var(random) + var(residual)) if available."""
    try:
        re = float(result.cov_re.iloc[0, 0])
        resid = float(result.scale)
        if re + resid > 0:
            return re / (re + resid)
    except Exception:
        pass
    return float("nan")


def run_all_mixedlm(df: pd.DataFrame) -> tuple[str, dict[str, dict[str, Any]]]:
    """Fit each Likert DV; return markdown report and per-DV diagnostics."""
    out_lines: list[str] = []
    out_lines.append("## Linear mixed models (random intercept: participant)\n")
    out_lines.append(
        "Each model: **DV ~ C(design_cell)** with a random intercept for `response_id`. "
        "This accounts for repeated measures (five clips per person). "
        "We report a **likelihood-ratio test (χ²)** comparing that model to an intercept-only model "
        "(same random structure). That test is based on **model likelihood**, not on a formula applied "
        "directly to the raw means in the table below—but it addresses the same scientific question: "
        "whether ratings **differ across design cells** after accounting for participant clustering.\n\n"
    )
    out_lines.append(observed_means_by_design_cell_md(df))

    dvs = ["naturalness", "trust", "emotion_strength", "tone_vs_words"]
    all_diag: dict[str, dict[str, Any]] = {}

    for dv in dvs:
        if dv not in df.columns:
            continue
        try:
            res, diag = fit_mixedlm_outcome(df, dv)
            all_diag[dv] = diag
            out_lines.append(f"### Outcome: `{dv}`\n")
            out_lines.append(
                f"- N clips = {diag['n']}, N participants = {diag['n_participants']}\n"
            )
            if "fit_method" in diag:
                out_lines.append(
                    f"- **Fit:** REML={diag.get('fit_reml')}, optimizer={diag.get('fit_method')}, "
                    f"maxiter={diag.get('fit_maxiter')}\n"
                )
            out_lines.append(
                f"- LRT vs. null: χ²({diag['lr_df']:.0f}) = {diag['lr_statistic']:.3f}, "
                f"p = {diag['lr_pvalue']:.4g}\n"
            )
            if not getattr(res, "converged", True):
                out_lines.append(
                    "- **Note:** Optimizer did not fully converge; interpret with caution.\n"
                )
            if not np.isnan(diag["icc_approx"]):
                out_lines.append(f"- Approx. ICC (participant): {diag['icc_approx']:.3f}\n")
            buf = io.StringIO()
            buf.write(res.summary().as_text())
            out_lines.append("\n```\n" + buf.getvalue() + "\n```\n")
        except Exception as e:
            all_diag[dv] = {"error": str(e)}
            out_lines.append(f"### Outcome: `{dv}`\n")
            out_lines.append(f"*Model failed:* `{e}`\n")

    # Supplementary: classical tests ignoring repeated measures (for comparison only)
    out_lines.append(supplementary_classical_tests(_clean_df(df)))

    return "\n".join(out_lines), all_diag


def supplementary_classical_tests(d: pd.DataFrame) -> str:
    """
    One-way ANOVA and Kruskal–Wallis for design_cell × each Likert DV.
    **Ignores non-independence of clips** — use mixed models above for inference.
    """
    lines: list[str] = [
        "## Supplementary: one-way ANOVA & Kruskal–Wallis (independence assumption)\n",
        "These treat each **clip rating** as independent. Your data are **not** independent "
        "(five clips per person), so p-values here can be anti-conservative. "
        "They are shown for transparency and for comparison with textbook ANOVA.\n",
    ]
    dvs = ["naturalness", "trust", "emotion_strength", "tone_vs_words"]
    cells = sorted(d["design_cell"].dropna().unique().tolist())

    for dv in dvs:
        if dv not in d.columns:
            continue
        sub = d.dropna(subset=[dv])
        groups = [sub.loc[sub["design_cell"] == c, dv].values for c in cells]
        groups = [g for g in groups if len(g) > 0]
        if len(groups) < 2:
            continue
        try:
            f_stat, p_a = f_oneway(*groups)
        except Exception:
            f_stat, p_a = float("nan"), float("nan")
        try:
            h_stat, p_k = kruskal(*groups)
        except Exception:
            h_stat, p_k = float("nan"), float("nan")
        lines.append(f"### `{dv}`\n")
        lines.append(
            f"- One-way ANOVA: F = {f_stat:.3f}, p = {p_a:.4g}\n"
            f"- Kruskal–Wallis: H = {h_stat:.3f}, p = {p_k:.4g}\n"
        )

    return "\n".join(lines)
