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

    # Null: intercept only
    null = smf.mixedlm(f"{dv} ~ 1", d, groups=d["response_id"])
    res_null = null.fit(reml=True, method="lbfgs", maxiter=500)

    full = smf.mixedlm(f"{dv} ~ C(design_cell)", d, groups=d["response_id"])
    res_full = full.fit(reml=True, method="lbfgs", maxiter=500)

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
        "We report a likelihood-ratio test vs. an intercept-only model (same random structure).\n"
    )

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
