"""
Gradient boosting classifiers for emotion labels from acoustic features.

Default backend: sklearn HistGradientBoostingClassifier (portable).
Optional backend: XGBoost if available.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.preprocessing import LabelEncoder

try:
    from xgboost import XGBClassifier

    _HAS_XGB = True
except Exception:
    XGBClassifier = None  # type: ignore[misc, assignment]
    _HAS_XGB = False


def _feature_columns(df: pd.DataFrame) -> list[str]:
    """Numeric columns from merged acoustic export (after trust)."""
    skip = {
        "response_id",
        "start_date",
        "duration_sec",
        "finished",
        "form",
        "clip_order_in_form",
        "clip_slot_index",
        "filename",
        "stem",
        "intended_tone",
        "intended_words",
        "filename_variant",
        "design_cell",
        "perceived_emotion",
        "emotion_strength",
        "tone_vs_words",
        "naturalness",
        "trust",
        "acoustic_filename",
    }
    feats: list[str] = []
    for c in df.columns:
        if c in skip:
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            feats.append(c)
        else:
            t = pd.to_numeric(df[c], errors="coerce")
            if t.notna().sum() > len(df) * 0.5:
                feats.append(c)
    return feats


def _make_classifier(random_state: int, use_xgb: bool) -> Any:
    if use_xgb and _HAS_XGB:
        # Tuned for tabular acoustics: shallow defaults under-fit duplicate X rows
        # (same clip features, different raters). Deeper trees better approximate
        # per-clip label modes on the training participants.
        return XGBClassifier(
            n_estimators=500,
            max_depth=9,
            learning_rate=0.06,
            subsample=0.85,
            colsample_bytree=0.85,
            random_state=random_state,
            n_jobs=-1,
            eval_metric="mlogloss",
            tree_method="hist",
        )
    return HistGradientBoostingClassifier(
        max_iter=250,
        max_depth=8,
        learning_rate=0.06,
        random_state=random_state,
    )


def train_emotion_classifier(
    df: pd.DataFrame,
    target_col: str = "perceived_emotion",
    *,
    random_state: int = 42,
    test_size: float = 0.25,
    use_xgb: bool = False,
    group_col: str | None = None,
) -> tuple[Any, LabelEncoder, dict[str, Any], np.ndarray, np.ndarray]:
    """
    Returns fitted model, label encoder, metrics dict, y_test and y_pred (encoded).
    """
    df = df.copy()
    if target_col not in df.columns:
        raise ValueError(f"Missing target column: {target_col}")
    df[target_col] = df[target_col].astype(str).str.lower().str.strip()
    df = df[df[target_col].notna() & (df[target_col] != "nan")]

    feats = _feature_columns(df)
    if not feats:
        raise ValueError("No numeric feature columns found. Merge acoustic features first.")

    X = df[feats].apply(pd.to_numeric, errors="coerce")
    y_raw = df[target_col].values
    mask = X.notna().all(axis=1)
    X = X.loc[mask]
    y_raw = y_raw[mask.values]

    counts = pd.Series(y_raw).value_counts()
    rare = counts[counts < 2].index.tolist()
    if rare:
        keep = ~pd.Series(y_raw).isin(rare)
        X = X.loc[keep.values]
        y_raw = y_raw[keep.values]

    le = LabelEncoder()
    y = le.fit_transform(y_raw)

    if len(np.unique(y)) < 2:
        raise ValueError(f"Need at least 2 classes in {target_col} with enough rows.")

    split_note = "row_stratified"
    if group_col and group_col in df.columns:
        groups = df.loc[X.index, group_col].astype(str).values
        gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
        train_idx, test_idx = next(gss.split(X.values, y, groups=groups))
        X_train, X_test = X.values[train_idx], X.values[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        split_note = f"grouped_by_{group_col}"
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X.values,
            y,
            test_size=test_size,
            random_state=random_state,
            stratify=y,
        )

    clf = _make_classifier(random_state, use_xgb)
    # XGBoost can fail if y_train labels are not contiguous (e.g., a class is
    # missing in the training split). If that happens, remap labels based on
    # what appears in y_train, and drop test rows whose labels are not in
    # training.
    le_used = le
    X_test_used = X_test
    y_test_used = y_test
    y_train_used = y_train
    if use_xgb and _HAS_XGB:
        unique_train = np.unique(y_train)
        expected = np.arange(len(unique_train))
        if not np.array_equal(unique_train, expected):
            mapping = {int(old): int(i) for i, old in enumerate(unique_train.tolist())}
            y_train_used = np.array([mapping[int(v)] for v in y_train], dtype=int)
            keep_mask = np.isin(y_test, unique_train)
            X_test_used = X_test[keep_mask]
            y_test_used = np.array([mapping[int(v)] for v in y_test[keep_mask]], dtype=int)
            le_used = LabelEncoder()
            le_used.classes_ = np.array(le.classes_[unique_train], dtype=object)

    clf.fit(X_train, y_train_used)
    y_pred = clf.predict(X_test_used)
    acc = accuracy_score(y_test_used, y_pred)

    # Diagnostic: on this split, accuracy if we always predicted the training-set
    # majority emotion for each clip stem. Acoustics map 1:1 to stems in this study,
    # so this approximates an upper bound for clip-only inputs when raters disagree.
    majority_baseline_acc: float | None = None
    if group_col and group_col in df.columns and "stem" in df.columns:
        stems_all = df.loc[X.index, "stem"].astype(str).values
        y_labels_all = le.inverse_transform(y).astype(str)
        st_tr = stems_all[train_idx]
        lab_tr = y_labels_all[train_idx]
        mode_by_stem: dict[str, str] = {}
        for s in np.unique(st_tr):
            m = lab_tr[st_tr == s]
            vals, counts = np.unique(m, return_counts=True)
            mode_by_stem[str(s)] = str(vals[int(np.argmax(counts))])
        fallback = str(pd.Series(lab_tr).mode().iloc[0])
        st_te = stems_all[test_idx]
        if use_xgb and _HAS_XGB:
            utr = np.unique(y_train)
            if not np.array_equal(utr, np.arange(len(utr))):
                st_te = st_te[np.isin(y_test, utr)]
        lab_te = le_used.inverse_transform(y_test_used).astype(str)
        pred_maj = np.array([mode_by_stem.get(st_te[i], fallback) for i in range(len(st_te))])
        majority_baseline_acc = float(np.mean(pred_maj == lab_te))

    report = classification_report(
        y_test_used,
        y_pred,
        labels=np.arange(len(le_used.classes_)),
        target_names=le_used.classes_,
        zero_division=0,
    )

    backend = "xgboost" if use_xgb and _HAS_XGB else "sklearn_hist_gbrt"

    metrics: dict[str, Any] = {
        "accuracy": float(acc),
        "n_train": int(len(y_train_used)),
        "n_test": int(len(y_test_used)),
        "n_classes": int(len(le_used.classes_)),
        "classes": list(le_used.classes_),
        "classification_report": report,
        "feature_names": feats,
        "backend": backend,
        "split": split_note,
        "majority_baseline_acc": majority_baseline_acc,
    }

    return clf, le_used, metrics, y_test_used, y_pred


def plot_confusion_matrix_png(
    le: LabelEncoder,
    y_test: np.ndarray,
    y_pred: np.ndarray,
    out_path: Path,
) -> None:
    from sklearn.metrics import confusion_matrix

    cm = confusion_matrix(y_test, y_pred, labels=np.arange(len(le.classes_)))
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=list(le.classes_),
        yticklabels=list(le.classes_),
        ylabel="True label",
        xlabel="Predicted label",
        title="Confusion matrix (held-out test set)",
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    mx = float(cm.max()) if cm.size else 0.0
    thresh = mx / 2.0 if mx else 0.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                format(cm[i, j], "d"),
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
            )
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_feature_importance_png(
    clf: Any,
    feature_names: list[str],
    out_path: Path,
    top_k: int = 20,
) -> None:
    if hasattr(clf, "feature_importances_"):
        imp = np.asarray(clf.feature_importances_, dtype=float)
    else:
        imp = np.ones(len(feature_names)) / len(feature_names)
    order = np.argsort(imp)[::-1][: min(top_k, len(imp))]
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(np.arange(len(order)), imp[order][::-1])
    ax.set_yticks(np.arange(len(order)))
    ax.set_yticklabels([feature_names[i] for i in order[::-1]])
    ax.set_xlabel("Importance")
    ax.set_title(f"Feature importance (top {len(order)})")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def run_xgb_pipeline(
    df: pd.DataFrame,
    results_dir: Path,
    random_state: int = 42,
    use_xgb: bool = False,
    group_col: str = "response_id",
    test_size: float = 0.25,
) -> str:
    """Train model for perceived_emotion, save plots, return markdown section."""
    clf, le, metrics, y_test, y_pred = train_emotion_classifier(
        df,
        target_col="perceived_emotion",
        random_state=random_state,
        use_xgb=use_xgb,
        group_col=group_col,
        test_size=test_size,
    )

    results_dir.mkdir(parents=True, exist_ok=True)
    plot_confusion_matrix_png(le, y_test, y_pred, results_dir / "xgb_confusion_matrix.png")
    plot_feature_importance_png(
        clf, metrics["feature_names"], results_dir / "xgb_feature_importance.png"
    )

    backend_note = metrics["backend"]
    lines = [
        "## Supervised learning: perceived emotion from acoustic features\n",
        f"- **Backend:** `{backend_note}` (install OpenMP + `xgboost` and pass `--use-xgb` to force XGBoost)\n",
        f"- **Split:** `{metrics['split']}`",
        f"- **Accuracy (test):** {metrics['accuracy']:.3f}",
        f"- **Train / test clips:** {metrics['n_train']} / {metrics['n_test']}",
        f"- **Classes:** {metrics['n_classes']} ({', '.join(metrics['classes'])})\n",
    ]
    mb = metrics.get("majority_baseline_acc")
    if mb is not None:
        lines.append(
            f"- **Reference (train majority emotion per clip):** {mb:.3f} — "
            "approximate ceiling for models that only see clip-level inputs, because the same "
            "acoustics receive different emotion labels from different participants.\n"
        )
    lines += [
        "### Classification report (test)\n",
        "```\n" + metrics["classification_report"] + "\n```\n",
        "![Confusion matrix](xgb_confusion_matrix.png)\n",
        "![Feature importance](xgb_feature_importance.png)\n",
    ]
    return "\n".join(lines)


def run_label_pipeline(
    df: pd.DataFrame,
    results_dir: Path,
    *,
    target_col: str,
    title: str,
    file_prefix: str,
    random_state: int = 42,
    use_xgb: bool = False,
    test_size: float = 0.25,
) -> str:
    """Generic pipeline for any label column."""
    clf, le, metrics, y_test, y_pred = train_emotion_classifier(
        df,
        target_col=target_col,
        random_state=random_state,
        use_xgb=use_xgb,
        group_col="stem",
        test_size=test_size,
    )
    results_dir.mkdir(parents=True, exist_ok=True)
    cm_file = f"{file_prefix}_confusion_matrix.png"
    fi_file = f"{file_prefix}_feature_importance.png"
    plot_confusion_matrix_png(le, y_test, y_pred, results_dir / cm_file)
    plot_feature_importance_png(clf, metrics["feature_names"], results_dir / fi_file)
    return "\n".join(
        [
            f"## {title}\n",
            f"- **Target:** `{target_col}`",
            f"- **Backend:** `{metrics['backend']}`",
            f"- **Split:** `{metrics['split']}`",
            f"- **Accuracy (test):** {metrics['accuracy']:.3f}",
            f"- **Train / test clips:** {metrics['n_train']} / {metrics['n_test']}",
            f"- **Classes:** {metrics['n_classes']} ({', '.join(metrics['classes'])})\n",
            "### Classification report (test)\n",
            "```\n" + metrics["classification_report"] + "\n```\n",
            f"![Confusion matrix]({cm_file})\n",
            f"![Feature importance]({fi_file})\n",
        ]
    )
