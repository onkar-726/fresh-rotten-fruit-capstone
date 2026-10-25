"""Error analysis: where and why does a model fail?"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

from .config import CLASS_NAMES, DISPLAY_NAMES


def top_confusions(cm, k: int = 6) -> pd.DataFrame:
    """The k most frequent (true -> predicted) mistakes."""
    cm = np.asarray(cm)
    rows = []
    for i in range(len(CLASS_NAMES)):
        for j in range(len(CLASS_NAMES)):
            if i != j and cm[i, j] > 0:
                rows.append({"true class": CLASS_NAMES[i], "predicted as": CLASS_NAMES[j], "count": int(cm[i, j]),
                             "share of true class": round(cm[i, j] / max(cm[i].sum(), 1), 3)})
    cols = ["true class", "predicted as", "count", "share of true class"]
    return pd.DataFrame(rows, columns=cols).sort_values("count", ascending=False).head(k).reset_index(drop=True)


def status_vs_fruit_errors(pred: pd.DataFrame) -> dict:
    """Are mistakes about the FRESHNESS or about the FRUIT TYPE?  (class index: 0-2 fresh, 3-5 rotten; i % 3 = fruit)"""
    wrong = pred[pred.y_true != pred.y_pred]
    status_err = (wrong.y_true // 3) != (wrong.y_pred // 3)
    fruit_err = (wrong.y_true % 3) != (wrong.y_pred % 3)
    return {
        "n_errors": int(len(wrong)),
        "freshness_mistakes (fresh<->rotten)": int(status_err.sum()),
        "fruit_type_mistakes": int(fruit_err.sum()),
        "freshness_accuracy": float(((pred.y_true // 3) == (pred.y_pred // 3)).mean()),
        "fruit_type_accuracy": float(((pred.y_true % 3) == (pred.y_pred % 3)).mean()),
    }


def confidence_summary(pred: pd.DataFrame, high: float = 0.9) -> dict:
    ok = pred.y_true == pred.y_pred
    return {
        "mean_confidence_correct": float(pred.loc[ok, "p_max"].mean()) if ok.any() else None,
        "mean_confidence_wrong": float(pred.loc[~ok, "p_max"].mean()) if (~ok).any() else None,
        f"wrong_with_confidence_>={high}": int(((~ok) & (pred.p_max >= high)).sum()),
    }


def error_table(pred: pd.DataFrame) -> pd.DataFrame:
    """All mistakes, most confident first (the dangerous ones)."""
    wrong = pred[pred.y_true != pred.y_pred].copy()
    wrong["true class"] = wrong.y_true.map(dict(enumerate(CLASS_NAMES)))
    wrong["predicted"] = wrong.y_pred.map(dict(enumerate(CLASS_NAMES)))
    return wrong.sort_values("p_max", ascending=False)[
        ["image_path", "true class", "predicted", "p_max", "aug_type"]].reset_index(drop=True)


def misclassified_gallery(pred: pd.DataFrame, dataset_root, n: int = 12, cols: int = 4):
    """Pictures of the most confident mistakes."""
    errs = error_table(pred).head(n)
    if errs.empty:
        return None
    rows = int(np.ceil(len(errs) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(3.2 * cols, 3.4 * rows), squeeze=False)
    for ax in axes.ravel():
        ax.axis("off")
    for k, r in errs.iterrows():
        ax = axes[k // cols, k % cols]
        ax.imshow(Image.open(Path(dataset_root) / r.image_path).convert("RGB"))
        ax.set_title(f"true {DISPLAY_NAMES[r['true class']]}\npred {DISPLAY_NAMES[r['predicted']]} ({r.p_max:.2f})",
                     fontsize=8, color="red")
    fig.tight_layout()
    return fig


def write_error_report(run_name: str, pred: pd.DataFrame, cm, out_path) -> str:
    """Markdown summary used in the final report (numbers only from the predictions)."""
    sv = status_vs_fruit_errors(pred)
    cs = confidence_summary(pred)
    text = (f"# Error analysis - {run_name}\n\n"
            f"* images evaluated: {len(pred)}; mistakes: {sv['n_errors']}\n"
            f"* freshness accuracy (fresh vs rotten, ignoring fruit type): {sv['freshness_accuracy']:.3f}\n"
            f"* fruit-type accuracy (apple / banana / orange, ignoring freshness): {sv['fruit_type_accuracy']:.3f}\n"
            f"* mistakes that swap fresh and rotten: {sv['freshness_mistakes (fresh<->rotten)']}; "
            f"mistakes that swap the fruit: {sv['fruit_type_mistakes']}\n"
            f"* mean confidence when correct: {cs['mean_confidence_correct']}; when wrong: {cs['mean_confidence_wrong']}\n\n"
            f"## Most frequent mistakes\n\n{top_confusions(cm).to_markdown(index=False)}\n")
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(text)
    return text
