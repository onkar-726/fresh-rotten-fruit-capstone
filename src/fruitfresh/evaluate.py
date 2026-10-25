"""Evaluation of a trained model: accuracy, precision, recall, F1, confusion matrices, intervals.

Evaluation views
----------------
* ``test_original``  PRIMARY. One original (non-augmented) image per source photo of the test split
                      -> independent samples, no augmented copies.
* ``test_full``       every test image (originals + the dataset's pre-augmented copies); copies of one
                      photo are not independent, so intervals resample whole source photos.
* ``validation``      all validation images (used to choose between experiments).
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support

from .config import CLASS_NAMES, NUM_CLASSES, TARGETS
from .datasets import make_dataset

LABELS = list(range(NUM_CLASSES))


def original_only(df: pd.DataFrame) -> pd.DataFrame:
    """One non-augmented image per source photo."""
    return df[df["is_original"]].drop_duplicates("source_id").reset_index(drop=True)


def aug_group(aug_type: str) -> str:
    """'rotated_by_15' -> 'rotated'; '' -> 'original'; chains are joined with '+'."""
    if not aug_type:
        return "original"
    parts = ["rotated" if t.startswith("rotated") else t for t in aug_type.split("+")]
    return "+".join(parts)


def predict_df(model, df: pd.DataFrame, dataset_root, batch_size: int = 128) -> pd.DataFrame:
    """Run ``model`` over ``df`` (input size read from the model) -> per-image prediction table."""
    size = tuple(model.input_shape[1:3])
    ds = make_dataset(df, dataset_root, image_size=size, batch_size=batch_size, training=False, cache=False)
    probs = model.predict(ds, verbose=0)
    out = df[["image_path", "source_id", "class_name", "is_original", "aug_type", "label"]].copy()
    out = out.rename(columns={"label": "y_true"}).reset_index(drop=True)
    out["y_pred"] = probs.argmax(1)
    out["p_max"] = probs.max(1)
    for i, name in enumerate(CLASS_NAMES):
        out[f"p_{name}"] = probs[:, i]
    return out


def metrics_from_predictions(pred: pd.DataFrame) -> dict:
    y, p = pred["y_true"].to_numpy(), pred["y_pred"].to_numpy()
    prec, rec, f1, sup = precision_recall_fscore_support(y, p, labels=LABELS, zero_division=0)
    return {
        "n_images": int(len(pred)),
        "n_sources": int(pred["source_id"].nunique()),
        "accuracy": float((y == p).mean()),
        "macro_f1": float(f1.mean()),
        "macro_precision": float(prec.mean()),
        "macro_recall": float(rec.mean()),
        "weighted_f1": float(np.average(f1, weights=np.maximum(sup, 1))),
        "per_class": {
            c: {"precision": float(prec[i]), "recall": float(rec[i]), "f1": float(f1[i]), "support": int(sup[i])}
            for i, c in enumerate(CLASS_NAMES)
        },
        "confusion_matrix": confusion_matrix(y, p, labels=LABELS).tolist(),
    }


def _acc_macro_f1(cm: np.ndarray):
    tp = np.diag(cm).astype(float)
    sup, pred = cm.sum(1), cm.sum(0)
    prec = np.divide(tp, pred, out=np.zeros_like(tp), where=pred > 0)
    rec = np.divide(tp, sup, out=np.zeros_like(tp), where=sup > 0)
    f1 = np.divide(2 * prec * rec, prec + rec, out=np.zeros_like(tp), where=(prec + rec) > 0)
    return tp.sum() / cm.sum(), f1[sup > 0].mean()


def bootstrap_ci(pred: pd.DataFrame, n_boot: int = 2000, seed: int = 0, alpha: float = 0.05) -> dict:
    """Cluster bootstrap: resample whole SOURCE PHOTOS, recompute accuracy and macro-F1."""
    sources = pred["source_id"].unique()
    per_src = np.zeros((len(sources), NUM_CLASSES, NUM_CLASSES))
    pos = {s: i for i, s in enumerate(sources)}
    np.add.at(per_src, (pred["source_id"].map(pos).to_numpy(), pred["y_true"].to_numpy(), pred["y_pred"].to_numpy()), 1)
    rng = np.random.default_rng(seed)
    accs, f1s = np.empty(n_boot), np.empty(n_boot)
    for b in range(n_boot):
        accs[b], f1s[b] = _acc_macro_f1(per_src[rng.integers(0, len(sources), len(sources))].sum(0))
    lo, hi = 100 * alpha / 2, 100 * (1 - alpha / 2)
    return {
        "accuracy_ci": [float(np.percentile(accs, lo)), float(np.percentile(accs, hi))],
        "macro_f1_ci": [float(np.percentile(f1s, lo)), float(np.percentile(f1s, hi))],
        "n_boot": n_boot,
    }


def check_targets(metrics: dict, targets: dict = TARGETS) -> dict:
    """Proposal targets: accuracy >= 90 % and macro-F1 > 0.85 (checked, never assumed)."""
    acc_ok = metrics["accuracy"] >= targets["accuracy"]
    f1_ok = metrics["macro_f1"] > targets["macro_f1"]
    return {"accuracy_ok": bool(acc_ok), "macro_f1_ok": bool(f1_ok), "passed": bool(acc_ok and f1_ok)}


def by_augmentation(pred: pd.DataFrame) -> dict:
    """Accuracy per kind of image (original / rotated / vertical_flip / ...): a robustness view."""
    g = pred.assign(group=pred["aug_type"].fillna("").map(aug_group))
    return {k: {"n": int(len(v)), "accuracy": float((v.y_true == v.y_pred).mean())} for k, v in g.groupby("group")}


def plot_confusion_matrix(cm, path=None, title="Confusion matrix", normalize=False):
    cm = np.asarray(cm, dtype=float)
    shown = cm / np.maximum(cm.sum(1, keepdims=True), 1) if normalize else cm
    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    im = ax.imshow(shown, cmap="Blues")
    ax.set_xticks(range(NUM_CLASSES), CLASS_NAMES, rotation=45, ha="right")
    ax.set_yticks(range(NUM_CLASSES), CLASS_NAMES)
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("True class")
    ax.set_title(title)
    for i in range(NUM_CLASSES):
        for j in range(NUM_CLASSES):
            v = shown[i, j]
            ax.text(j, i, f"{v:.2f}" if normalize else f"{int(v)}", ha="center", va="center",
                    color="white" if v > shown.max() / 2 else "black", fontsize=9)
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=200, bbox_inches="tight")
    return fig


def _write_view(out_dir: Path, run_name: str, view: str, pred: pd.DataFrame, m: dict) -> None:
    pred.to_csv(out_dir / f"predictions_{view}.csv", index=False)
    ci = ""
    if "accuracy_ci" in m:
        ci = (f"  95% CI acc {m['accuracy_ci'][0]:.3f}-{m['accuracy_ci'][1]:.3f}, "
              f"macro-F1 {m['macro_f1_ci'][0]:.3f}-{m['macro_f1_ci'][1]:.3f}")
    (out_dir / f"report_{view}.txt").write_text(
        f"{run_name} | {view} | {m['n_images']} images from {m['n_sources']} source photos\n"
        f"accuracy {m['accuracy']:.4f}   macro F1 {m['macro_f1']:.4f}{ci}\n\n"
        + classification_report(pred["y_true"], pred["y_pred"], labels=LABELS, target_names=CLASS_NAMES,
                                digits=4, zero_division=0)
    )
    for norm, suffix in ((False, ""), (True, "_normalized")):
        fig = plot_confusion_matrix(m["confusion_matrix"], out_dir / f"confusion_{view}{suffix}.png",
                                    f"{run_name} - {view}", normalize=norm)
        plt.close(fig)


def evaluate_validation(run_dir, val_df: pd.DataFrame, dataset_root, force: bool = False) -> dict:
    """Validation metrics of one trained run (cached in <run_dir>/val_eval/)."""
    from tensorflow import keras

    run_dir = Path(run_dir)
    out = run_dir / "val_eval"
    cache = out / "metrics.json"
    if cache.exists() and not force:
        return json.loads(cache.read_text())
    out.mkdir(parents=True, exist_ok=True)
    pred = predict_df(keras.models.load_model(run_dir / "model.keras"), val_df, dataset_root)
    m = metrics_from_predictions(pred)
    _write_view(out, run_dir.parent.name, "validation", pred, m)
    cache.write_text(json.dumps(m, indent=2))
    return m


def evaluate_run(run_dir, run_name: str, splits: dict, dataset_root, n_boot: int = 2000, force: bool = False,
                 batch_size: int = 128) -> dict:
    """Full evaluation: validation + test_original + test_full, saved to <run_dir>/eval/."""
    from tensorflow import keras

    run_dir = Path(run_dir)
    out = run_dir / "eval"
    cache = out / "metrics.json"
    if cache.exists() and not force:
        return json.loads(cache.read_text())
    out.mkdir(parents=True, exist_ok=True)
    model = keras.models.load_model(run_dir / "model.keras")
    views = {
        "validation": splits["validation"],
        "test_original": original_only(splits["test"]),
        "test_full": splits["test"],
    }
    result = {"run": run_name, "targets": {}}
    for view, df in views.items():
        pred = predict_df(model, df, dataset_root, batch_size=batch_size)
        m = metrics_from_predictions(pred)
        m.update(bootstrap_ci(pred, n_boot=n_boot))
        if view == "test_full":
            m["by_augmentation"] = by_augmentation(pred)
        result[view] = m
        result["targets"][view] = check_targets(m)
        _write_view(out, run_name, view, pred, m)
    cache.write_text(json.dumps(result, indent=2))
    return result
