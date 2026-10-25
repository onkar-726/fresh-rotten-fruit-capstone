"""Fair comparison of the two models: accuracy + uncertainty + significance + size + speed.

Every number in the tables comes from files written by evaluate_run / train_model.
Nothing in this module invents or hard-codes a result.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import binomtest

from .config import TARGETS
from .train import run_dir_for

VIEW_LABELS = {"test_original": "test (originals only, PRIMARY)", "test_full": "test (all images)",
               "validation": "validation"}


def _fmt(value, ci):
    return f"{value:.3f} [{ci[0]:.3f}, {ci[1]:.3f}]"


def metric_table(results: dict, view: str = "test_original") -> pd.DataFrame:
    """accuracy, macro precision / recall / F1 (with 95 % CI) for every model."""
    rows = []
    for name, r in results.items():
        m = r[view]
        rows.append({
            "model": name,
            "images": m["n_images"], "source photos": m["n_sources"],
            "accuracy [95% CI]": _fmt(m["accuracy"], m["accuracy_ci"]),
            "macro precision": round(m["macro_precision"], 3),
            "macro recall": round(m["macro_recall"], 3),
            "macro F1 [95% CI]": _fmt(m["macro_f1"], m["macro_f1_ci"]),
        })
    return pd.DataFrame(rows)


def per_class_table(results: dict, view: str = "test_original", metric: str = "f1") -> pd.DataFrame:
    return pd.DataFrame({n: {c: v[metric] for c, v in r[view]["per_class"].items()} for n, r in results.items()}).round(3)


def targets_table(results: dict, targets: dict = TARGETS) -> pd.DataFrame:
    """PASS/FAIL against the proposal targets, for every model and test view."""
    rows = []
    for name, r in results.items():
        for view in ("test_original", "test_full"):
            m, t = r[view], r["targets"][view]
            rows.append({
                "model": name, "view": VIEW_LABELS[view],
                f"accuracy (target >= {targets['accuracy']:.0%})": f"{m['accuracy']:.3f}",
                f"macro-F1 (target > {targets['macro_f1']})": f"{m['macro_f1']:.3f}",
                "result": "PASS" if t["passed"] else "FAIL",
            })
    return pd.DataFrame(rows)


def mcnemar(pred_a: pd.DataFrame, pred_b: pd.DataFrame) -> dict:
    """Exact McNemar test on the SAME images (rows must be aligned)."""
    assert (pred_a["image_path"].values == pred_b["image_path"].values).all(), "different images"
    ca = (pred_a["y_pred"] == pred_a["y_true"]).to_numpy()
    cb = (pred_b["y_pred"] == pred_b["y_true"]).to_numpy()
    only_a, only_b = int((ca & ~cb).sum()), int((~ca & cb).sum())
    p = 1.0 if only_a + only_b == 0 else float(binomtest(min(only_a, only_b), only_a + only_b, 0.5).pvalue)
    return {"a_only_correct": only_a, "b_only_correct": only_b, "p_value": p}


def mcnemar_between(work_dir, run_a: str, run_b: str, seed: int, view: str = "test_original") -> dict:
    load = lambda r: pd.read_csv(run_dir_for(work_dir, r, seed) / "eval" / f"predictions_{view}.csv")  # noqa: E731
    return mcnemar(load(run_a), load(run_b))


def fairness_table(work_dir, runs, seed: int) -> pd.DataFrame:
    """Evidence that the comparison is fair: what was identical, what differs on purpose."""
    infos = {r: json.loads((run_dir_for(work_dir, r, seed) / "train_info.json").read_text()) for r in runs}
    manifest = Path(work_dir).parent / "data" / "splits" / "split_manifest.json"
    rows = {
        "split (same CSV files)": {r: "see split_manifest.json" for r in runs},
        "seed": {r: i["seed"] for r, i in infos.items()},
        "input size": {r: "x".join(map(str, i["image_size"])) for r, i in infos.items()},
        "training augmentation": {r: i["augment"] for r, i in infos.items()},
        "class weights": {r: i["class_weights"] is not None for r, i in infos.items()},
        "loss / optimizer": {r: "sparse categorical cross-entropy / Adam" for r in runs},
        "train / val images": {r: f"{i['n_train_images']} / {i['n_val_images']}" for r, i in infos.items()},
        "pretrained (ImageNet)": {r: i["pretrained"] for r, i in infos.items()},
        "epochs run": {r: i["epochs_run"] for r, i in infos.items()},
        "hardware": {r: f"{i['hardware']['num_replicas']} x {i['hardware']['device_type']}, "
                        f"global batch {i['hardware']['global_batch_size']}, {i['hardware']['mixed_precision']}"
                     for r, i in infos.items()},
    }
    return pd.DataFrame(rows).T


# ----------------------------------------------------------------- efficiency
def keras_latency_ms(model, n_runs: int = 30, warmup: int = 5) -> float:
    x = np.random.uniform(0, 255, (1, *model.input_shape[1:])).astype("float32")
    for _ in range(warmup):
        model(x, training=False)
    times = []
    for _ in range(n_runs):
        t = time.perf_counter()
        model(x, training=False)
        times.append((time.perf_counter() - t) * 1000)
    return float(np.median(times))


def efficiency_table(work_dir, runs, seed: int) -> pd.DataFrame:
    """Parameters, file size, single-image latency (on the machine running this), training time."""
    from tensorflow import keras

    rows = []
    for name in runs:
        d = run_dir_for(work_dir, name, seed)
        info = json.loads((d / "train_info.json").read_text())
        model = keras.models.load_model(d / "model.keras")
        rows.append({
            "model": name,
            "params (M)": round(model.count_params() / 1e6, 2),
            "file size (MB)": round((d / "model.keras").stat().st_size / 1e6, 1),
            "latency (ms/image, batch 1)": round(keras_latency_ms(model), 1),
            "training time (min)": round(info["training_seconds"] / 60, 1),
        })
    return pd.DataFrame(rows)


def select_final_model(results: dict, efficiency: pd.DataFrame, tolerance: float = 0.01, size_factor: float = 2.0) -> dict:
    """Pre-registered rule (uses VALIDATION only, never the test set):

    1. highest validation macro-F1;
    2. if another model is within ``tolerance`` and at least ``size_factor`` times smaller, prefer it.
    """
    val = {n: r["validation"]["macro_f1"] for n, r in results.items()}
    best = max(val, key=val.get)
    size = efficiency.set_index("model")["file size (MB)"].to_dict()
    chosen, reason = best, "highest validation macro-F1"
    smaller = [n for n in val if val[best] - val[n] <= tolerance and size[n] * size_factor <= size[best]]
    if smaller:
        chosen = min(smaller, key=lambda n: size[n])
        reason = (f"within {tolerance:.0%} of the best validation macro-F1 ({best}) and at least "
                  f"{size_factor:g}x smaller")
    return {"final_model": chosen, "reason": reason, "validation_macro_f1": val}


# --------------------------------------------------------------------- plots
def plot_metric_bars(results: dict, view: str = "test_original"):
    names = list(results)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    for ax, metric, target in zip(axes, ("accuracy", "macro_f1"), (TARGETS["accuracy"], TARGETS["macro_f1"])):
        vals = np.array([results[n][view][metric] for n in names])
        cis = np.array([results[n][view][f"{metric}_ci"] for n in names])
        ax.bar(names, vals, yerr=[vals - cis[:, 0], cis[:, 1] - vals], capsize=6, color=["#4C72B0", "#DD8452"][: len(names)])
        ax.axhline(target, color="red", ls="--", lw=1, label=f"target {target}")
        ax.set_ylim(max(0, min(cis.min(), target) - 0.1), 1.0)
        ax.set_title(f"{metric.replace('_', '-')} - {VIEW_LABELS[view]}", fontsize=10)
        ax.legend(loc="lower right"); ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    return fig


def write_comparison_report(results: dict, work_dir, runs, seed: int, out_path) -> str:
    """Markdown report assembled only from saved results."""
    eff = efficiency_table(work_dir, runs, seed)
    final = select_final_model(results, eff)
    parts = ["# Model comparison (generated automatically - do not edit by hand)\n"]
    for view in ("test_original", "test_full"):
        parts += [f"\n## {VIEW_LABELS[view]}\n", metric_table(results, view).to_markdown(index=False), "\n",
                  "\nPer-class F1:\n", per_class_table(results, view).to_markdown(), "\n"]
    parts += ["\n## Proposal targets (accuracy >= 90 %, macro-F1 > 0.85)\n", targets_table(results).to_markdown(index=False),
              "\n\n## Fairness of the comparison\n", fairness_table(work_dir, runs, seed).to_markdown(),
              "\n\n## Efficiency\n", eff.to_markdown(index=False)]
    if len(runs) == 2:
        mc = mcnemar_between(work_dir, runs[0], runs[1], seed)
        parts += ["\n\n## Is the difference significant? (exact McNemar test, test originals)\n",
                  f"{runs[0]} only correct: {mc['a_only_correct']}; {runs[1]} only correct: {mc['b_only_correct']}; "
                  f"p-value = {mc['p_value']:.4f}. A p-value above 0.05 means the difference could be chance.\n"]
    parts += ["\n## Final model for the application\n",
              f"**{final['final_model']}** - {final['reason']} (validation macro-F1: "
              + ", ".join(f"{k} {v:.3f}" for k, v in final["validation_macro_f1"].items()) + ").\n"]
    text = "".join(parts)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(text)
    return text
