"""Fair comparison of Model 1 (CNN) and Model 2 (MobileNetV2) + final-model choice.

    python scripts/compare.py                                   # baseline_cnn vs mobilenet_v2
    python scripts/compare.py --use-best                        # use the best experiment run of each family
Writes results/: comparison_report.md, tables (CSV), figures (PNG), final_model.json
"""
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
from _common import add_path_args, data_root, load_splits

from fruitfresh import compare as C
from fruitfresh.evaluate import evaluate_run
from fruitfresh.plots import plot_training_curves
from fruitfresh.train import run_dir_for


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--baseline-run", default="baseline_cnn")
    ap.add_argument("--mobilenet-run", default="mobilenet_v2")
    ap.add_argument("--use-best", action="store_true", help="use results/best_<family>.json written by summarize_experiments.py")
    ap.add_argument("--n-boot", type=int, default=2000)
    add_path_args(ap)
    a = ap.parse_args()
    out = Path(a.results_dir)
    out.mkdir(parents=True, exist_ok=True)
    if a.use_best:
        a.baseline_run = json.loads((out / "best_baseline_cnn.json").read_text())["best_run"]
        a.mobilenet_run = json.loads((out / "best_mobilenet_v2.json").read_text())["best_run"]
    runs = [a.baseline_run, a.mobilenet_run]

    train_df, val_df, test_df = load_splits(a.split_dir)
    root = data_root(a)
    results = {r: evaluate_run(run_dir_for(a.work_dir, r, a.seed), r, {"validation": val_df, "test": test_df},
                               root, n_boot=a.n_boot) for r in runs}

    for view in ("test_original", "test_full"):
        C.metric_table(results, view).to_csv(out / f"metrics_{view}.csv", index=False)
        C.per_class_table(results, view).to_csv(out / f"per_class_f1_{view}.csv")
        C.plot_metric_bars(results, view).savefig(out / f"metric_bars_{view}.png", dpi=200, bbox_inches="tight")
    C.targets_table(results).to_csv(out / "targets_check.csv", index=False)
    C.efficiency_table(a.work_dir, runs, a.seed).to_csv(out / "efficiency.csv", index=False)
    plot_training_curves({r: run_dir_for(a.work_dir, r, a.seed) / "history.csv" for r in runs}).savefig(
        out / "training_curves.png", dpi=200, bbox_inches="tight")
    plt.close("all")

    C.write_comparison_report(results, a.work_dir, runs, a.seed, out / "comparison_report.md")
    final = C.select_final_model(results, C.efficiency_table(a.work_dir, runs, a.seed))
    (out / "final_model.json").write_text(json.dumps(final, indent=2))

    print(C.metric_table(results, "test_original").to_string(index=False))
    print()
    print(C.targets_table(results).to_string(index=False))
    print("\nfinal model for the app:", final["final_model"], "-", final["reason"])
    print("report:", out / "comparison_report.md")


if __name__ == "__main__":
    main()
