"""Evaluate trained runs on validation + test (accuracy, precision, recall, F1, confusion matrices, 95% CIs).

    python scripts/evaluate.py --runs baseline_cnn mobilenet_v2
Writes workspace/runs/<run>/seed<seed>/eval/ (metrics.json, reports, predictions, confusion matrices).
"""
import argparse

from _common import add_path_args, data_root, load_splits

from fruitfresh.evaluate import evaluate_run
from fruitfresh.train import run_dir_for


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", nargs="+", default=["baseline_cnn", "mobilenet_v2"])
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--force", action="store_true")
    add_path_args(ap)
    a = ap.parse_args()
    train_df, val_df, test_df = load_splits(a.split_dir)
    root = data_root(a)
    for run in a.runs:
        r = evaluate_run(run_dir_for(a.work_dir, run, a.seed), run, {"validation": val_df, "test": test_df},
                         root, n_boot=a.n_boot, force=a.force)
        for view in ("test_original", "test_full"):
            m = r[view]
            print(f"{run:<24} {view:<14} acc {m['accuracy']:.4f}  macro-F1 {m['macro_f1']:.4f}  "
                  f"({m['n_images']} images / {m['n_sources']} photos)  targets: "
                  f"{'PASS' if r['targets'][view]['passed'] else 'FAIL'}")


if __name__ == "__main__":
    main()
