"""Validation table of the default run + experiments of a model; picks the best run (validation only).

    python scripts/summarize_experiments.py --family mobilenet_v2
Writes results/experiments_<family>.csv and results/best_<family>.json (used by compare.py --use-best).
"""
import argparse
import json
from pathlib import Path

from _common import add_path_args, data_root, load_splits

from fruitfresh import config
from fruitfresh.experiments import experiment_run_name, pick_best, plot_experiments, summarize_runs
from fruitfresh.train import run_dir_for


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--family", required=True, choices=list(config.EXPERIMENTS))
    add_path_args(ap)
    a = ap.parse_args()
    _, val_df, _ = load_splits(a.split_dir)
    runs = [a.family] + [r for r in (experiment_run_name(a.family, n) for n in config.EXPERIMENTS[a.family])
                         if (run_dir_for(a.work_dir, r, a.seed) / "model.keras").exists()]
    table = summarize_runs(a.work_dir, runs, a.seed, val_df, data_root(a))
    print(table.sort_values("val_macro_f1", ascending=False).to_string(index=False))
    out = Path(a.results_dir)
    out.mkdir(parents=True, exist_ok=True)
    table.to_csv(out / f"experiments_{a.family}.csv", index=False)
    plot_experiments(table, a.family).savefig(out / f"experiments_{a.family}.png", dpi=200, bbox_inches="tight")
    best = pick_best(table)["run"]
    (out / f"best_{a.family}.json").write_text(json.dumps({"family": a.family, "best_run": best}, indent=2))
    print("\nbest run on VALIDATION:", best)


if __name__ == "__main__":
    main()
