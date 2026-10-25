"""Experiments on the DGX: one-change variants of a model, judged on VALIDATION only.

    python scripts/run_experiments.py --family mobilenet_v2 --names head_only ft_top20 ft_top80
    python scripts/run_experiments.py --family baseline_cnn --names all --config configs/dgx_b200.json
Then:  python scripts/summarize_experiments.py --family mobilenet_v2
"""
import argparse

from _common import add_hardware_args, add_path_args, data_root, hardware_from_args, load_splits

from fruitfresh import config
from fruitfresh.experiments import run_experiments


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--family", required=True, choices=list(config.EXPERIMENTS))
    ap.add_argument("--names", nargs="+", default=["all"], help="experiment names or 'all'")
    add_path_args(ap)
    add_hardware_args(ap)
    a = ap.parse_args()
    names = list(config.EXPERIMENTS[a.family]) if a.names == ["all"] else a.names
    hw = hardware_from_args(a)
    train_df, val_df, _ = load_splits(a.split_dir)
    run_experiments(a.family, names, train_df, val_df, data_root(a), a.work_dir, a.seed,
                    pretrained=not a.no_pretrained, weights_path=a.weights_file, hardware=hw, force=a.force)


if __name__ == "__main__":
    main()
