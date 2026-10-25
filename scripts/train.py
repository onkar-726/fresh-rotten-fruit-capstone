"""Train Model 1 (baseline CNN) or Model 2 (MobileNetV2) - on one GPU or all GPUs of the DGX.

    python scripts/train.py --model baseline_cnn --split-dir data/splits --data-root data/raw
    python scripts/train.py --model mobilenet_v2 --config configs/dgx_b200.json
    python scripts/train.py --model mobilenet_v2 --gpus 2 --set ft_layers=20 --run-name my_variant
"""
import argparse

from _common import add_hardware_args, add_path_args, data_root, hardware_from_args, load_splits, parse_set, profile

from fruitfresh import config
from fruitfresh.train import train_model


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, choices=list(config.MODEL_SPECS))
    ap.add_argument("--run-name", default=None, help="output folder name (default: the model name)")
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE",
                    help="override model settings, e.g. ft_layers=20 augment=false epochs=5")
    add_path_args(ap)
    add_hardware_args(ap)
    a = ap.parse_args()

    hw = hardware_from_args(a)
    print("hardware:", hw.describe())
    train_df, val_df, _ = load_splits(a.split_dir)     # the TEST split is not used for training
    overrides = {**profile(a).get("overrides", {}).get(a.model, {}), **parse_set(a.set)}
    train_model(a.model, train_df, val_df, data_root(a), a.work_dir, seed=a.seed, run_name=a.run_name,
                overrides=overrides or None, force=a.force, pretrained=not a.no_pretrained,
                weights_path=a.weights_file, hardware=hw, smoke=a.smoke or config.SMOKE)


if __name__ == "__main__":
    main()
