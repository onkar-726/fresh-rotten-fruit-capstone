"""Shared helpers for all scripts (paths, hardware options). Imported as ``from _common import ...``."""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fruitfresh import config  # noqa: E402


def add_path_args(ap: argparse.ArgumentParser) -> None:
    ap.add_argument("--split-dir", default=str(config.DEFAULT_SPLIT_DIR), help="folder with train/validation/test.csv")
    ap.add_argument("--data-root", default=None,
                    help="folder containing the Kaggle train/ and test/ (default: $FRUIT_DATA_ROOT, data/raw, or Kaggle download)")
    ap.add_argument("--work-dir", default=str(config.DEFAULT_WORK_DIR), help="where runs are stored")
    ap.add_argument("--results-dir", default=str(config.DEFAULT_RESULTS_DIR), help="where tables/figures are written")
    ap.add_argument("--seed", type=int, default=config.SEED)


def add_hardware_args(ap: argparse.ArgumentParser) -> None:
    ap.add_argument("--config", default=None, help="JSON profile, e.g. configs/dgx_b200.json")
    ap.add_argument("--gpus", default=None, help='"all" (default), a number, or ids like "0,1,2"')
    ap.add_argument("--per-gpu-batch-size", type=int, default=None)
    ap.add_argument("--mixed-precision", choices=["off", "bf16", "fp16"], default=None)
    ap.add_argument("--lr-scaling", choices=["none", "sqrt", "linear"], default=None)
    ap.add_argument("--weights-file", default=None, help="local MobileNetV2 ImageNet weights (offline machines)")
    ap.add_argument("--no-pretrained", action="store_true", help="train MobileNetV2 from random weights (tests only)")
    ap.add_argument("--smoke", action="store_true", help="tiny/fast run for testing the pipeline")
    ap.add_argument("--force", action="store_true", help="retrain even if a finished run exists")


def profile(args) -> dict:
    return json.loads(Path(args.config).read_text()) if args.config else {}


def hardware_from_args(args):
    """CLI flag > JSON profile > defaults."""
    from fruitfresh.distributed import hardware_from_profile

    return hardware_from_profile(args.config, gpus=args.gpus, per_gpu_batch_size=args.per_gpu_batch_size,
                                 mixed_precision=args.mixed_precision, lr_scaling=args.lr_scaling)


def parse_set(items) -> dict:
    """['ft_layers=20', 'augment=false'] -> {'ft_layers': 20, 'augment': False}"""
    out = {}
    for item in items or []:
        key, _, value = item.partition("=")
        try:
            out[key] = json.loads(value)
        except json.JSONDecodeError:
            out[key] = value
    return out


def load_splits(split_dir):
    from fruitfresh.splits import load_split

    return load_split(split_dir)  # (train, validation, test); also verifies the manifest hashes


def data_root(args) -> Path:
    from fruitfresh.data_index import get_dataset_root

    return get_dataset_root(args.data_root)
