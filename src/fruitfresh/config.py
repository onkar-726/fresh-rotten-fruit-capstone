"""Central configuration for the Fresh-versus-Rotten Fruit Classification project.

Everything that is a "setting" lives here so notebooks and scripts stay short.
Environment variables (all optional):

    FRUIT_DATA_ROOT     folder that contains train/ and test/ of the Kaggle dataset
    FRUIT_WORK_DIR      where runs (models, histories, predictions) are written (default: <project>/workspace)
    FRUIT_RESULTS_DIR   where tables/figures/reports are written (default: <project>/results)
    FRUIT_SPLIT_DIR     folder with train/validation/test.csv (default: <project>/data/splits)
    FRUIT_SMOKE=1       tiny/fast mode used by the automated tests (1 epoch, few images)
    FRUIT_IMAGE_SIZE    override the 224 input size (tests use 64)
    FRUIT_FAKE_REPLICAS number of *logical CPU* replicas, to test multi-GPU code without GPUs
"""
from __future__ import annotations

import json
import os
from pathlib import Path

PROJECT_NAME = "Fresh-versus-Rotten Fruit Classification"
KAGGLE_DATASET = "sriramr/fruits-fresh-and-rotten-for-classification"

# Class order is FIXED for the whole project (label index = position in this list).
CLASS_NAMES = [
    "freshapples",
    "freshbanana",
    "freshoranges",
    "rottenapples",
    "rottenbanana",
    "rottenoranges",
]
NUM_CLASSES = len(CLASS_NAMES)
CLASS_TO_INDEX = {name: i for i, name in enumerate(CLASS_NAMES)}
DISPLAY_NAMES = {
    "freshapples": "Fresh Apple",
    "freshbanana": "Fresh Banana",
    "freshoranges": "Fresh Orange",
    "rottenapples": "Rotten Apple",
    "rottenbanana": "Rotten Banana",
    "rottenoranges": "Rotten Orange",
}


def _int(name: str, default: int) -> int:
    return int(os.getenv(name, default))


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SMOKE = os.getenv("FRUIT_SMOKE", "0") == "1"
IMAGE_SIZE = (_int("FRUIT_IMAGE_SIZE", 224), _int("FRUIT_IMAGE_SIZE", 224))
SEED = 42

# Conventional folders (all relative to the project root).
DEFAULT_SPLIT_DIR = Path(os.getenv("FRUIT_SPLIT_DIR") or PROJECT_ROOT / "data" / "splits")
DEFAULT_RAW_DIR = PROJECT_ROOT / "data" / "raw"
DEFAULT_WORK_DIR = Path(os.getenv("FRUIT_WORK_DIR") or PROJECT_ROOT / "workspace")
DEFAULT_RESULTS_DIR = Path(os.getenv("FRUIT_RESULTS_DIR") or PROJECT_ROOT / "results")

# ---------------------------------------------------------------------------
# Data split: 70 % train / 15 % validation / 15 % test, reproducible (seed 42).
# See docs/METHODOLOGY.md - the split is done on SOURCE PHOTOS in time blocks,
# because the Kaggle files are mostly augmented copies of ~1,500 originals.
# ---------------------------------------------------------------------------
SPLIT_PARAMS = {
    "block_size": _int("FRUIT_BLOCK_SIZE", 25),
    "buffer_seconds": _int("FRUIT_BUFFER_SECONDS", 60),
    "val_fraction": 0.15,
    "test_fraction": 0.15,
}

# Targets from the proposal. They are CHECKED against real results, never assumed.
TARGETS = {"accuracy": 0.90, "macro_f1": 0.85}

# ---------------------------------------------------------------------------
# Hardware defaults (override on the command line, see docs/DGX_B200_GUIDE.md).
# ---------------------------------------------------------------------------
HARDWARE_DEFAULTS = {
    "gpus": "all",               # "all", a number ("2"), or ids ("0,1,2")
    "per_gpu_batch_size": 32,    # global batch = per_gpu_batch_size x number of GPUs
    "mixed_precision": "off",    # "off" | "bf16" (recommended to try on B200) | "fp16"
    "lr_scaling": "none",        # "none" | "sqrt" | "linear" (scale LR with the global batch)
}

# ---------------------------------------------------------------------------
# Models. Both take RAW RGB pixels (float 0..255); the right pre-processing is
# inside the model, so saved models are self-contained (training, app, TFLite).
# Any key can be overridden per run (see EXPERIMENTS).
# ---------------------------------------------------------------------------
MODEL_SPECS = {
    "baseline_cnn": {
        "title": "Baseline CNN (from scratch)",
        "kind": "scratch",
        "preprocess": "rescale_255",
        "epochs": 40,
        "lr": 1e-3,
        "patience": 8,
        "dropout": 0.3,
        "augment": True,
        "class_weights": True,
        "image_size": None,      # None -> IMAGE_SIZE
    },
    "mobilenet_v2": {
        "title": "MobileNetV2 (transfer learning)",
        "kind": "transfer",
        "preprocess": "mobilenet_v2",
        "alpha": 1.0,            # width multiplier
        "head_epochs": 10,       # stage 1: backbone frozen, train the new classifier head
        "head_lr": 1e-3,
        "ft_epochs": 20,         # stage 2: fine-tune the top backbone layers (0 = skip)
        "ft_lr": 2e-5,
        "ft_layers": 40,
        "patience": 5,
        "dropout": 0.3,
        "augment": True,
        "class_weights": True,
        "image_size": None,
    },
}

# Experiments: ONE change per run relative to the default run of that model.
# They are judged on the VALIDATION split only.
EXPERIMENTS = {
    "baseline_cnn": {
        "no_augmentation": {"augment": False},
        "no_class_weights": {"class_weights": False},
        "dropout_0.5": {"dropout": 0.5},
        "lr_3e-4": {"lr": 3e-4},
    },
    "mobilenet_v2": {
        "head_only": {"ft_epochs": 0},
        "ft_top20": {"ft_layers": 20},
        "ft_top80": {"ft_layers": 80},
        "ft_lr_1e-5": {"ft_lr": 1e-5},
        "ft_lr_5e-5": {"ft_lr": 5e-5},
        "no_class_weights": {"class_weights": False},
        "no_augmentation": {"augment": False},
        "dropout_0.5": {"dropout": 0.5},
        "alpha_0.75": {"alpha": 0.75},
        "size_160": {"image_size": (160, 160)},
    },
}
# Overrides that only change stage 2, so the default run's stage-1 model can be reused.
FINETUNE_ONLY_KEYS = {"ft_epochs", "ft_lr", "ft_layers"}


def load_json_overrides(path) -> dict:
    """Read a small JSON file such as configs/dgx_b200.json."""
    return json.loads(Path(path).read_text())
