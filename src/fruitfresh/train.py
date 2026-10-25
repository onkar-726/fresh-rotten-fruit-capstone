"""Training of both project models, multi-GPU aware and resumable.

Output layout (inside the work directory, default <project>/workspace):

    runs/<run_name>/seed<seed>/
        model.keras          final (best-validation) model - self-contained, raw 0..255 RGB input
        stage1.keras         head-only model (MobileNetV2 only; lets a crashed run resume)
        history.csv          per-epoch metrics of all stages
        train_info.json      settings, hardware, timings, library versions, trainable-layer reports
"""
from __future__ import annotations

import json
import platform
import time
from pathlib import Path

import pandas as pd
import tensorflow as tf
from tensorflow import keras

from .config import IMAGE_SIZE, MODEL_SPECS, SEED, SMOKE
from .datasets import compute_class_weights, make_dataset
from .distributed import Hardware, configure_hardware
from .models import build_model, set_trainable_stage

SMOKE_MAX_TRAIN_IMAGES = 192


def run_dir_for(work_dir, run_name: str, seed: int) -> Path:
    return Path(work_dir) / "runs" / run_name / f"seed{seed}"


def _compile(model: keras.Model, lr: float) -> None:
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr),
        loss=keras.losses.SparseCategoricalCrossentropy(),  # integer labels = categorical cross-entropy
        metrics=["accuracy"],
    )


def _callbacks(checkpoint: Path, patience: int):
    return [
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=patience, restore_best_weights=True, verbose=1),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=max(1, patience // 2), min_lr=1e-7, verbose=1
        ),
        keras.callbacks.ModelCheckpoint(str(checkpoint), monitor="val_loss", save_best_only=True),
    ]


def _fit_stage(hw: Hardware, model, stage, lr, epochs, patience, train_ds, val_ds, run_dir, verbose):
    ckpt = run_dir / f"checkpoint_{stage}.keras"
    with hw.strategy.scope():
        _compile(model, hw.scale_lr(lr))
        t0 = time.time()
        hist = model.fit(train_ds, validation_data=val_ds, epochs=epochs, callbacks=_callbacks(ckpt, patience),
                         shuffle=False,  # the tf.data pipeline already shuffles every epoch
                         verbose=verbose)
        best = keras.models.load_model(ckpt)  # guaranteed best-val_loss weights
    df = pd.DataFrame(hist.history)
    df.insert(0, "stage", stage)
    return best, df, time.time() - t0


def train_model(
    name: str,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    dataset_root,
    work_dir,
    seed: int = SEED,
    run_name: str | None = None,
    overrides: dict | None = None,
    stage1_from=None,
    force: bool = False,
    pretrained: bool = True,
    weights_path=None,
    hardware: Hardware | None = None,
    smoke: bool = SMOKE,
    verbose: int = 2,
) -> dict:
    """Train one model (or return the stored result if that run already finished).

    name        "baseline_cnn" or "mobilenet_v2"
    run_name    folder name of this run (default = name); experiments use their own
    overrides   dict replacing keys of MODEL_SPECS[name] for this run only
    stage1_from path of an existing stage-1 model to start stage 2 from
    hardware    result of configure_hardware(); default = one device, batch 32
    """
    hw = hardware or configure_hardware()
    spec = {**MODEL_SPECS[name], **(overrides or {})}
    run_name = run_name or name
    image_size = tuple(spec.get("image_size") or IMAGE_SIZE)
    run_dir = run_dir_for(work_dir, run_name, seed)
    run_dir.mkdir(parents=True, exist_ok=True)
    final_path, info_path = run_dir / "model.keras", run_dir / "train_info.json"
    if final_path.exists() and info_path.exists() and not force:
        print(f"[{run_name}] already trained -> {final_path} (use force / --force to retrain)")
        return json.loads(info_path.read_text())

    keras.utils.set_random_seed(seed)
    if smoke:
        train_df = train_df.sample(min(len(train_df), SMOKE_MAX_TRAIN_IMAGES), random_state=seed)
    class_weight = compute_class_weights(train_df["label"].to_numpy()) if spec["class_weights"] else None
    train_ds = make_dataset(train_df, dataset_root, image_size=image_size, batch_size=hw.global_batch_size,
                            training=True, augment=spec["augment"], class_weights=class_weight, seed=seed)
    val_ds = make_dataset(val_df, dataset_root, image_size=image_size, batch_size=hw.global_batch_size, training=False)

    info = {
        "run_name": run_name, "model": name, "title": spec["title"], "seed": seed,
        "pretrained": bool(pretrained or weights_path),
        "overrides": {k: list(v) if isinstance(v, tuple) else v for k, v in (overrides or {}).items()},
        "n_train_images": int(len(train_df)), "n_val_images": int(len(val_df)),
        "image_size": list(image_size), "augment": spec["augment"],
        "class_weights": class_weight, "dropout": spec["dropout"], "hardware": hw.describe(), "stages": {},
        "versions": {"tensorflow": tf.__version__, "keras": keras.__version__, "python": platform.python_version()},
    }
    histories, total = [], 0.0
    e = (lambda n: min(n, 1)) if smoke else (lambda n: n)

    if spec["kind"] == "scratch":
        with hw.strategy.scope():
            model = build_model(name, image_size=image_size, pretrained=False, dropout=spec["dropout"])
            info["stages"]["all"] = set_trainable_stage(model, "all")
        model, h, sec = _fit_stage(hw, model, "all", spec["lr"], e(spec["epochs"]), spec["patience"],
                                   train_ds, val_ds, run_dir, verbose)
        histories.append(h); total += sec
    else:
        stage1_path = run_dir / "stage1.keras"
        if stage1_from is not None:
            print(f"[{run_name}] starting stage 2 from {stage1_from}")
            with hw.strategy.scope():
                model = keras.models.load_model(stage1_from)
            info["stage1_from"] = str(stage1_from)
        elif stage1_path.exists() and not force:
            print(f"[{run_name}] resuming from {stage1_path}")
            with hw.strategy.scope():
                model = keras.models.load_model(stage1_path)
        else:
            with hw.strategy.scope():
                model = build_model(name, image_size=image_size, pretrained=pretrained, dropout=spec["dropout"],
                                    alpha=spec["alpha"], weights_path=weights_path)
                info["stages"]["head"] = set_trainable_stage(model, "head")
            model, h, sec = _fit_stage(hw, model, "head", spec["head_lr"], e(spec["head_epochs"]),
                                       spec["patience"], train_ds, val_ds, run_dir, verbose)
            model.save(stage1_path)
            histories.append(h); total += sec
        if spec["ft_epochs"] > 0:
            with hw.strategy.scope():
                info["stages"]["finetune"] = set_trainable_stage(model, "finetune", spec["ft_layers"])
            model, h, sec = _fit_stage(hw, model, "finetune", spec["ft_lr"], e(spec["ft_epochs"]),
                                       spec["patience"], train_ds, val_ds, run_dir, verbose)
            histories.append(h); total += sec
        else:
            print(f"[{run_name}] ft_epochs=0 -> stage-1 model is the final model")

    model.save(final_path)
    if histories:
        hist = pd.concat(histories, ignore_index=True)
    else:  # stage 1 reused and no fine-tuning: take the stored stage-1 history
        src = Path(stage1_from).parent / "history.csv" if stage1_from else run_dir / "history.csv"
        hist = pd.read_csv(src)
        hist = hist[hist["stage"] == "head"].drop(columns=["epoch"])
    hist.insert(0, "epoch", range(1, len(hist) + 1))
    hist.to_csv(run_dir / "history.csv", index=False)
    best = hist.loc[hist["val_loss"].idxmin()]
    info.update(
        training_seconds=round(total, 1), epochs_run=int(len(hist)), best_epoch=int(best["epoch"]),
        best_val_loss=float(best["val_loss"]), best_val_accuracy=float(best["val_accuracy"]),
        params=int(model.count_params()),
    )
    info_path.write_text(json.dumps(info, indent=2))
    print(f"[{run_name}] done in {total/60:.1f} min on {hw.num_replicas} x {hw.device_type}, "
          f"best val_loss {info['best_val_loss']:.4f}")
    return info
