"""Hardware setup: one GPU, several GPUs (MirroredStrategy) or - for tests - several logical CPUs.

The DGX B200 has several GPUs. ``configure_hardware(gpus="all")`` uses every visible GPU with
``tf.distribute.MirroredStrategy`` (synchronous data parallelism, NCCL all-reduce on GPUs).
Select fewer GPUs with ``gpus="2"`` / ``gpus="0,1"`` or the CUDA_VISIBLE_DEVICES variable.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

import tensorflow as tf
from tensorflow import keras

from .config import HARDWARE_DEFAULTS


@dataclass
class Hardware:
    strategy: tf.distribute.Strategy
    num_replicas: int
    device_type: str            # "GPU" or "CPU"
    device_names: list
    mixed_precision: str
    per_gpu_batch_size: int
    lr_scaling: str

    @property
    def global_batch_size(self) -> int:
        return self.per_gpu_batch_size * self.num_replicas

    def scale_lr(self, lr: float) -> float:
        """Scale the learning rate with the number of replicas (default: no scaling)."""
        factor = {"none": 1.0, "sqrt": self.num_replicas ** 0.5, "linear": float(self.num_replicas)}[self.lr_scaling]
        return lr * factor

    def describe(self) -> dict:
        return {
            "device_type": self.device_type,
            "num_replicas": self.num_replicas,
            "devices": self.device_names,
            "per_replica_batch_size": self.per_gpu_batch_size,
            "global_batch_size": self.global_batch_size,
            "mixed_precision": self.mixed_precision,
            "lr_scaling": self.lr_scaling,
            "strategy": type(self.strategy).__name__,
        }


def _make_fake_cpu_replicas(n: int) -> None:
    cpus = tf.config.list_physical_devices("CPU")
    try:
        tf.config.set_logical_device_configuration(cpus[0], [tf.config.LogicalDeviceConfiguration() for _ in range(n)])
    except RuntimeError:  # TensorFlow already initialised -> keep whatever is configured
        pass


def _select_gpus(spec: str, gpus: list):
    spec = str(spec).strip().lower()
    if spec in ("all", ""):
        return gpus
    if "," in spec:
        return [gpus[int(i)] for i in spec.split(",")]
    n = int(spec)
    if n > len(gpus):
        raise ValueError(f"Requested {n} GPUs but only {len(gpus)} are visible")
    return gpus[:n]


def configure_hardware(
    gpus: str = HARDWARE_DEFAULTS["gpus"],
    per_gpu_batch_size: int = HARDWARE_DEFAULTS["per_gpu_batch_size"],
    mixed_precision: str = HARDWARE_DEFAULTS["mixed_precision"],
    lr_scaling: str = HARDWARE_DEFAULTS["lr_scaling"],
) -> Hardware:
    """Call ONCE at the start of a process, before any TensorFlow operation."""
    fake = int(os.getenv("FRUIT_FAKE_REPLICAS", "0"))
    if fake > 1:
        _make_fake_cpu_replicas(fake)

    physical = tf.config.list_physical_devices("GPU")
    for g in physical:
        try:
            tf.config.experimental.set_memory_growth(g, True)
        except RuntimeError:
            pass

    if physical:
        chosen = _select_gpus(gpus, physical)
        try:
            tf.config.set_visible_devices(chosen, "GPU")
        except RuntimeError:  # already initialised: the strategy below still uses all visible GPUs
            pass
        n = len(chosen)
        strategy = tf.distribute.MirroredStrategy() if n > 1 else tf.distribute.OneDeviceStrategy("/gpu:0")
        device_type, names = "GPU", [g.name for g in chosen]
    elif fake > 1:
        names = [f"CPU:{i}" for i in range(fake)]
        strategy = tf.distribute.MirroredStrategy(names)
        device_type, n = "CPU", fake
    else:
        strategy, device_type, n, names = tf.distribute.get_strategy(), "CPU", 1, ["CPU:0"]

    policy = {"off": "float32", "bf16": "mixed_bfloat16", "fp16": "mixed_float16"}[mixed_precision]
    keras.mixed_precision.set_global_policy(policy)
    return Hardware(strategy, n, device_type, names, mixed_precision, per_gpu_batch_size, lr_scaling)


def hardware_from_profile(path=None, **cli_overrides) -> Hardware:
    """Build the Hardware object from a JSON profile (e.g. configs/dgx_b200.json) plus optional overrides.

    Precedence: keyword overrides > profile file > config.HARDWARE_DEFAULTS. Used by the scripts and notebook 02.
    """
    import json
    from pathlib import Path

    settings = dict(HARDWARE_DEFAULTS)
    if path:
        settings.update(json.loads(Path(path).read_text()).get("hardware", {}))
    settings.update({k: v for k, v in cli_overrides.items() if v is not None})
    return configure_hardware(**settings)
