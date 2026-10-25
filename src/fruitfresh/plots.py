"""Training-curve plots."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def plot_history(history_csv, title: str = ""):
    """Loss / accuracy curves of one run; a dashed line marks the switch between training stages."""
    h = pd.read_csv(history_csv)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    for ax, key in zip(axes, ("loss", "accuracy")):
        ax.plot(h["epoch"], h[key], label="train")
        ax.plot(h["epoch"], h[f"val_{key}"], label="validation")
        if "stage" in h and h["stage"].nunique() > 1:
            switch = h.index[h["stage"] != h["stage"].iloc[0]][0]
            ax.axvline(h["epoch"].iloc[switch] - 0.5, color="gray", ls="--", lw=1)
        ax.set_xlabel("epoch"); ax.set_ylabel(key); ax.grid(alpha=0.3); ax.legend()
    fig.suptitle(title)
    fig.tight_layout()
    return fig


def plot_training_curves(histories: dict):
    """Overlay training/validation curves of several runs. histories = {label: history.csv path}."""
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    for label, path in histories.items():
        h = pd.read_csv(path)
        for col, key in enumerate(("loss", "accuracy")):
            axes[0, col].plot(h["epoch"], h[key], label=label)
            axes[1, col].plot(h["epoch"], h[f"val_{key}"], label=label)
    for r, split in enumerate(("training", "validation")):
        for c, key in enumerate(("loss", "accuracy")):
            ax = axes[r, c]
            ax.set_title(f"{split} {key}"); ax.set_xlabel("epoch"); ax.grid(alpha=0.3); ax.legend()
    fig.tight_layout()
    return fig
