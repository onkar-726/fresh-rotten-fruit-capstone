"""Leakage audit: how similar is each test photo to its nearest training photo?

Tiny grey-scale thumbnails + cosine similarity: a cheap proxy for "is this test photo practically a
duplicate of a training photo?". Compare a plain random split with the time-block split to SHOW the
difference (notebook 01).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image


def thumbnail_features(paths, root, size: int = 24) -> np.ndarray:
    feats = []
    for p in paths:
        im = Image.open(Path(root) / p).convert("L").resize((size, size), Image.BILINEAR)
        v = np.asarray(im, dtype="float32").ravel()
        v -= v.mean()
        feats.append(v / (np.linalg.norm(v) + 1e-8))
    return np.stack(feats)


def nearest_train_similarity(index_split: pd.DataFrame, root, size: int = 24) -> np.ndarray:
    """For every ORIGINAL test photo: cosine similarity to the most similar ORIGINAL train photo of the same class."""
    kept = index_split[index_split.is_original].drop_duplicates("source_id")
    out = []
    for _, g in kept.groupby("class_name"):
        tr, te = g[g.split == "train"], g[g.split == "test"]
        if tr.empty or te.empty:
            continue
        out.append((thumbnail_features(te.image_path, root, size) @ thumbnail_features(tr.image_path, root, size).T).max(1))
    return np.concatenate(out)


def similarity_summary(sims: np.ndarray, thresholds=(0.95, 0.98)) -> dict:
    d = {"n": int(len(sims)), "median": float(np.median(sims)), "mean": float(sims.mean())}
    for t in thresholds:
        d[f"share_above_{t}"] = float((sims >= t).mean())
    return d
