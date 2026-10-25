"""Move data between machines (Colab / PC -> DGX) and verify it arrived intact.

Typical flow:  Colab builds the split  ->  make_data_bundle()  ->  scp the zip to the DGX  ->
unzip at the project root  ->  verify_data()  (scripts/verify_data.py).
"""
from __future__ import annotations

import hashlib
import json
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from .splits import SPLITS, load_split


def make_data_bundle(dataset_root, split_dir, out_zip, include_images: bool = True) -> Path:
    """Zip the split CSVs (+ manifest) and, optionally, every image they reference.

    Layout inside the zip (so unzipping at the project root just works):
        data/splits/{train,validation,test}.csv, split_manifest.json
        data/raw/<train|test>/<class>/<file>       (only if include_images)
        bundle_manifest.json
    """
    dataset_root, split_dir, out_zip = Path(dataset_root), Path(split_dir), Path(out_zip)
    frames = [pd.read_csv(split_dir / f"{s}.csv") for s in SPLITS]
    paths = sorted(set(pd.concat(frames)["image_path"]))
    out_zip.parent.mkdir(parents=True, exist_ok=True)
    n_bytes = 0
    with zipfile.ZipFile(out_zip, "w", zipfile.ZIP_STORED) as z:  # images are already compressed
        for f in sorted(split_dir.glob("*")):
            if f.is_file() and f.name != ".gitkeep":
                z.write(f, f"data/splits/{f.name}")
        if include_images:
            for p in paths:
                src = dataset_root / p
                z.write(src, f"data/raw/{p}")
                n_bytes += src.stat().st_size
        z.writestr("bundle_manifest.json", json.dumps({
            "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "includes_images": include_images, "n_images": len(paths) if include_images else 0,
            "image_bytes": n_bytes,
        }, indent=2))
    return out_zip


def verify_data(split_dir, dataset_root, check_decodable: bool = False) -> dict:
    """Raise if anything is missing/changed; return a summary otherwise."""
    train, val, test = load_split(split_dir)          # also checks the manifest SHA-256 of every CSV
    summary = {}
    missing = []
    for name, df in zip(SPLITS, (train, val, test)):
        miss = [p for p in df["image_path"] if not (Path(dataset_root) / p).is_file()]
        missing += miss
        summary[name] = {"images": int(len(df)), "sources": int(df["source_id"].nunique()), "missing": len(miss)}
    if missing:
        raise FileNotFoundError(f"{len(missing)} images listed in the split are missing under {dataset_root}, "
                                f"e.g. {missing[:3]}")
    if check_decodable:
        from PIL import Image

        for df in (train, val, test):
            for p in df["image_path"]:
                with Image.open(Path(dataset_root) / p) as im:
                    im.verify()
    return summary


def sha256_file(path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()
