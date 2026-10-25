import json
import zipfile

import pandas as pd
import pytest

from fake_dataset import make_fake_dataset
from fruitfresh import config
from fruitfresh.colab import backup_workspace, restore_workspace, setup_workspace
from fruitfresh.data_index import build_index, get_dataset_root, resolve_dataset_root
from fruitfresh.splits import make_time_block_split, save_split, verify_split
from fruitfresh.transfer import make_data_bundle, verify_data


@pytest.fixture(scope="module")
def fake_root(tmp_path_factory):
    return make_fake_dataset(tmp_path_factory.mktemp("fake"), sources_per_class=40)


def test_workspace_is_local(tmp_path, monkeypatch):
    monkeypatch.setenv("FRUIT_WORK_DIR", str(tmp_path / "w"))
    w = setup_workspace()
    assert w == tmp_path / "w" and "drive" not in str(w).lower()


def test_backup_restore(tmp_path):
    w = tmp_path / "ws"
    (w / "runs" / "a" / "seed42").mkdir(parents=True)
    (w / "runs" / "a" / "seed42" / "model.keras").write_bytes(b"m")
    (w / "runs" / "a" / "seed42" / "checkpoint_all.keras").write_bytes(b"dup")
    z = backup_workspace(w, download=False)
    assert "runs/a/seed42/checkpoint_all.keras" not in zipfile.ZipFile(z).namelist()
    restore_workspace(tmp_path / "new", z)
    assert (tmp_path / "new" / "runs" / "a" / "seed42" / "model.keras").read_bytes() == b"m"


def test_data_root_resolution(fake_root, monkeypatch):
    monkeypatch.setenv("FRUIT_DATA_ROOT", str(fake_root))
    assert get_dataset_root() == resolve_dataset_root(fake_root)
    assert (get_dataset_root() / "train").is_dir()


def test_index_dedupes_and_bundle_roundtrip(fake_root, tmp_path, monkeypatch):
    monkeypatch.setattr(config, "SPLIT_PARAMS", {**config.SPLIT_PARAMS, "block_size": 6, "buffer_seconds": 20})
    root = resolve_dataset_root(fake_root)
    index = build_index(root)
    assert index.attrs["exact_duplicates_dropped"] > 0           # files copied into both Kaggle folders
    assert index.source_id.nunique() < len(index)                # many augmented copies per source
    idx_split, _ = make_time_block_split(index, block_size=6, buffer_seconds=20)
    stats = verify_split(idx_split, 20)
    save_split(idx_split, tmp_path / "splits", {}, stats)
    summary = verify_data(tmp_path / "splits", root)            # everything present
    assert sum(v["images"] for v in summary.values()) == sum(stats["images"].values())

    z = make_data_bundle(root, tmp_path / "splits", tmp_path / "bundle.zip")
    names = zipfile.ZipFile(z).namelist()
    assert "data/splits/train.csv" in names and "bundle_manifest.json" in names
    assert any(n.startswith("data/raw/train/") for n in names)
    target = tmp_path / "dgx"
    zipfile.ZipFile(z).extractall(target)                        # "unzip at the project root"
    assert verify_data(target / "data" / "splits", resolve_dataset_root(target / "data" / "raw"))


def test_verify_data_detects_missing_image(fake_root, tmp_path):
    root = resolve_dataset_root(fake_root)
    index = build_index(root)
    idx_split, _ = make_time_block_split(index, block_size=6, buffer_seconds=20)
    save_split(idx_split, tmp_path / "s", {}, verify_split(idx_split, 20))
    train = pd.read_csv(tmp_path / "s" / "train.csv")
    with pytest.raises(FileNotFoundError):
        verify_data(tmp_path / "s", tmp_path / "empty_root")
    assert len(train) > 0
