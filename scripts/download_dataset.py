"""Download the public Kaggle dataset (needs internet) and optionally link it into data/raw.

    python scripts/download_dataset.py --link data/raw
"""
import argparse
import os
import shutil

from _common import ROOT, config  # noqa: F401

from fruitfresh.data_index import resolve_dataset_root


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--link", default=None, help="create this folder as a symlink to the downloaded dataset")
    ap.add_argument("--copy", action="store_true", help="copy instead of symlink")
    a = ap.parse_args()
    import kagglehub

    root = resolve_dataset_root(kagglehub.dataset_download(config.KAGGLE_DATASET))
    print("dataset root:", root)
    if a.link:
        dest = os.path.abspath(a.link)
        if os.path.lexists(dest):
            print("destination exists, leaving it untouched:", dest)
        elif a.copy:
            shutil.copytree(root, dest)
            print("copied to", dest)
        else:
            os.symlink(root, dest)
            print("symlinked", dest, "->", root)
    print("\nUse it with:  --data-root", root, " (or export FRUIT_DATA_ROOT=...)")


if __name__ == "__main__":
    main()
