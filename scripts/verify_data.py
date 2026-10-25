"""Check that the split CSVs and the images are all present and unchanged (run this on the DGX after transfer).

    python scripts/verify_data.py --split-dir data/splits --data-root data/raw
"""
import argparse
import json
import sys

from _common import add_path_args, data_root

from fruitfresh.transfer import verify_data


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_path_args(ap)
    ap.add_argument("--decode", action="store_true", help="also open every image (slow)")
    a = ap.parse_args()
    try:
        print(json.dumps(verify_data(a.split_dir, data_root(a), a.decode), indent=2))
        print("DATA OK")
    except (FileNotFoundError, AssertionError) as e:
        print("DATA PROBLEM:", e)
        sys.exit(1)


if __name__ == "__main__":
    main()
