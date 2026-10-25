"""Step 1 (Colab or any machine): index the dataset, create the 70/15/15 split, verify, save.

    python scripts/prepare_data.py --data-root /path/to/dataset --out data/splits
    python scripts/prepare_data.py --audit --bundle fruit_data_bundle.zip      # + leakage audit + transfer zip
"""
import argparse
import json

from _common import ROOT, config  # noqa: F401  (sets sys.path)

from fruitfresh.audit import nearest_train_similarity, similarity_summary
from fruitfresh.data_index import build_index, find_unreadable, get_dataset_root, summarize_index
from fruitfresh.splits import (make_random_group_split, make_time_block_split, save_split, verify_split)
from fruitfresh.transfer import make_data_bundle


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", default=None)
    ap.add_argument("--out", default=str(config.DEFAULT_SPLIT_DIR))
    ap.add_argument("--seed", type=int, default=config.SEED)
    ap.add_argument("--method", choices=["time_block", "random_group"], default="time_block",
                    help="time_block = official leakage-safe split; random_group = naive comparison split")
    ap.add_argument("--check-images", action="store_true", help="try to open every image (slow, finds corrupt files)")
    ap.add_argument("--audit", action="store_true", help="print the near-duplicate leakage audit")
    ap.add_argument("--bundle", default=None, help="also write a zip for transfer to the DGX")
    ap.add_argument("--bundle-no-images", action="store_true", help="bundle only the split CSVs (DGX downloads the images itself)")
    a = ap.parse_args()

    root = get_dataset_root(a.data_root)
    print("dataset root:", root)
    index = build_index(root)
    print(json.dumps(summarize_index(index), indent=2))
    if a.check_images:
        bad = find_unreadable(index, root)
        print(f"unreadable images: {len(bad)}", bad[:5])

    params = {**config.SPLIT_PARAMS, "seed": a.seed, "method": a.method}
    if a.method == "time_block":
        idx_split, rep = make_time_block_split(index, seed=a.seed, **config.SPLIT_PARAMS)
        stats = verify_split(idx_split, config.SPLIT_PARAMS["buffer_seconds"])
        params.update(rep)
    else:
        idx_split = make_random_group_split(index, seed=a.seed)
        stats = verify_split(idx_split)
    manifest = save_split(idx_split, a.out, params, stats)
    print("\nsplit saved to", a.out)
    print("images :", stats["images"])
    print("sources:", stats["sources"])
    print("min cross-split gap (s):", stats["min_cross_split_gap_seconds"])
    total = sum(stats["images"].values())
    print({k: f"{v / total:.1%}" for k, v in stats["images"].items()}, "of kept images")
    print("manifest hashes:", {k: v[:12] + "..." for k, v in manifest["files"].items()})

    if a.audit:
        print("\nleakage audit (share of test originals with a near-identical train photo):")
        print(" random split    :", similarity_summary(nearest_train_similarity(make_random_group_split(index, a.seed), root)))
        print(" time-block split:", similarity_summary(nearest_train_similarity(idx_split, root)))
    if a.bundle:
        z = make_data_bundle(root, a.out, a.bundle, include_images=not a.bundle_no_images)
        print(f"\nbundle written: {z} ({z.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
