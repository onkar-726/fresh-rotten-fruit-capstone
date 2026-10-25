"""Download the MobileNetV2 ImageNet weights once and copy them to assets/ (for offline machines).

Run on a machine WITH internet (Colab / PC), then copy assets/*.h5 to the DGX project folder and train with
    python scripts/train.py --model mobilenet_v2 --weights-file assets/<file>.h5
"""
import argparse
import shutil
from pathlib import Path

from _common import ROOT


def main():
    argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter).parse_args()
    from tensorflow import keras

    keras.applications.MobileNetV2(input_shape=(224, 224, 3), include_top=False, weights="imagenet")
    cache = Path.home() / ".keras" / "models"
    found = sorted(cache.glob("mobilenet_v2_weights_tf_dim_ordering_tf_kernels_1.0_224_no_top*.h5"))
    if not found:
        raise SystemExit(f"Weights file not found in {cache}")
    dest = ROOT / "assets" / found[0].name
    dest.parent.mkdir(exist_ok=True)
    shutil.copy(found[0], dest)
    print("saved:", dest, f"({dest.stat().st_size / 1e6:.1f} MB)")
    print("copy to the DGX:  scp", dest, "<user>@<dgx-host>:<project>/assets/")


if __name__ == "__main__":
    main()
