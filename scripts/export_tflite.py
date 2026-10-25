"""Export a trained run to TensorFlow Lite and check it matches the Keras model.

    python scripts/export_tflite.py --run mobilenet_v2 --quantization dynamic
"""
import argparse
import json
from pathlib import Path

import numpy as np
from _common import add_path_args, data_root, load_splits
from PIL import Image

from fruitfresh.export import export_tflite, validate_tflite
from fruitfresh.paths import resolve_model


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", default="auto", help="run name, 'auto' (= final model) or a .keras path")
    ap.add_argument("--quantization", choices=["none", "dynamic", "float16"], default="dynamic")
    ap.add_argument("--n-check", type=int, default=16)
    add_path_args(ap)
    a = ap.parse_args()
    keras_path = resolve_model(a.run, a.work_dir, a.results_dir, a.seed)
    out = Path(a.results_dir) / "export" / f"{keras_path.parent.parent.name}.tflite"
    tflite = export_tflite(keras_path, out, a.quantization)
    from tensorflow import keras
    size = tuple(keras.models.load_model(keras_path).input_shape[1:3])
    _, _, test_df = load_splits(a.split_dir)
    root = data_root(a)
    imgs = np.stack([np.array(Image.open(Path(root) / p).convert("RGB").resize(size))
                     for p in test_df.sample(min(a.n_check, len(test_df)), random_state=0).image_path]).astype("float32")
    report = validate_tflite(keras_path, tflite, imgs)
    print(json.dumps(report, indent=2))
    print("TFLite model:", tflite)


if __name__ == "__main__":
    main()
