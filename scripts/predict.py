"""Command-line prediction (same pipeline as the web app).

    python scripts/predict.py --model auto --image apple.jpg banana.jpg
    python scripts/predict.py --model workspace/runs/mobilenet_v2/seed42/model.keras --image x.jpg --json
"""
import argparse
import json

from _common import add_path_args
from PIL import Image

from fruitfresh.paths import resolve_model
from fruitfresh.predict import FruitPredictor


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="auto", help="'auto', a run name, or a .keras/.tflite path")
    ap.add_argument("--image", nargs="+", required=True)
    ap.add_argument("--json", action="store_true")
    add_path_args(ap)
    a = ap.parse_args()
    predictor = FruitPredictor(resolve_model(a.model, a.work_dir, a.results_dir, a.seed))
    for path in a.image:
        r = predictor.predict(Image.open(path))
        if a.json:
            print(json.dumps({"image": path, **r}))
        else:
            warn = "  (LOW CONFIDENCE)" if r["uncertain"] else ""
            print(f"{path}: {r['label']} | {r['status'].upper()} | confidence {r['confidence']:.1%}{warn}")


if __name__ == "__main__":
    main()
