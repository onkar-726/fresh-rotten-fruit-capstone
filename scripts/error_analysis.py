"""Error analysis of one evaluated run: top confusions, mistakes gallery, Grad-CAM, written report.

    python scripts/error_analysis.py --run mobilenet_v2
Writes results/error_analysis/<run>/
"""
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from _common import add_path_args, data_root
from PIL import Image

from fruitfresh import error_analysis as EA
from fruitfresh.gradcam import plot_gradcam_grid
from fruitfresh.train import run_dir_for


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True)
    ap.add_argument("--view", default="test_original", choices=["test_original", "test_full", "validation"])
    ap.add_argument("--n", type=int, default=12, help="images in the galleries")
    add_path_args(ap)
    a = ap.parse_args()
    from tensorflow import keras

    run_dir = run_dir_for(a.work_dir, a.run, a.seed)
    metrics = json.loads((run_dir / "eval" / "metrics.json").read_text())      # run scripts/evaluate.py first
    pred = pd.read_csv(run_dir / "eval" / f"predictions_{a.view}.csv")
    root = data_root(a)
    out = Path(a.results_dir) / "error_analysis" / a.run
    out.mkdir(parents=True, exist_ok=True)

    EA.error_table(pred).to_csv(out / f"mistakes_{a.view}.csv", index=False)
    EA.top_confusions(metrics[a.view]["confusion_matrix"]).to_csv(out / f"top_confusions_{a.view}.csv", index=False)
    print(EA.write_error_report(a.run, pred, metrics[a.view]["confusion_matrix"], out / f"error_report_{a.view}.md"))
    fig = EA.misclassified_gallery(pred, root, n=a.n)
    if fig is not None:
        fig.savefig(out / f"mistakes_gallery_{a.view}.png", dpi=150, bbox_inches="tight")

    model = keras.models.load_model(run_dir / "model.keras")
    size = tuple(model.input_shape[1:3])
    errs = EA.error_table(pred).head(a.n // 2)
    right = pred[pred.y_true == pred.y_pred].sample(min(a.n // 2, int((pred.y_true == pred.y_pred).sum())), random_state=0)
    items = [(r.image_path, int(pred.loc[pred.image_path == r.image_path, "y_true"].iloc[0])) for r in errs.itertuples()]
    items += [(r.image_path, int(r.y_true)) for r in right.itertuples()]
    if items:
        imgs = [np.array(Image.open(Path(root) / p).convert("RGB").resize(size)) for p, _ in items]
        plot_gradcam_grid(model, imgs, [t for _, t in items]).savefig(out / f"gradcam_{a.view}.png", dpi=150,
                                                                      bbox_inches="tight")
    plt.close("all")
    print("written to", out)


if __name__ == "__main__":
    main()
