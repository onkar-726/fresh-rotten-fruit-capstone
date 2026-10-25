"""Pack everything produced on the DGX into ONE zip to copy back to your laptop.

    python scripts/package_results.py --out dgx_results.zip            # final models + results
    python scripts/package_results.py --out dgx_results.zip --all-runs # include every experiment model
On the laptop:  unzip dgx_results.zip -d <project folder>
"""
import argparse
import json
import zipfile
from pathlib import Path

from _common import ROOT, config


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="dgx_results.zip")
    ap.add_argument("--all-runs", action="store_true", help="include models of every experiment run")
    ap.add_argument("--work-dir", default=str(config.DEFAULT_WORK_DIR))
    ap.add_argument("--results-dir", default=str(config.DEFAULT_RESULTS_DIR))
    a = ap.parse_args()
    work, results = Path(a.work_dir), Path(a.results_dir)
    keep_runs = {"baseline_cnn", "mobilenet_v2"}
    for f in ("best_baseline_cnn.json", "best_mobilenet_v2.json", "final_model.json"):
        if (results / f).exists():
            d = json.loads((results / f).read_text())
            keep_runs |= {d.get("best_run") or d.get("final_model")}
    n = 0
    with zipfile.ZipFile(a.out, "w", zipfile.ZIP_DEFLATED) as z:
        for base in (work / "runs", results):
            for f in sorted(base.rglob("*")):
                if not f.is_file() or f.name.startswith("checkpoint_") or f.name == ".gitkeep":
                    continue
                rel = (Path("workspace") / "runs" if base == work / "runs" else Path("results")) / f.relative_to(base)
                if base == work / "runs" and not a.all_runs and f.relative_to(base).parts[0] not in keep_runs:
                    continue
                z.write(f, rel.as_posix())
                n += 1
        for f in ("data/splits/split_manifest.json",):
            if (ROOT / f).exists():
                z.write(ROOT / f, f)
    print(f"{n} files -> {a.out} ({Path(a.out).stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
