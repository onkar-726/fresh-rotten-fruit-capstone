"""Run the full test-suite AND execute the four notebooks on synthetic data, then write docs/test_summary.json
(read by tools/build_manual.py so the manual states real, machine-recorded numbers).

    python tools/run_tests.py                  # pytest + notebooks   (about 10-15 minutes on a CPU)
    python tools/run_tests.py --skip-notebooks
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = ["01_colab_data_preparation", "02_dgx_training", "03_evaluation_comparison", "04_error_analysis_and_app"]


def run_pytest(tmp: Path) -> dict:
    junit = tmp / "junit.xml"
    r = subprocess.run([sys.executable, "-m", "pytest", "tests", "-q", f"--junitxml={junit}"], cwd=ROOT)
    s = ET.parse(junit).getroot()
    s = s if s.tag == "testsuite" else s.find("testsuite")
    tests, fail, err, skip = (int(s.get(k)) for k in ("tests", "failures", "errors", "skipped"))
    return {"passed": tests - fail - err - skip, "failed": fail + err, "skipped": skip, "tests": tests,
            "pytest_seconds": round(float(s.get("time")))}


def run_notebooks(tmp: Path) -> dict:
    sys.path.insert(0, str(ROOT / "tests"))
    from fake_dataset import make_fake_dataset

    data = make_fake_dataset(tmp / "fake", sources_per_class=60)
    proj = tmp / "fresh-rotten-fruit-capstone"
    shutil.copytree(ROOT, proj, ignore=shutil.ignore_patterns("__pycache__", ".pytest_cache", "workspace", "*.zip", "*.pdf"))
    (proj / "workspace").mkdir(exist_ok=True)
    env = {**os.environ, "FRUIT_SMOKE": "1", "FRUIT_NO_PRETRAINED": "1", "FRUIT_SKIP_PIP": "1", "FRUIT_DATA_ROOT": str(data),
           "FRUIT_IMAGE_SIZE": "64", "FRUIT_BLOCK_SIZE": "8", "FRUIT_BUFFER_SECONDS": "20", "MPLBACKEND": "Agg",
           "TF_CPP_MIN_LOG_LEVEL": "3"}
    for k in ("FRUIT_WORK_DIR", "FRUIT_RESULTS_DIR", "FRUIT_SPLIT_DIR"):
        env.pop(k, None)
    ok = 0
    for nb in NOTEBOOKS:
        r = subprocess.run([sys.executable, "-m", "jupyter", "nbconvert", "--to", "notebook", "--execute",
                            "--ExecutePreprocessor.timeout=1500", "--output", f"{nb}.executed.ipynb", f"{nb}.ipynb"],
                           cwd=proj / "notebooks", env=env, capture_output=True, text=True)
        ok += r.returncode == 0
        print(f"notebook {nb}: {'OK' if r.returncode == 0 else 'FAILED'}")
        if r.returncode:
            print(r.stderr[-1500:])
    return {"notebooks_executed": ok, "notebooks_total": len(NOTEBOOKS)}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--skip-notebooks", action="store_true")
    a = ap.parse_args()
    import tensorflow as tf

    with tempfile.TemporaryDirectory() as t:
        tmp = Path(t)
        summary = run_pytest(tmp)
        if not a.skip_notebooks:
            summary.update(run_notebooks(tmp))
    summary.update(tensorflow=tf.__version__, python=sys.version.split()[0], date=str(date.today()),
                   environment="CPU only, synthetic stand-in dataset")
    (ROOT / "docs" / "test_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    sys.exit(1 if summary["failed"] or summary.get("notebooks_executed", 4) != summary.get("notebooks_total", 4) else 0)


if __name__ == "__main__":
    main()
