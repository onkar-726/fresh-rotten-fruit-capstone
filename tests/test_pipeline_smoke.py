
"""End-to-end smoke test on a synthetic dataset: split -> train both models -> evaluate -> compare ->
error analysis -> predict -> TFLite -> app logic. (1 epoch each, 64 px, no ImageNet weights.)
"""
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
pytest.importorskip("tensorflow")
from fake_dataset import make_fake_dataset  # noqa: E402


@pytest.fixture(scope="module")
def env(tmp_path_factory):
    base = tmp_path_factory.mktemp("e2e")
    data = make_fake_dataset(base / "data", sources_per_class=60)

    e = {
        **os.environ,
        "FRUIT_SMOKE": "1",
        "FRUIT_IMAGE_SIZE": "64",
        "FRUIT_BLOCK_SIZE": "8",
        "FRUIT_BUFFER_SECONDS": "20",
        "FRUIT_DATA_ROOT": str(data),
        "FRUIT_WORK_DIR": str(base / "work"),
        "FRUIT_RESULTS_DIR": str(base / "results"),
        "FRUIT_SPLIT_DIR": str(base / "splits"),
        "TF_CPP_MIN_LOG_LEVEL": "3",
        "TF_NUM_INTRAOP_THREADS": "2",
        "TF_NUM_INTEROP_THREADS": "2",
    }

    return {
        "env": e,
        "base": base,
        "split": base / "splits",
        "results": base / "results",
        "work": base / "work",
    }


def run(env, *args, **kw):
    r = subprocess.run(
        [sys.executable, *map(str, args)],
        cwd=ROOT,
        env=env["env"],
        capture_output=True,
        text=True,
        timeout=900,
        **kw,
    )
    assert r.returncode == 0, (
        f"{args}\nSTDOUT:\n{r.stdout[-2500:]}\nSTDERR:\n{r.stderr[-2500:]}"
    )
    return r.stdout


def ensure_splits(env):
    """Prepare synthetic splits if this test is run independently."""
    if not (env["split"] / "train.csv").exists():
        run(
            env,
            "scripts/prepare_data.py",
            "--out",
            env["split"],
            "--audit",
            "--bundle",
            env["base"] / "b.zip",
            "--bundle-no-images",
        )


def test_01_prepare_data(env):
    out = run(
        env,
        "scripts/prepare_data.py",
        "--out",
        env["split"],
        "--audit",
        "--bundle",
        env["base"] / "b.zip",
        "--bundle-no-images",
    )
    assert (
        "split saved" in out
        and "leakage audit" in out
        and (env["base"] / "b.zip").exists()
    )
    assert "DATA OK" in run(
        env, "scripts/verify_data.py", "--split-dir", env["split"]
    )


def test_02_train_both_models_two_replicas(env):
    """Exercise the two-logical-CPU-replica path without exposing physical GPUs."""
    ensure_splits(env)

    env["env"]["FRUIT_FAKE_REPLICAS"] = "2"
    env["env"]["CUDA_VISIBLE_DEVICES"] = ""

    common = [
        "--split-dir",
        env["split"],
        "--no-pretrained",
        "--smoke",
        "--per-gpu-batch-size",
        "4",
    ]

    run(env, "scripts/train.py", "--model", "baseline_cnn", *common)
    run(env, "scripts/train.py", "--model", "mobilenet_v2", *common)

    for model_name in ("baseline_cnn", "mobilenet_v2"):
        run_dir = env["work"] / "runs" / model_name / "seed42"
        info = json.loads((run_dir / "train_info.json").read_text())

        assert info["hardware"]["num_replicas"] == 2
        assert info["hardware"]["strategy"] == "MirroredStrategy"
        assert info["hardware"]["device_type"] == "CPU"
        assert info["hardware"]["global_batch_size"] == 8
        assert (run_dir / "model.keras").exists()
        assert (run_dir / "history.csv").exists()

    info = json.loads(
        (
            env["work"]
            / "runs"
            / "mobilenet_v2"
            / "seed42"
            / "train_info.json"
        ).read_text()
    )
    assert info["stages"]["finetune"]["backbone_batchnorm_trainable"] == 0


def test_03_resume_skips_finished_runs(env):
    ensure_splits(env)
    out = run(
        env,
        "scripts/train.py",
        "--model",
        "baseline_cnn",
        "--split-dir",
        env["split"],
        "--smoke",
    )
    assert "already trained" in out


def test_04_experiment_and_summary(env):
    ensure_splits(env)
    run(
        env,
        "scripts/run_experiments.py",
        "--family",
        "mobilenet_v2",
        "--names",
        "head_only",
        "ft_top20",
        "--split-dir",
        env["split"],
        "--no-pretrained",
        "--smoke",
    )
    out = run(
        env,
        "scripts/summarize_experiments.py",
        "--family",
        "mobilenet_v2",
        "--split-dir",
        env["split"],
        "--results-dir",
        env["results"],
    )
    assert "best run on VALIDATION" in out
    assert (env["results"] / "best_mobilenet_v2.json").exists()


def test_05_evaluate_compare_error_analysis(env):
    ensure_splits(env)
    args = ["--split-dir", env["split"], "--results-dir", env["results"]]

    out = run(env, "scripts/evaluate.py", "--n-boot", "100", *args)
    assert "test_original" in out and ("PASS" in out or "FAIL" in out)

    out = run(env, "scripts/compare.py", "--n-boot", "100", *args)
    assert "final model for the app" in out

    for filename in (
        "comparison_report.md",
        "final_model.json",
        "targets_check.csv",
        "efficiency.csv",
        "training_curves.png",
        "metric_bars_test_original.png",
    ):
        assert (env["results"] / filename).exists(), filename

    report = (env["results"] / "comparison_report.md").read_text()
    assert "McNemar" in report
    assert "Fairness" in report

    run(env, "scripts/error_analysis.py", "--run", "mobilenet_v2", *args)
    assert (
        env["results"]
        / "error_analysis"
        / "mobilenet_v2"
        / "error_report_test_original.md"
    ).exists()


def test_06_predict_export_and_app(env):
    from PIL import Image

    ensure_splits(env)

    img = next(
        (env["base"] / "data" / "dataset" / "train" / "freshapples").glob(
            "Screen*.png"
        )
    )

    out = run(
        env,
        "scripts/predict.py",
        "--model",
        "auto",
        "--image",
        img,
        "--results-dir",
        env["results"],
    )
    assert "confidence" in out

    out = run(
        env,
        "scripts/export_tflite.py",
        "--run",
        "auto",
        "--split-dir",
        env["split"],
        "--results-dir",
        env["results"],
    )
    report = json.loads(out[out.index("{"): out.rindex("}") + 1])
    assert report["top1_agreement"] >= 0.9
    assert report["tflite_mb"] < report["keras_mb"]

    sys.path.insert(0, str(ROOT / "app"))
    from app import format_result  # noqa: E402
    from fruitfresh.predict import FruitPredictor  # noqa: E402

    final = json.loads(
        (env["results"] / "final_model.json").read_text()
    )["final_model"]
    model = env["work"] / "runs" / final / "seed42" / "model.keras"
    tflite = next((env["results"] / "export").glob("*.tflite"))

    for model_path in (model, tflite):
        result = FruitPredictor(model_path).predict(Image.open(img))
        assert abs(sum(result["probabilities"].values()) - 1) < 1e-4
        assert result["status"] in ("fresh", "rotten")
        assert "Confidence" in format_result(result)


def test_07_baseline_experiment_and_compare_use_best(env):
    ensure_splits(env)

    run(
        env,
        "scripts/run_experiments.py",
        "--family",
        "baseline_cnn",
        "--names",
        "no_augmentation",
        "--split-dir",
        env["split"],
        "--smoke",
    )
    args = ["--split-dir", env["split"], "--results-dir", env["results"]]
    run(
        env,
        "scripts/summarize_experiments.py",
        "--family",
        "baseline_cnn",
        *args,
    )
    out = run(
        env,
        "scripts/compare.py",
        "--use-best",
        "--n-boot",
        "50",
        *args,
    )
    assert "final model for the app" in out


def test_08_package_results_has_stable_layout(env):
    import zipfile

    archive = env["base"] / "pack.zip"
    run(
        env,
        "scripts/package_results.py",
        "--out",
        archive,
        "--work-dir",
        env["work"],
        "--results-dir",
        env["results"],
    )

    with zipfile.ZipFile(archive) as z:
        names = z.namelist()

    assert "workspace/runs/baseline_cnn/seed42/model.keras" in names
    assert "workspace/runs/mobilenet_v2/seed42/eval/metrics.json" in names
    assert "results/comparison_report.md" in names
    assert "results/final_model.json" in names
    assert not any("checkpoint_" in name for name in names)


def test_09_dgx_shell_pipeline_end_to_end(env):
    """Check pipeline wiring; training artifacts should already exist."""
    out_zip = env["base"] / "pipeline.zip"
    pipeline_env = {
        **env["env"],
        "ALLOW_CPU": "1",
        "DATA_ROOT": env["env"]["FRUIT_DATA_ROOT"],
        "EXTRA_ARGS": "--no-pretrained --smoke",
        "OUT_ZIP": str(out_zip),
        "PYTHON": sys.executable,
    }

    r = subprocess.run(
        ["bash", "scripts/run_dgx_pipeline.sh"],
        cwd=ROOT,
        env=pipeline_env,
        capture_output=True,
        text=True,
        timeout=900,
    )
    assert r.returncode == 0, (
        f"STDOUT:\n{r.stdout[-2500:]}\nSTDERR:\n{r.stderr[-2500:]}"
    )

    for step in (
        "1/8 GPU check",
        "2/8 Verify data",
        "6/8 Compare",
        "8/8 Package results",
        "DONE",
    ):
        assert step in r.stdout

    assert out_zip.exists()
