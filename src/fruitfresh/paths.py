"""Find files the scripts, notebooks and app need."""
from __future__ import annotations

import json
from pathlib import Path

from .config import DEFAULT_RESULTS_DIR, DEFAULT_WORK_DIR, SEED


def run_model_path(run_name: str, work_dir=None, seed: int = SEED) -> Path:
    return Path(work_dir or DEFAULT_WORK_DIR) / "runs" / run_name / f"seed{seed}" / "model.keras"


def resolve_model(spec: str = "auto", work_dir=None, results_dir=None, seed: int = SEED) -> Path:
    """'auto' -> the final model chosen by compare.py; a run name -> its model.keras; otherwise a file path."""
    if spec == "auto":
        info = Path(results_dir or DEFAULT_RESULTS_DIR) / "final_model.json"
        if not info.exists():
            raise FileNotFoundError(
                f"{info} not found. Run scripts/compare.py first, or pass --model <path to .keras/.tflite>.")
        spec = json.loads(info.read_text())["final_model"]
    p = Path(spec)
    if p.suffix in (".keras", ".tflite", ".h5"):
        return p
    candidate = run_model_path(spec, work_dir, seed)
    if candidate.exists():
        return candidate
    raise FileNotFoundError(f"No model found for '{spec}' (looked for {candidate}).")
