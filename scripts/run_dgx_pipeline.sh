#!/usr/bin/env bash
# Run the whole DGX part: check -> verify data -> train both models -> evaluate -> compare -> error analysis -> package.
#   bash scripts/run_dgx_pipeline.sh
# Optional environment variables:
#   PYTHON=python3  PROFILE=configs/default.json  DATA_ROOT=data/raw  EXPERIMENTS=1 (also run experiments)
#   OUT_ZIP=dgx_results.zip   ALLOW_CPU=1 (testing without GPUs)   EXTRA_ARGS="..." (extra flags for the train scripts)
set -euo pipefail
cd "$(dirname "$0")/.."
PYTHON="${PYTHON:-python3}"
PROFILE="${PROFILE:-configs/default.json}"
DATA_ROOT="${DATA_ROOT:-data/raw}"
CPU_ARG=""; if [[ "${ALLOW_CPU:-0}" == "1" ]]; then CPU_ARG="--allow-cpu"; fi
EXTRA_ARGS="${EXTRA_ARGS:-}"
OUT_ZIP="${OUT_ZIP:-dgx_results.zip}"
WEIGHTS_ARG=""
if ls assets/mobilenet_v2_weights*.h5 >/dev/null 2>&1; then WEIGHTS_ARG="--weights-file $(ls assets/mobilenet_v2_weights*.h5 | head -1)"; fi

echo "== 1/8 GPU check";        $PYTHON scripts/check_gpu.py $CPU_ARG
echo "== 2/8 Verify data";      $PYTHON scripts/verify_data.py --data-root "$DATA_ROOT"
echo "== 3/8 Train Model 1 (baseline CNN)"
$PYTHON scripts/train.py --model baseline_cnn  --config "$PROFILE" --data-root "$DATA_ROOT" $EXTRA_ARGS
echo "== 4/8 Train Model 2 (MobileNetV2)"
$PYTHON scripts/train.py --model mobilenet_v2 --config "$PROFILE" --data-root "$DATA_ROOT" $WEIGHTS_ARG $EXTRA_ARGS
if [[ "${EXPERIMENTS:-0}" == "1" ]]; then
  echo "== 4b experiments"
  $PYTHON scripts/run_experiments.py --family baseline_cnn --names all --config "$PROFILE" --data-root "$DATA_ROOT" $EXTRA_ARGS
  $PYTHON scripts/run_experiments.py --family mobilenet_v2 --names all --config "$PROFILE" --data-root "$DATA_ROOT" $WEIGHTS_ARG $EXTRA_ARGS
  $PYTHON scripts/summarize_experiments.py --family baseline_cnn  --data-root "$DATA_ROOT"
  $PYTHON scripts/summarize_experiments.py --family mobilenet_v2 --data-root "$DATA_ROOT"
fi
echo "== 5/8 Evaluate";         $PYTHON scripts/evaluate.py --data-root "$DATA_ROOT"
echo "== 6/8 Compare";          $PYTHON scripts/compare.py --data-root "$DATA_ROOT"
echo "== 7/8 Error analysis"
for run in baseline_cnn mobilenet_v2; do $PYTHON scripts/error_analysis.py --run "$run" --data-root "$DATA_ROOT"; done
echo "== 8/8 Package results";  $PYTHON scripts/package_results.py --out "$OUT_ZIP"
echo "DONE. Copy $OUT_ZIP to your laptop (see docs/DGX_B200_GUIDE.md)."
