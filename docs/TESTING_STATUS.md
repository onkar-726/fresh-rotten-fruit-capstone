# Testing status - what was verified, and what only your servers can verify

Test environment used to build this project: Linux, **CPU only**, Python 3.12, TensorFlow 2.21 / Keras 3, a **synthetic stand-in dataset**
(same folder layout and file naming as the Kaggle data, tiny coloured images). No GPU, no Kaggle access, no Colab, no DGX.

## Verified locally (automated, `python -m pytest tests -q`)

| Area | Tests |
|---|---|
| File-name parsing (augmentation prefixes, capture time, noon/midnight, chained prefixes, U+202F) | `test_parsing.py` |
| Split: leak-free, reproducible, all copies together, time gap, manifest hashes, random-group + group-k-fold helpers | `test_splits.py` |
| Local workspace, backup/restore, data-root resolution, duplicate removal, transfer bundle round-trip, missing-image detection | `test_workspace_and_transfer.py` |
| tf.data pipeline order/shape/weights, class weights, both models build / freeze correctly / train one step / Grad-CAM | `test_models_and_data.py` |
| Real Gradio server, image upload through the API, result text, Grad-CAM option | `test_app_ui.py` |
| Every script starts and prints `--help` | `test_scripts_help.py` |
| End to end: prepare_data (+audit +bundle), verify_data, train both models on **2 logical CPU replicas**, resume, experiments, summarize, evaluate, compare, error analysis, predict, TFLite export (Keras-vs-TFLite agreement), package_results layout, `compare --use-best`, the one-command DGX shell pipeline | `test_pipeline_smoke.py` |
| Docs mention only scripts / notebooks / configs / experiment names that exist | `test_docs_consistency.py` |
| The four notebooks execute top to bottom (nbconvert, synthetic data, 1 epoch) | run manually before delivery |

The exact pass/fail counts of the last full run are in `docs/test_summary.json` (and printed in the PDF manual).

## NOT verified here - requires your environment

| Item | Why it could not be tested | How you verify it |
|---|---|---|
| Kaggle download, real dataset statistics, real split sizes | no Kaggle access | notebook 01 prints everything |
| ImageNet weights download for MobileNetV2 | no internet access to the weight host | first MobileNetV2 training; or `fetch_pretrained_weights.py` |
| Colab behaviour (zip unpacking, browser downloads, public app link) | not run in Colab | run notebook 01 in Colab |
| **Real GPUs, NCCL all-reduce, bfloat16 on B200, TensorFlow/CUDA compatibility with Blackwell, throughput** | no GPU | `python scripts/check_gpu.py`, then a run with `configs/default.json` |
| **Accuracy, F1, which model is better, whether the 90 % / 0.85 targets are met** | needs training on the real data | `scripts/evaluate.py`, `scripts/compare.py` (PASS/FAIL is computed, never assumed) |

No accuracy, F1 score, training curve or DGX execution result appears anywhere in this delivery.
