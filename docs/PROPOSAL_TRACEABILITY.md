# Proposal traceability

| Proposal item | Implemented in |
|---|---|
| Objective 1 - baseline CNN, six classes | `src/fruitfresh/models.py` (`baseline_cnn`), `scripts/train.py --model baseline_cnn` |
| Objective 2 - MobileNet with transfer learning | `models.py` (`mobilenet_v2`), `train.py` (2 stages), `scripts/train.py --model mobilenet_v2` |
| Objective 3 - accuracy, precision, recall, F1 on the test set | `evaluate.py`, `scripts/evaluate.py` -> `workspace/runs/<run>/seed42/eval/` |
| Objective 4 - compare CNN and MobileNet, decide which makes sense on limited hardware | `compare.py`, `scripts/compare.py` (accuracy + CI + McNemar + size + latency), `export.py` (TFLite) |
| Dataset - Kaggle Fruits Fresh and Rotten, 6 classes | `data_index.py`, notebook 01, `scripts/download_dataset.py` |
| "Resize, normalise, split into train/validation/test" | `datasets.py` (resize), models (normalise), `splits.py` (70/15/15) |
| "Some classes have more images than others" | class balance plot (notebook 01), class weights (`datasets.py`) |
| "Lighting/backgrounds not consistent; half-rotten fruit" | augmentation (brightness/contrast), error analysis, real-world photo test (notebook 04) |
| Pipeline: input -> preprocessing -> augmentation -> CNN/MobileNet -> classification -> fresh/rotten prediction -> evaluation | architecture diagram (PDF manual, figure 1); `predict.py` for the prediction step |
| Data split 70 / 15 / 15 | `splits.py`, `config.SPLIT_PARAMS` |
| Loss: categorical cross-entropy; optimiser: Adam | `train.py` (`_compile`) |
| Evaluation metrics: accuracy, precision, recall, F1, confusion matrix | `evaluate.py`, notebook 03 |
| Baseline / comparison: CNN vs MobileNet | `compare.py` |
| Deliverable: working CNN and MobileNet, trained and tested, with comparison | notebooks 02-03, scripts, `results/comparison_report.md` (generated) |
| Deliverable: prototype - photo in, fresh/rotten + confidence out | `app/app.py`, `predict.py`, `scripts/predict.py` |
| Deliverable: confusion matrix for error analysis, common failure cases | `error_analysis.py`, `scripts/error_analysis.py`, notebook 04 |
| Extra (not in the proposal) | Grad-CAM, bootstrap intervals, TFLite export, multi-GPU training, experiments, data-transfer bundle |
