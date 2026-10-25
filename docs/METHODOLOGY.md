# Methodology

## 1. Problem
Six-class image classification: `freshapples, freshbanana, freshoranges, rottenapples, rottenbanana, rottenoranges`
(label index = position in this list, fixed project-wide). Loss: categorical cross-entropy (implemented as sparse categorical
cross-entropy on integer labels - identical mathematics). Optimiser: Adam. Metrics: accuracy, precision, recall, F1, confusion matrix.

## 2. What the dataset really contains
These are observations made while developing this project; **notebook 01 recomputes all of them for your copy of the data - trust its output**.

| Observation | Consequence |
|---|---|
| About 13,600 files but only about 1,500 *original* photos; most files are pre-augmented copies (rotated, flipped, translated, noisy) whose file names start with `rotated_by_N_`, `vertical_flip_`, `translation_`, `saltandpepper_` | all copies of one photo must stay in one split (`source_id`); report the primary test result on originals |
| Many originals appear in both the Kaggle `train/` and `test/` folders | Kaggle's own folders are **not** a valid test set; byte-identical duplicates are removed (MD5) |
| Originals are screenshots whose names carry the capture time ("Screen Shot 2018-06-08 at 5.18.26 PM.png"); neighbours are seconds apart and often show the same fruit | split by time blocks with a time buffer |

## 3. The split (70 % / 15 % / 15 %, seed 42)
`fruitfresh.splits.make_time_block_split`
1. Per class, sort source photos by capture time.
2. Cut into contiguous blocks of 25 sources.
3. Assign whole blocks to test, then validation, then train, so validation and test each land near 15 % of the class.
4. Drop every source photo within 60 s of a source in another split (training photos are dropped first).
5. `verify_split` asserts: no source or file overlap, all six classes in every split, minimum gap >= 60 s.
6. CSVs + `split_manifest.json` (SHA-256 per file) are saved. Loading a CSV that was edited fails.

The final image percentages are close to, not exactly, 70/15/15 (whole blocks, buffer). The achieved numbers are printed by notebook 01 and stored in the manifest.
Parameters (block 25, buffer 60 s, seed 42) are fixed **before** training and must not be tuned on test results.

Known weakness: each class has only a handful of test blocks, so one unusual capture session can move that class's score; the confidence
intervals and `make_cv_split` (optional group k-fold) address this. A **random** split (`prepare_data.py --method random_group`) is available only for comparison.

## 4. Pre-processing and augmentation
Decode -> RGB -> bilinear resize to 224 x 224 -> round to uint8 -> float32 (0..255). Normalisation lives **inside** each model
(`Rescaling(1/255)` for the CNN; `Rescaling(1/127.5, offset=-1)` for MobileNetV2). Training-time augmentation: horizontal/vertical flip, rotation +-8 % of a turn,
zoom 15 %, translation 8 %, brightness 20 %, contrast 20 %. Validation/test are never augmented.

## 5. Models
Common structure: `image -> [rescale] -> backbone -> GlobalAveragePooling -> Dropout(0.3) -> Dense(6, softmax)`.

* **Baseline CNN** - backbone: 4 blocks (32, 64, 128, 256 filters), each `Conv3x3-BatchNorm-ReLU` twice then `MaxPool`. Trained from scratch (up to 40 epochs, lr 1e-3, patience 8).
* **MobileNetV2** - ImageNet weights, width multiplier 1.0. Stage 1: backbone frozen, head trained (up to 10 epochs, lr 1e-3). Stage 2: top 40 backbone layers unfrozen, lr 2e-5 (up to 20 epochs); BatchNorm layers stay frozen and run in inference mode.

Both: Adam, balanced class weights (computed on the training split), early stopping + ReduceLROnPlateau on validation loss, the best-validation-loss weights are kept.

## 6. Evaluation protocol
* **Primary:** test split, one original image per source photo - independent samples.
* **Secondary:** all test images (originals + the dataset's pre-augmented copies) - a robustness view; per-augmentation accuracies are reported.
* Metrics per class and macro-averaged; confusion matrices (counts and normalised).
* Uncertainty: cluster bootstrap over source photos (2000 resamples) -> 95 % intervals.
* Model comparison: same split, seed, input size, augmentation, class weights, optimiser and loss (`fairness_table` prints the evidence);
  exact McNemar test on the same images; parameters, file size, latency, training time.
* Targets (proposal): accuracy >= 0.90 and macro-F1 > 0.85 for **both** models; `targets_table` prints PASS/FAIL for each model and view.
* **Final model** for the app (rule fixed in advance, uses validation only): highest validation macro-F1; if another model is within 1 point and at least 2x smaller, prefer the smaller one.

## 7. Reading the results critically
* Primary-view test sets contain only a few hundred independent photos: differences of one or two points are within noise - check the intervals and McNemar p-value.
* Accuracy of 99 % or more on the primary view is a **red flag** (leakage or an easy test set), not a success: look at the leakage audit and the error analysis.
* If a target is missed, report it honestly and use the error analysis (confusions, Grad-CAM, per-class F1) to explain why.
* Photos in this dataset are studio-like; real-world performance can be lower - use the real-world photo step in notebook 04.
