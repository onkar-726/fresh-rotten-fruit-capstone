# Fresh-versus-Rotten Fruit Classification

Capstone project - Deep Learning Lab, Department of Electronics Engineering, Ramdeobaba University, Nagpur
(Group 19, academic session 2026-27, semester V).

A fruit photo goes in; the system answers **which of six classes it is** (fresh/rotten x apple/banana/orange),
**whether the fruit is fresh or rotten**, and **how confident it is**. Two models are trained and compared fairly:

| | Model | Idea |
|---|---|---|
| 1 | **Baseline CNN** | four convolution blocks trained from scratch |
| 2 | **MobileNetV2** | ImageNet-pretrained, trained in two stages (new head, then fine-tuning) |

Data: Kaggle *Fruits Fresh and Rotten for Classification* (`sriramr/fruits-fresh-and-rotten-for-classification`).
Targets from the proposal: **accuracy >= 90 %** and **F1-score > 0.85** for both models - *checked by the code against real
results, never assumed* (`scripts/compare.py` prints PASS/FAIL).

> **Honest status.** This repository ships **no trained models and no accuracy numbers**. Training happens on your DGX B200;
> the numbers appear only after you run it. Everything that could be tested without a GPU server and without Kaggle access
> was tested on a synthetic stand-in dataset - see [docs/TESTING_STATUS.md](docs/TESTING_STATUS.md) for exactly what was and was not verified.

## Where each part runs

| Phase | Where | What | Start with |
|---|---|---|---|
| A. Data | **Google Colab** (CPU is enough) | download, explore, validate pre-processing, preview augmentation, 70/15/15 split, verification, transfer bundle | `notebooks/01_colab_data_preparation.ipynb` |
| B. Transfer | your computer | move the bundle to the DGX | `scp` (see below) |
| C. Training | **DGX B200** (all GPUs) | train both models, optional experiments | `bash scripts/run_dgx_pipeline.sh` or `notebooks/02_dgx_training.ipynb` |
| D. Evaluation | DGX | metrics, confusion matrices, comparison, error analysis | `notebooks/03_...`, `notebooks/04_...` (or the scripts) |
| E. Application | laptop or DGX | upload an image -> class, fresh/rotten, confidence | `python app/app.py` |

Everything is saved **locally** (folders inside the project). Nothing uses Google Drive.

## Quick start

```bash
# ---- A. Colab: open notebooks/01_colab_data_preparation.ipynb and run all cells.
#         It downloads the dataset, creates data/splits/*.csv and fruit_data_bundle.zip (downloaded to your computer).

# ---- B. copy the bundle to the DGX and unpack it inside the project folder
scp fruit_data_bundle.zip <user>@<dgx-host>:<path>/fresh-rotten-fruit-capstone/
ssh <user>@<dgx-host>
cd <path>/fresh-rotten-fruit-capstone && unzip -o fruit_data_bundle.zip

# ---- C. on the DGX: environment (see docs/DGX_B200_GUIDE.md), then
python scripts/check_gpu.py                                   # must print GPU CHECK PASSED
python scripts/verify_data.py --split-dir data/splits --data-root data/raw      # must print DATA OK
bash scripts/run_dgx_pipeline.sh                              # trains + evaluates + compares + packages
# (or step by step)
python scripts/train.py --model baseline_cnn  --config configs/default.json
python scripts/train.py --model mobilenet_v2  --config configs/default.json
python scripts/evaluate.py
python scripts/compare.py                                     # PASS/FAIL against the proposal targets
python scripts/error_analysis.py --run baseline_cnn && python scripts/error_analysis.py --run mobilenet_v2
python scripts/package_results.py --out dgx_results.zip

# ---- E. laptop: bring the results back and start the application
scp <user>@<dgx-host>:<path>/fresh-rotten-fruit-capstone/dgx_results.zip .
unzip -o dgx_results.zip -d fresh-rotten-fruit-capstone
cd fresh-rotten-fruit-capstone && pip install -r requirements.txt
python app/app.py                                             # open http://127.0.0.1:7860
```

The full, illustrated walk-through (with expected outputs and troubleshooting) is the PDF manual
**`docs/Fruit_Classification_User_Manual.pdf`**.

## Project layout

```
fresh-rotten-fruit-capstone/
├── README.md  LICENSE  Makefile  pyproject.toml
├── requirements.txt            laptop / general        requirements-train.txt   DGX (TensorFlow excluded)
├── requirements-colab.txt      Colab extras only
├── configs/                    default.json | dgx_b200.json | single_gpu_debug.json   (hardware profiles)
├── notebooks/                  01 Colab data prep | 02 DGX training | 03 evaluation | 04 error analysis + app
├── scripts/                    prepare_data  download_dataset  verify_data  check_gpu  fetch_pretrained_weights
│                               train  run_experiments  summarize_experiments  evaluate  compare  error_analysis
│                               export_tflite  predict  package_results  run_dgx_pipeline.sh
├── slurm/train_dgx.sbatch      optional Slurm job script
├── app/app.py                  Gradio web interface
├── src/fruitfresh/             data_index  splits  datasets  models  distributed  train  evaluate  experiments
│                               compare  error_analysis  gradcam  predict  export  transfer  audit  plots  viz  paths  colab  config
├── tests/                      unit + end-to-end smoke tests (synthetic data)
├── docs/                       METHODOLOGY  DGX_B200_GUIDE  TROUBLESHOOTING  DESIGN_DECISIONS  PROPOSAL_TRACEABILITY  TESTING_STATUS
├── tools/build_manual.py       regenerates the PDF manual and its figures
├── data/splits/                split CSVs + split_manifest.json   (created by notebook 01 / scripts/prepare_data.py)
├── data/raw/                   dataset images on the DGX          (not in git)
├── workspace/                  runs: models, histories, predictions (not in git)
└── results/                    tables, figures, reports, final_model.json (small files; commit these)
```

## Method in one page

* **Split (70/15/15, seed 42, reproducible).** The Kaggle files are mostly *augmented copies* of about 1,500 original
  photos, and the originals are screenshots taken seconds apart. A random file-level split would put copies and near-duplicates
  of the same fruit in both training and test, inflating accuracy. The split is therefore done on **source photos in time blocks
  with a 60 s gap** between splits ([docs/METHODOLOGY.md](docs/METHODOLOGY.md)). Percentages of *images* come out close to, not exactly,
  70/15/15 (whole blocks + the buffer).
* **Pre-processing.** Every image -> RGB -> 224x224 -> float 0..255; each model contains its own normalisation, so the saved
  model, the app and the TFLite export all take the same input.
* **Augmentation (training batches only).** flips, rotation, zoom, translation, brightness, contrast.
* **Training.** Adam, categorical cross-entropy, balanced class weights, early stopping, learning-rate reduction on plateau,
  checkpoint of the best validation loss. MobileNetV2: stage 1 head only, stage 2 top 40 layers (BatchNorm frozen).
  Multi-GPU through `tf.distribute.MirroredStrategy`.
* **Evaluation.** accuracy, precision, recall, F1 (per class and macro), confusion matrices, 95 % bootstrap intervals over source
  photos, exact McNemar test, size/latency, PASS/FAIL against the targets. **Primary test view = one original image per source
  photo**; secondary view = every test image.
* **Application.** Gradio interface; Keras or TFLite model; low-confidence warning; optional Grad-CAM.

## Hardware profiles

| Profile | Use | Settings |
|---|---|---|
| `configs/default.json` | first run on the DGX | all GPUs, 32 images/GPU, full precision |
| `configs/dgx_b200.json` | performance run after the default works | all GPUs, 64 images/GPU, bfloat16 |
| `configs/single_gpu_debug.json` | isolate multi-GPU problems | 1 GPU |

Override anything on the command line: `--gpus 2 --per-gpu-batch-size 16 --mixed-precision off`.

## Tests

```bash
python -m pytest tests -q          # whole suite on CPU with synthetic data (about 10 minutes)
make test-fast                     # unit tests only
```

## Documentation

* `docs/Fruit_Classification_User_Manual.pdf` - step-by-step manual (which file, where to run, commands, outputs, troubleshooting, checklist)
* [docs/DGX_B200_GUIDE.md](docs/DGX_B200_GUIDE.md) - DGX set-up, assumptions, multi-GPU notes
* [docs/METHODOLOGY.md](docs/METHODOLOGY.md) - dataset facts, split, training, evaluation protocol
* [docs/DESIGN_DECISIONS.md](docs/DESIGN_DECISIONS.md) - every decision made where the proposal was silent
* [docs/PROPOSAL_TRACEABILITY.md](docs/PROPOSAL_TRACEABILITY.md) - proposal item -> file
* [docs/TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md), [docs/TESTING_STATUS.md](docs/TESTING_STATUS.md)

## Licence

MIT - see [LICENSE](LICENSE). The dataset keeps its own Kaggle licence.
