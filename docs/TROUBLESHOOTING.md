# Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| Notebook says "Project folder not found" (Colab) | project not uploaded | Upload `fresh-rotten-fruit-capstone.zip` to `/content` (Files panel) **or** set `REPO_URL`, re-run the setup cell |
| `kagglehub` download fails / times out | no internet, Kaggle rate limit | retry; on a machine with the data set `FRUIT_DATA_ROOT`; or download the dataset in the browser and unzip it |
| `ModuleNotFoundError: fruitfresh` | scripts run from the wrong folder, or notebook setup cell skipped | run commands from the project root; run the setup cell first |
| `verify_data.py` -> `DATA PROBLEM: ... missing` | images not unpacked / wrong `--data-root` | unzip the bundle **inside the project folder**; check that `data/raw/train/<class>/` exists, or pass the right `--data-root` |
| `AssertionError: test.csv changed since the split was created` | a split CSV was edited or re-saved | re-copy the original CSVs and `split_manifest.json` from the bundle; never open/re-save them in a spreadsheet |
| `check_gpu.py`: "No GPU visible" | wrong environment/container, driver or CUDA mismatch, `CUDA_VISIBLE_DEVICES` empty | run `nvidia-smi`; use a Blackwell-capable TensorFlow build or NGC container (docs/DGX_B200_GUIDE.md) |
| "no kernel image is available" / PTX errors | the TensorFlow build has no kernels for the B200 | use a TensorFlow/NGC release built for Blackwell (CUDA 12.8+) |
| Out-of-memory | batch too large or other users on the GPU | `--per-gpu-batch-size 16`; check `nvidia-smi` for other jobs |
| NCCL / hang at the start of multi-GPU training | communication problem | try `--config configs/single_gpu_debug.json`; set `NCCL_DEBUG=INFO` to see details; ask the admins |
| Training is slow, GPUs idle | input pipeline / CPU bound / data on slow disk | copy the data to local SSD; make sure enough CPU cores are allocated; first epoch is slower (images are cached afterwards) |
| MobileNetV2 stage 1 fails downloading weights | DGX offline | `python scripts/fetch_pretrained_weights.py` elsewhere, copy `assets/*.h5`, use `--weights-file` |
| Accuracy about 17 % and not improving | labels/paths mismatched, learning rate too high | check `train_info.json` (class weights, sizes); re-run with `--force`; look at the history curves |
| Validation accuracy >> training accuracy | dropout + augmentation make the training metric pessimistic | normal early on; judge by validation/test |
| Test accuracy 99 %+ | possible leakage / easy test set | read the leakage audit; check `verify_split` output; do not report it as a success without analysis |
| One class has very low recall | confusion with a similar class (e.g. rotten vs fresh orange), few test blocks | `error_analysis.py`: top confusions, Grad-CAM; consider more fine-tuning depth |
| `compare.py`: `FileNotFoundError ... predictions` | `evaluate_run` has not run | run `python scripts/evaluate.py` (compare.py also evaluates automatically if the model exists) |
| `compare.py --use-best` fails | missing `results/best_*.json` | run `summarize_experiments.py` for **both** families first |
| App: "Model file not found" / `final_model.json not found` | compare.py not run or results not copied | run `python scripts/compare.py`, or start the app with `python app/app.py --model <path to .keras>` |
| App: port already in use | another Gradio instance | `python app/app.py --port 7861` |
| Cannot open `.keras` file ("Keras 3 required" / version error) | the model was saved with a newer Keras than the one loading it | use the same major Keras/TensorFlow version on the laptop as on the DGX (`pip install "tensorflow>=2.16"`) |
| TFLite predictions differ slightly | dynamic quantisation | expected small difference; `export_tflite.py` prints agreement and max difference |
| Colab runtime reset, files gone | Colab disk is temporary | re-run the setup cell; the data bundle was downloaded to your computer in notebook 01, step 7 - only re-run notebook 01 if you lost it |
