# Design decisions (where the proposal was silent or needed a choice)

| # | Topic | Decision | Reason |
|---|---|---|---|
| 1 | Split | 70/15/15 by **source photo in time blocks** with a 60 s buffer, seed 42 | Dataset = mostly augmented copies of ~1,500 photos taken seconds apart; a file-level random split leaks |
| 2 | Primary test view | one original image per source photo | independent samples; no augmented copies |
| 3 | Targets | checked by code (PASS/FAIL), shown for both views | the proposal says "subject to actual results"; nothing is assumed |
| 4 | Loss | sparse categorical cross-entropy | same as categorical cross-entropy with integer labels |
| 5 | Normalisation inside the model | `Rescaling` layer is part of each model | saved model, app and TFLite share one input format (float 0..255) |
| 6 | Class imbalance | balanced class weights from the training split | proposal notes some classes are larger |
| 7 | Augmentation | on-the-fly, training only | dataset already contains copies; random variants add diversity, never touch val/test |
| 8 | MobileNetV2 training | 2 stages, BatchNorm frozen, small fine-tune lr | standard, stable transfer-learning recipe |
| 9 | Multi-GPU | `MirroredStrategy`, global batch = per-GPU batch x GPUs, class weights as per-sample weights | simplest robust synchronous scheme; works with `tf.data` |
| 10 | Mixed precision | off by default, bfloat16 as a profile | B200 supports bf16, but it was not benchmarked here - opt-in |
| 11 | Learning-rate scaling | none by default (`--lr-scaling sqrt/linear` available) | tiny dataset; large batches already reduce steps per epoch |
| 12 | Storage | everything local (`workspace/`, `results/`, `data/`), no Google Drive | requested; Colab outputs are downloaded to the user's computer |
| 13 | Data transfer | zip bundle with SHA-256-verified CSVs; images optional | DGX may be offline; `verify_data.py` proves the transfer was lossless |
| 14 | Offline weights | `fetch_pretrained_weights.py` + `--weights-file` | DGX may be offline |
| 15 | Experiments | one change per run, validation only, fixed selection rule | avoids tuning on the test set |
| 16 | Statistics | bootstrap intervals over source photos + exact McNemar | small test sets; honest uncertainty |
| 17 | App framework | Gradio (Keras or TFLite model) | upload -> result with minimal code; works on laptop, DGX (SSH tunnel) and Colab |
| 18 | Low-confidence warning | below 60 % the app warns | the model only knows six classes; a heuristic, not a guarantee |
| 19 | Seeds | 42 everywhere | reproducibility (exact bit-for-bit equality across GPU types is not guaranteed) |
| 20 | Result files | tables/figures generated from saved metrics only | no hand-typed numbers anywhere |
