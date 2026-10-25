# DGX B200 guide

## Assumptions (please check them - they are not hard-coded anywhere)

The project description says the GPU specifications and the Colab/DGX workflow were already given. Those details were **not available**
when this project was generated, so the following *assumptions* were made. Each one is configurable; none is baked into the code.

| Assumption | If it is wrong |
|---|---|
| A DGX B200 node has **8 NVIDIA B200 (Blackwell) GPUs**; you have shell/SSH or JupyterLab access | use `--gpus N` or `CUDA_VISIBLE_DEVICES` to match what you are allowed to use |
| You may run Python code directly, or inside a container, or through Slurm | pick the matching route in section 2 / 4 |
| The DGX **may or may not** have internet access | use the offline routes (data bundle with images, `fetch_pretrained_weights.py`) |
| Several users may share the machine | start with `--gpus 1`, ask for more when needed |
| TensorFlow for Blackwell needs CUDA 12.8 or newer | **verify with the release notes of your TensorFlow build / NGC container** - the authors could not test on a B200 |

## 1. What the code does with several GPUs

`scripts/train.py` calls `configure_hardware()` which creates a `tf.distribute.MirroredStrategy` over all visible GPUs
(`--gpus all`, `--gpus 4`, or `--gpus 0,1`). Each GPU gets `per_gpu_batch_size` images, so the **global batch = per-GPU batch x number of GPUs**
(default 32 x 8 = 256). Weights are kept identical on all GPUs; gradients are averaged (NCCL all-reduce).

Notes
* The training set is only a few thousand images. One B200 is already more than enough; eight GPUs mainly shorten each epoch but also
  give very few optimisation steps per epoch with a large global batch. If accuracy is lower with 8 GPUs than with 1, try
  `--per-gpu-batch-size 8` or `--lr-scaling sqrt`, or train on `--gpus 1`.
* `--mixed-precision bf16` (profile `configs/dgx_b200.json`) uses bfloat16 compute; the final softmax stays float32. Try it only after a full-precision run works.
* The multi-GPU code path is covered by an automated test that uses **two logical CPU replicas** (no GPU needed). Real NCCL/B200 behaviour
  was **not** tested here.

## 2. Create the Python environment (choose ONE route)

**Route A - NVIDIA NGC TensorFlow container (recommended for Blackwell GPUs).** Pick a release of the NGC TensorFlow container that lists B200/Blackwell support in its release notes.
```bash
# with Docker + NVIDIA Container Toolkit (your site may use Singularity/Enroot/Pyxis instead)
docker run --gpus all -it --rm --shm-size=16g -v "$PWD":/workspace -w /workspace <ngc-tensorflow-image> bash
pip install -r requirements-train.txt          # TensorFlow is already inside the container
python scripts/check_gpu.py
```
**Route B - virtual environment with a TensorFlow build that supports your CUDA/driver.**
```bash
python3 -m venv .venv && source .venv/bin/activate
pip install tensorflow            # only if its CUDA build supports Blackwell; otherwise use Route A
pip install -r requirements-train.txt
python scripts/check_gpu.py
```
**Route C - conda** (same idea as B). In every route, `python scripts/check_gpu.py` is the acceptance test.

## 3. Put the data on the DGX

| Situation | Do this |
|---|---|
| DGX has **no internet** | In Colab keep `INCLUDE_IMAGES = True` in notebook 01; `scp fruit_data_bundle.zip` to the DGX; `unzip -o fruit_data_bundle.zip` in the project folder. |
| DGX **has internet** | Copy only the CSVs (set `INCLUDE_IMAGES = False`, or `scripts/prepare_data.py --bundle x.zip --bundle-no-images`), unzip, then `python scripts/download_dataset.py --link data/raw` (needs internet + `kagglehub`). |
| You already have the dataset somewhere on the DGX | `export FRUIT_DATA_ROOT=/path/with/train-and-test-folders` (or pass `--data-root`). |

Then **always** run `python scripts/verify_data.py --split-dir data/splits --data-root data/raw` - it must print `DATA OK`.
It checks the SHA-256 hash of every split CSV and that every listed image exists.

**MobileNetV2 ImageNet weights on an offline DGX.** On a machine with internet: `python scripts/fetch_pretrained_weights.py` (creates `assets/mobilenet_v2_weights_*.h5`);
copy that file to the DGX `assets/` folder. `scripts/run_dgx_pipeline.sh` and notebook 02 pick it up automatically; manually: `--weights-file assets/<file>.h5`.

## 4. Train

Direct (login shell or interactive session):
```bash
python scripts/train.py --model baseline_cnn --config configs/default.json
python scripts/train.py --model mobilenet_v2 --config configs/default.json     # add --weights-file assets/... if offline
```
Everything in one go: `bash scripts/run_dgx_pipeline.sh`. Optional environment variables: `PROFILE=configs/dgx_b200.json` (hardware profile), `DATA_ROOT=data/raw`, `EXPERIMENTS=1` (also run the experiments), `OUT_ZIP=dgx_results.zip`, `PYTHON=python3`, `EXTRA_ARGS="..."` (extra flags passed to the training scripts) and `ALLOW_CPU=1` (only for testing without GPUs).

Slurm sites: edit the `#SBATCH` lines in `slurm/train_dgx.sbatch` for your partition/time/GPU rules, then `sbatch slurm/train_dgx.sbatch`; follow with `tail -f logs/fruit-train-<jobid>.out`.

Long sessions: use `tmux`/`screen` so a dropped SSH connection does not stop training. A finished run is never retrained (`--force` overrides),
and a MobileNetV2 run that stopped after stage 1 resumes from `stage1.keras`.

Monitor GPUs: `watch -n 2 nvidia-smi`. Expected: memory used on every selected GPU, utilisation well above zero.

## 5. Experiments (optional, "experimentation on the DGX")

One change per run, judged on **validation only**:
```bash
python scripts/run_experiments.py --family mobilenet_v2 --names head_only ft_top20 ft_top80 ft_lr_1e-5 --config configs/default.json
python scripts/summarize_experiments.py --family mobilenet_v2        # table + best run on validation
python scripts/summarize_experiments.py --family baseline_cnn        # (needed for --use-best; run experiments for it first)
python scripts/compare.py --use-best                                  # compare the best run of each model
```
Available names are listed in `src/fruitfresh/config.py` (`EXPERIMENTS`).

## 6. Evaluate, compare, analyse
```bash
python scripts/evaluate.py && python scripts/compare.py
python scripts/error_analysis.py --run baseline_cnn
python scripts/error_analysis.py --run mobilenet_v2
```
`compare.py` writes `results/comparison_report.md` (tables only from saved numbers), CSV tables, figures and `results/final_model.json`.

## 7. Bring results to your laptop
```bash
python scripts/package_results.py --out dgx_results.zip      # final models + results (add --all-runs for experiment models too)
# on the laptop
scp <user>@<dgx-host>:<path>/fresh-rotten-fruit-capstone/dgx_results.zip .
unzip -o dgx_results.zip -d fresh-rotten-fruit-capstone
```

## 8. Application on the DGX instead of the laptop
```bash
python app/app.py --host 127.0.0.1 --port 7860
# on your laptop, in another terminal:
ssh -L 7860:localhost:7860 <user>@<dgx-host>      # then open http://127.0.0.1:7860
```
Do not expose the port to the whole network unless your site allows it.
