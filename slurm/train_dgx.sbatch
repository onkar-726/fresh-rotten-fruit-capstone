#!/bin/bash
# OPTIONAL: only if your DGX uses the Slurm scheduler. Edit the #SBATCH lines to your site's rules
# (partition names, GPU counts, time limits differ per installation).  Submit with:  sbatch slurm/train_dgx.sbatch
#SBATCH --job-name=fruit-train
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x-%j.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
mkdir -p logs
# Activate your environment here (conda / venv / module load ...). If you use the NGC TensorFlow container,
# run this script INSIDE it:  srun --container-image=<image> bash scripts/run_dgx_pipeline.sh   (site-specific)
bash scripts/run_dgx_pipeline.sh
