#!/bin/bash
# Example SLURM job: train the baseline on one GPU.
# Adapt the #SBATCH lines (partition, account, time) to your cluster, then:
#     sbatch scripts/slurm/train.sbatch                       # baseline
#     sbatch scripts/slurm/train.sbatch data.fold=1 run_name=fold1   # with overrides
#SBATCH --job-name=forest-baseline
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=%x_%j.log

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

# export DISFOR_DATA_ROOT=/path/to/data     # if not already set in your ~/.bashrc
source .venv/bin/activate
python scripts/train.py --config configs/statistical_mlp.yaml data.num_workers=8 "$@"
