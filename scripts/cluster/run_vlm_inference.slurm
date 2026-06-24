#!/bin/bash
#SBATCH --job-name=polvifidel-vlm
#SBATCH --output=logs/vlm_%A_%a.out
#SBATCH --error=logs/vlm_%A_%a.err
#SBATCH --array=0-1          # one task per open-source model (internvl, qwen)
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --partition=gpu
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=yc5201@princeton.edu

# Usage:
#   sbatch run_vlm_inference.slurm
# Runs all 4 prompting conditions for each open-source model in parallel.

module purge
module load anaconda3/2024.2
conda activate polvifidel

export POLVIFIDEL_ROOT=$HOME/POLVIFIDEL

MODELS=(internvl qwen)
MODEL=${MODELS[$SLURM_ARRAY_TASK_ID]}

echo "Running model: $MODEL on $(hostname) with GPU $CUDA_VISIBLE_DEVICES"

python $POLVIFIDEL_ROOT/scripts/04_generate_captions_cluster.py \
    --model $MODEL \
    --conditions baseline visual_aux textual_aux annotated_aux
