#!/bin/bash
#SBATCH --job-name=polvifidel-detect
#SBATCH --output=logs/detect_%j.out
#SBATCH --error=logs/detect_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --partition=gpu
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=yc5201@princeton.edu

# Runs Grounding-DINO object detection + face recognition on all 1,200 images.
# Outputs one JSON per image to data/detections/.

module purge
module load anaconda3/2024.2
conda activate polvifidel

export POLVIFIDEL_ROOT=$HOME/POLVIFIDEL

python $POLVIFIDEL_ROOT/scripts/02_detect_entities.py
