#!/bin/bash
#
#SBATCH --job-name=train_cost
#SBATCH --output=logs/train_cost_%j.out
#SBATCH --error=logs/train_cost_%j.err
#SBATCH --time=100:00:00
#SBATCH --gres gpu:p40:1
#SBATCH --cpus-per-task=4
#SBATCH --nodes=1
#SBATCH --mem=48000
#SBATCH --mail-type=END,FAIL # notifications for job done & fail

. ~/.bashrc
conda activate drive

cd ../
python -u train_cost.py \
    -model_dir "models/" \
    -mfile "model=fwd-cnn-vae-fp-layers=3-bsize=64-ncond=20-npred=20-lrt=0.0001-nfeature=256-dropout=0.1-nz=32-beta=1e-06-zdropout=0.5-gclip=5.0-warmstart=0-seed=1-output_h=True-pred_h=False.model" \
    -use_colored_lane True \
    -random_action $random_action \
    -random_std_v $random_std_v \
    -random_std_r $random_std_r \
    -cost_dropout $cost_dropout
