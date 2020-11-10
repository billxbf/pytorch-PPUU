#!/bin/bash
#
#SBATCH --job-name=eval_MPUR
#SBATCH --output=logs/eval_MPUR_%j.out
#SBATCH --error=logs/eval_MPUR_%j.err
#SBATCH --time=00:45:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=7
#SBATCH --nodes=1
#SBATCH --mem=60000  # notifications for job done & fail
#PRINCE PRINCE_GPU_COMPUTE_MODE=default

. ~/.bashrc
conda activate PPUU
cd ..

python -u eval_policy.py \
    -model_dir 'hmodels_fm_pred_30/' \
    -method policy-MPUR \
    -policy_model $policy \
    -colored_lane trajectory.jpg \
    -offroad_map 15g1actrajectory_offroad.png \
    -ksize $ksize \
    -position_threshold $position_threshold \
    -mfile $mfile
