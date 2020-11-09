#!/bin/bash
#
#SBATCH --job-name=generate_data_kp
#SBATCH --output=logs/generate_data_kp_%j.out
#SBATCH --error=logs/generate_data_kp_%j.err
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=4
#SBATCH --nodes=1
#SBATCH --mem=10gb
#SBATCH --mail-type=END,FAIL # notifications for job done & fail

. ~/.bashrc
conda activate PPUU

cd ../
python -u generate_trajectories.py \
    -map i80 \
    -time_slot $t \
    -colored_lane $colored_lane \
    -data_dir $data_dir \
    -offroad_map 15g1actrajectory_offroad.png
