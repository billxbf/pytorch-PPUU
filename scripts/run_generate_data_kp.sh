set -k
for ksize in 7; do
  for position_threshold in 1; do
    for t in 0 1 2; do
    colored_lane=$ksize"g"$position_threshold"actrajectory.png"
    data_dir=traffic-data/state-action-cost-$ksize-$position_threshold/
    sbatch submit_generate_data_kp.slurm \
            colored_lane=$colored_lane \
            data_dir=$data_dir \
            t=$t
    done
  done
done
