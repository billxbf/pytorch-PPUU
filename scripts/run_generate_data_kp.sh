set -k
for ksize in 7 15; do
  for position_threshold in 1 5 10 20 50; do
    for t in 1 2 3; do
    colored_lane=$ksize"g"$position_threshold"actrajectory.jpg"
    data_dir=traffic-data/state-action-cost-$ksize-$position_threshold/
    sbatch submit_generate_data_kp.slurm \
            colored_lane=$colored_lane \
            data_dir=$data_dir \
            t=$t
    done
  done
done
