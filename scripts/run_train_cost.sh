set -k
for random_action in True False; do
    echo 
    if [ $random_action == True ]
    then
        for random_std_v in 1 0.5 0.1; do
            for random_std_r in 1 0.5 0.1; do
                for cost_dropout in True False; do
                        sbatch submit_train_cost.slurm \
                                random_action=$random_action \
                                random_std_v=$random_std_v \
                                random_std_r=$random_std_r \
                                cost_dropout=$cost_dropout
                done
            done
        done
    else
        for random_std_v in 0; do
            for random_std_r in 0; do
                for cost_dropout in True False; do
                                            sbatch submit_train_cost.slurm \
                                                random_action=$random_action \
                                                random_std_v=$random_std_v \
                                                random_std_r=$random_std_r \
                                                cost_dropout=$cost_dropout
                done
            done
        done
    fi
done
