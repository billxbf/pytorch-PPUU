# Allows named arguments
set -k
for step in 5000 10000 15000 20000 25000 30000 35000 40000 45000 50000 55000 60000 65000 70000 75000 80000 85000 90000 95000 100000; do
for ksize in 7; do
    for position_threshold in 1; do
        for range in 1.0; do
		for seed in 1 2 3; do
			for u_reg in 0.2; do
				for lambda_a in 0.0; do
					for n_pred in 50; do
mfile="model=fwd-cnn-vae-fp-layers=3-bsize=64-ncond=20-npred=20-lrt=0.0001-nfeature=256-dropout=0.1-nz=32-beta=1e-06-zdropout=0.5-gclip=5.0-warmstart=0-seed=1-output_h=False-ksize="$ksize"-pt="$position_threshold".model"
policy="MPUR-policy-deterministic-model=vae-zdropout=0.5-nfeature=256-bsize=6-npred="$n_pred"-ureg="$u_reg"-lambdal=0.75-lambdao=0.25-lambdaa="$lambda_a"-gamma=0.99-lrtz=0.0-updatez=0-inferz=False-learnedcost=-seed="$seed"-pad=1-ksize="$ksize"-pt="$position_threshold"-range="$range"-novaluestep"$step".model"
                    sbatch submit_eval_mpur.slurm \
                    	policy=$policy \
                    	ksize=$ksize \
                    	position_threshold=$position_threshold \
                        mfile=$mfile
					done
				done
			done
                done
            done
        done
    done
done

