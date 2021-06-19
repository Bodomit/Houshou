#!/usr/bin/env bash
#SBATCH --job-name=Houshou
#SBATCH --partition=k2-gpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=100GB
#SBATCH --gres=gpu:1
#SBATCH --output=/mnt/scratch2/users/40057686/logs/houshou/%A-%a.log
#SBATCH --time=3-0
#SBATCH --signal=SIGUSR1@90

# Invoke with sbatch ./scripts/run_config_for_alphas_sequential.sh $CONFIG_PATH $RESULTS_ROOT_DIR

module add nvidia-cuda

CONFIG_PATH=$1
CONFIG_NAME="$(basename $CONFIG_PATH .yaml)"

echo "CONFIG_PATH: $CONFIG_PATH"
echo "CONFIG_NAME: $CONFIG_NAME"

RESULTS_ROOT_DIR=$2
echo "RESULTS_ROOT_DIR: $RESULTS_ROOT_DIR"

ALPHAS=(0 0.001 0.01 0.1 0.3 0.5 0.7 0.9 0.99 0.999 0.9999)

echo "CPU Stats"
python -c "import os; print('CPUS: ', len(os.sched_getaffinity(0)))"
echo ""

echo "GPU Stats:"
nvidia-smi
echo ""

BASERESULTSDIR=$RESULTS_ROOT_DIR/houshou/$CONFIG_NAME/

for ALPHA in "${ALPHAS[@]}"
do
    
    RESULTSDIR=$BASERESULTSDIR/$ALPHA

    echo "ALPHA: $ALPHA"
    echo "RESULTSDIR: $RESULTSDIR"

    srun python -m features_train \
        --config $CONFIG_PATH \
        --trainer.default_root_dir $RESULTSDIR \
        --model.init_args.lambda_value $ALPHA 

    srun python -m features_test $RESULTSDIR

    srun python -m attribute_train_test $RESULTSDIR
done

srun python -m lambda_charts $BASERESULTSDIR $BASERESULTSDIR/charts
