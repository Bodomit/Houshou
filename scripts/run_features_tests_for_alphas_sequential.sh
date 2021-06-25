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

# Invoke with sbatch ./scripts/run_feature_tests_for_alphas_sequential.sh $RESULTS_ROOT_DIR

module add nvidia-cuda

RESULTS_ROOT_DIR=$1
echo "RESULTS_ROOT_DIR: $RESULTS_ROOT_DIR"

ALPHAS=(0 0.0001 0.001 0.01 0.1 0.3 0.5 0.7 0.9 0.99 0.999 0.9999 0.75 0.8 0.85)

echo "CPU Stats"
python -c "import os; print('CPUS: ', len(os.sched_getaffinity(0)))"
echo ""

echo "GPU Stats:"
nvidia-smi
echo ""

for ALPHA in "${ALPHAS[@]}"
do
    
    RESULTSDIR=$RESULTS_ROOT_DIR/$ALPHA

    echo "ALPHA: $ALPHA"
    echo "RESULTSDIR: $RESULTSDIR"

    python -m features_test $RESULTSDIR
done

python -m lambda_charts $RESULTS_ROOT_DIR $RESULTS_ROOT_DIR/charts
