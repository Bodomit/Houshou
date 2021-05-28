#!/usr/bin/env bash
#SBATCH --job-name=Houshou
#SBATCH --partition=k2-gpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=200GB
#SBATCH --gres=gpu:1
#SBATCH --output=/mnt/scratch2/users/40057686/logs/houshou/%A-%a.log
#SBATCH --time=60

# Invoke with sbatch --array=0-10 ./scripts/run_features_tests_for_alphas.sh $RESULTS_ROOT_DIR

module add nvidia-cuda

RESULTS_ROOT_DIR=$1
echo "RESULTS_ROOT_DIR: $RESULTS_ROOT_DIR"

ALPHA_ID=${SLURM_ARRAY_TASK_ID:-0}

echo "ALPHA_ID: $ALPHA_ID"

ALPHAS=(0 0.001 0.01 0.1 0.3 0.5 0.7 0.9 0.99 0.999 0.9999)
ALPHA=${ALPHAS[$ALPHA_ID]}

RESULTSDIR=$RESULTS_ROOT_DIR/$ALPHA

echo "ALPHA: $ALPHA"
echo "RESULTSDIR: $RESULTSDIR"

echo "CPU Stats"
python -c "import os; print('CPUS: ', len(os.sched_getaffinity(0)))"
echo ""

echo "GPU Stats:"
nvidia-smi
echo ""

srun python -m features_test $RESULTSDIR
