#!/usr/bin/env bash
#SBATCH --job-name=Houshou
#SBATCH --partition=k2-gpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=100GB
#SBATCH --gres=gpu:1
#SBATCH --output=/mnt/scratch2/users/40057686/logs/houshou/%A-%a.log
#SBATCH --time=3-0

# Invoke with sbatch --array=0,1 ./scripts/run_attacker_aem_for_config $RESULTSDIR

module add nvidia-cuda


RESULTSDIR=$1
echo "RESULTSDIR: $RESULTSDIR"

EPOCH_ID=${SLURM_ARRAY_TASK_ID:-0}
echo "EPOCH_ID: $EPOCH_ID"

EPOCHS=(10 30)
EPOCH=${EPOCHS[$EPOCH_ID]}

echo "EPOCH: $EPOCH"
echo "RESULTSDIR: $RESULTSDIR"

echo "CPU Stats"
python -c "import os; print('CPUS: ', len(os.sched_getaffinity(0)))"
echo ""

echo "GPU Stats:"
nvidia-smi
echo ""

srun python -m attacker_attribute_train_test $RESULTSDIR --n-epochs EPOCH