#!/usr/bin/env bash
#SBATCH --job-name=Houshou
#SBATCH --partition=k2-gpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=100GB
#SBATCH --gres=gpu:1
#SBATCH --output=/mnt/scratch2/users/40057686/logs/houshou/%A-%a.log
#SBATCH --time=3-0

# Invoke with sbatch --array=0-10 ./scripts/run_config_for_alphas.sh $CONFIG_PATH $DATASETS_ROOT_DIR $RESULTS_ROOT_DIR

# module add nvidia-cuda

CONFIG_PATH=$1
CONFIG_NAME="$(basename $CONFIG_PATH .yaml)"

echo "CONFIG_PATH: $CONFIG_PATH"
echo "CONFIG_NAME: $CONFIG_NAME"

DATASETS_ROOT_DIR=$2
echo "DATASETS_ROOT_DIR: $DATASETS_ROOT_DIR"

RESULTS_ROOT_DIR=$3
echo "RESULTS_ROOT_DIR: $RESULTS_ROOT_DIR"

ALPHA_ID=${SLURM_ARRAY_TASK_ID:-0}

echo "ALPHA_ID: $ALPHA_ID"

ALPHAS=(0 0.001 0.01 0.1 0.3 0.5 0.7 0.9 0.99 0.999 0.9999)
ALPHA=${ALPHAS[$ALPHA_ID]}

RESULTSDIR=$RESULTS_ROOT_DIR/houshou/$CONFIG_NAME/$ALPHA
VGGFACE2_DATADIR=$DATASETS_ROOT_DIR/vggface2_MTCNN

echo "ALPHA: $ALPHA"
echo "RESULTSDIR: $RESULTSDIR"
echo "VGGFACE2_DATADIR: $VGGFACE2_DATADIR"

echo "CPU Stats"
python -c "import os; print('CPUS: ', len(os.sched_getaffinity(0)))"
echo ""

echo "GPU Stats:"
nvidia-smi
echo ""

python -m features_train \
    --data houshou.data.VGGFace2 \
    --data.init_args.data_dir $VGGFACE2_DATADIR \
    --trainer.default_root_dir $RESULTSDIR \
    --config $CONFIG_PATH \
