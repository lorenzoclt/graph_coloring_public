#!/bin/bash

#SBATCH --job-name=gen_graphs
#SBATCH --output=generate_dataset_%j.log
#SBATCH --error=generate_dataset_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --partition=bare-metal-GPU
#SBATCH --time=2-23:59:00
#SBATCH --export=ALL
#SBATCH --mem=0

set -euo pipefail

echo "=========================================="
echo "SLURM Job Information"
echo "=========================================="
echo "Job ID       : ${SLURM_JOB_ID:-unknown}"
echo "Job Name     : ${SLURM_JOB_NAME:-unknown}"
echo "Node         : ${SLURM_NODELIST:-unknown}"
echo "CPUs/Task    : ${SLURM_CPUS_PER_TASK:-unknown}"
echo "Start        : $(date)"
echo "=========================================="

# Parameters. You can override these with sbatch --export=ALL,VAR=value.
N_LIST="${N_LIST:-10000}"
C_LIST="${C_LIST:-15.5 16 16.5 17}"
NUM_COLORS="${NUM_COLORS:-5}"
NUM_GRAPHS="${NUM_GRAPHS:-20}"
SPLIT="${SPLIT:-0.0 0.0 1.0}"
SEED="${SEED:-4}"
SEED_STEP="${SEED_STEP:-4}"
OUTPUT_DIR="${OUTPUT_DIR:-data_confrontolayers}"
DATASET_TYPE="${DATASET_TYPE:-plt}"
LOG_EVERY="${LOG_EVERY:-0}"

# Virtual environment.
VENV_DIR="${VENV_DIR:-$HOME/colantol/venvs/myenv}"
echo "Using virtual environment: $VENV_DIR"
export PYTHONPATH="$VENV_DIR/lib/python3.9/site-packages:${PYTHONPATH:-}"
export PATH="$VENV_DIR/bin:$PATH"

# CPU threading.
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-32}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-32}"
export PYTHONUNBUFFERED=1

cd /home/c/colantol/colantol/graph_coloring_public-main
echo "Working directory: $(pwd)"
echo ""

echo "Python version:"
python3 --version
which python3
echo ""

echo "Required packages:"
python3 -c "import torch; print(f'torch: {torch.__version__}')"
python3 -c "import torch_geometric as pyg; print(f'torch_geometric: {pyg.__version__}')"
python3 -c "import tqdm; print(f'tqdm: {tqdm.__version__}')"
echo ""

read -r -a N_ARGS <<< "$N_LIST"
read -r -a C_ARGS <<< "$C_LIST"
read -r -a SPLIT_ARGS <<< "$SPLIT"

echo "Generation parameters"
echo "N_LIST      : $N_LIST"
echo "C_LIST      : $C_LIST"
echo "NUM_COLORS  : $NUM_COLORS"
echo "NUM_GRAPHS  : $NUM_GRAPHS"
echo "SPLIT       : $SPLIT"
echo "OUTPUT_DIR  : $OUTPUT_DIR"
echo "DATASET_TYPE: $DATASET_TYPE"
echo "=========================================="

srun python3 -u create_graph_dataset.py \
  -N "${N_ARGS[@]}" \
  -q "$NUM_COLORS" \
  -g "$NUM_GRAPHS" \
  -c "${C_ARGS[@]}" \
  --output-dir "$OUTPUT_DIR" \
  --create-dataset-layout \
  --dataset-type "$DATASET_TYPE" \
  --split "${SPLIT_ARGS[@]}" \
  --seed "$SEED" \
  --seed-step "$SEED_STEP" \
  --log-every "$LOG_EVERY"

echo ""
echo "=========================================="
echo "Job completed: $(date)"
echo "=========================================="
