#!/bin/bash

#SBATCH --job-name=denoisinggc
#SBATCH --output=denoising_%A_%a.log
#SBATCH --error=denoising_%A_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --partition=bare-metal-GPU
#SBATCH --gres=gpu:tesla:1
#SBATCH --array=0-4
#SBATCH --time=2-23:59:00
#SBATCH --export=ALL
#SBATCH --mem=0

set -euo pipefail

echo "=========================================="
echo "SLURM Job Information"
echo "=========================================="
echo "Job ID       : ${SLURM_JOB_ID:-unknown}"
echo "Array Job ID : ${SLURM_ARRAY_JOB_ID:-none}"
echo "Array Task ID: ${SLURM_ARRAY_TASK_ID:-none}"
echo "Job Name     : ${SLURM_JOB_NAME:-unknown}"
echo "Node         : ${SLURM_NODELIST:-unknown}"
echo "GPUs on node : ${SLURM_GPUS_ON_NODE:-unknown}"
echo "CPUs/Task    : ${SLURM_CPUS_PER_TASK:-unknown}"
echo "Start        : $(date)"
echo "=========================================="

# Parameters. Override with:
# sbatch --array=0-4 --export=ALL,NODE_LIST="5k 10k 20k",C_LIST="12.50 13.10",NUM_ITERATIONS="1000 5000",ALPHA_MIN="0.05 0.25",ALPHA_MAX="0.65 0.75 0.9" job_denoising.sbatch
# Each array task owns one shard and runs every NODE_LIST value sequentially.
PROJECT_DIR="${PROJECT_DIR:-/home/c/colantol/colantol/graph_coloring_public-main}"
DATA_ROOT="${DATA_ROOT:-data_training_modelb_corretti}"
GRAPH_TYPE_DIR="${GRAPH_TYPE_DIR:-plt}"
NODE_LIST="${NODE_LIST:-${NUM_NODES_LIST:- 10000}}"
SHARD_COUNT="${SHARD_COUNT:-10}"
DATA_DIR="${DATA_DIR:-}"
C_LIST="${C_LIST:- 14.0}"
SPLIT="${SPLIT:-test}"
DENOISING_SCRIPT="${DENOISING_SCRIPT:-denoising_ottimizzato.py}"
NUM_ITERATIONS="${NUM_ITERATIONS:-50000}" #  41943  167772 671089 2684354
ALPHA_MIN="${ALPHA_MIN:-0.25}"
ALPHA_MAX="${ALPHA_MAX:-0.90}"
CHECKPOINT="${CHECKPOINT:-training_modelb/t0_20260721-144800/checkpoint-epoch=105.ckpt}" #training_modelb/t0_20260721-144800/checkpoint-epoch=105.ckpt
LOG_FILE="${LOG_FILE:-}"
LOG_FILE_TEMPLATE="${LOG_FILE_TEMPLATE:-Logs_figura1modelb_martedì/{nodes}/logfile.csv}"
NUM_WORKERS="${NUM_WORKERS:-0}"
BATCH_SIZE="${BATCH_SIZE:-10}"
CUDA_STREAMS="${CUDA_STREAMS:-5}"
AGGREGATION_MODE="${AGGREGATION_MODE:-segment}"
PHI_INPUT_MODE="${PHI_INPUT_MODE:-factorized}"
DROP_SOLVED_GRAPHS="${DROP_SOLVED_GRAPHS:-1}"
PROFILE_WARMUP="${PROFILE_WARMUP:-20}"
PROFILE_STEPS="${PROFILE_STEPS:-25}"
PROFILE_TRACE="${PROFILE_TRACE:-}"
PROFILE_BATCH_SIZES="${PROFILE_BATCH_SIZES:-1 4 8 16}"
CHECK_EVERY="${CHECK_EVERY:-1}"
PROGRESS_EVERY="${PROGRESS_EVERY:-1000}"
SHARD_SIZE="${SHARD_SIZE:-4}"
ARRAY_TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
MAX_GRAPHS="${MAX_GRAPHS:-$SHARD_SIZE}"
SEED="${SEED:-42}"

path_for_check() {
  local path="$1"
  if [[ "$path" = /* ]]; then
    echo "$path"
  else
    echo "$PROJECT_DIR/$path"
  fi
}

node_dir_from_value() {
  local raw="$1"
  local value="${raw,,}"
  local data_root="${DATA_ROOT%/}"
  local data_root_check
  data_root_check="$(path_for_check "$data_root")"

  if [[ -d "$data_root_check/$value" ]]; then
    echo "$value"
    return
  fi

  if [[ "$value" =~ ^[0-9]+$ ]]; then
    local candidate

    if (( value % 1000 == 0 )); then
      candidate="$((value / 1000))k"
      if [[ -d "$data_root_check/$candidate" ]]; then
        echo "$candidate"
        return
      fi
    fi

    if (( value % 1024 == 0 )); then
      candidate="$((value / 1024))k"
      if [[ -d "$data_root_check/$candidate" ]]; then
        echo "$candidate"
        return
      fi
    fi
  fi

  echo "$value"
}

normalize_log_file_path() {
  local value="$1"

  # Accept the current template and common malformed overrides from older runs.
  value="${value//Logs_\{nodes_highc\//Logs_highc\/}"
  value="${value//Logs_\{nodes_modellob\//Logs_modellob\/}"
  value="${value//\{nodes\}/$NODE_DIR}"
  value="${value//\{nodes/$NODE_DIR}"
  value="${value//\}/}"

  echo "$value"
}

read -r -a NODE_ARGS <<< "${NODE_LIST//,/ }"
if (( ${#NODE_ARGS[@]} == 0 )); then
  echo "ERROR: NODE_LIST is empty" >&2
  exit 1
fi

if (( SHARD_COUNT < 1 )); then
  echo "ERROR: SHARD_COUNT must be >= 1" >&2
  exit 1
fi

EXPLICIT_DATA_DIR="$DATA_DIR"
USER_LOG_FILE="$LOG_FILE"
GRAPH_OFFSET_OVERRIDE="${GRAPH_OFFSET:-}"

SHARD_INDEX="$ARRAY_TASK_ID"
if (( SHARD_INDEX >= SHARD_COUNT )); then
  max_array_task=$((SHARD_COUNT - 1))
  echo "ERROR: SLURM_ARRAY_TASK_ID=$ARRAY_TASK_ID is outside SHARD_COUNT=$SHARD_COUNT." >&2
  echo "       This script now runs every NODE_LIST value sequentially inside each shard task." >&2
  echo "       Use --array=0-$max_array_task for $SHARD_COUNT shard tasks." >&2
  exit 1
fi

# Model parameters.
NUM_COLORS="${NUM_COLORS:-5}"
LATENT_DIM="${LATENT_DIM:-64}"
NUM_LAYERS="${NUM_LAYERS:-10}"
DROPOUT_RATE="${DROPOUT_RATE:-0.0}"
TEMP="${TEMP:-10.0}"
COMPILE_MODEL="${COMPILE_MODEL:-1}"
COMPILE_MODE="${COMPILE_MODE:-reduce-overhead}"

# Virtual environment.
VENV_DIR="${VENV_DIR:-$HOME/colantol/venvs/myenv}"
echo "Using virtual environment: $VENV_DIR"
export PYTHONPATH="$VENV_DIR/lib/python3.9/site-packages:${PYTHONPATH:-}"
export PATH="$VENV_DIR/bin:$PATH"

# Runtime settings.
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-max_split_size_mb:512,expandable_segments:True}"

cd "$PROJECT_DIR"
echo "Working directory: $(pwd)"
echo ""

echo "Python version:"
python3 --version
which python3
echo ""

echo "GPU Information:"
nvidia-smi --query-gpu=index,name,memory.total,memory.free,driver_version --format=csv,noheader || true
echo ""

echo "Required packages:"
python3 -c "import torch; print(f'torch: {torch.__version__}, cuda: {torch.cuda.is_available()}, devices: {torch.cuda.device_count()}')"
python3 -c "import torch_geometric as pyg; print(f'torch_geometric: {pyg.__version__}')"
python3 -c "import torch_scatter; print('torch_scatter: ok')"
echo ""

read -r -a C_ARGS <<< "$C_LIST"
read -r -a ITER_ARGS <<< "$NUM_ITERATIONS"
read -r -a ALPHA_MIN_ARGS <<< "$ALPHA_MIN"
read -r -a ALPHA_MAX_ARGS <<< "$ALPHA_MAX"

run_denoising_for_node() {
  local node_index="$1"
  local node_value="$2"
  local node_dir
  local data_dir
  local data_dir_check
  local graph_offset
  local log_file
  local log_stem
  local log_ext
  local -a cmd

  node_dir="$(node_dir_from_value "$node_value")"

  if [[ -n "$EXPLICIT_DATA_DIR" ]]; then
    data_dir="$EXPLICIT_DATA_DIR"
  else
    data_dir="${DATA_ROOT%/}/$node_dir/$GRAPH_TYPE_DIR"
  fi

  data_dir_check="$(path_for_check "$data_dir")"
  if [[ ! -d "$data_dir_check" ]]; then
    echo "ERROR: DATA_DIR does not exist: $data_dir" >&2
    echo "       Check DATA_ROOT, NODE_LIST, and GRAPH_TYPE_DIR." >&2
    exit 1
  fi

  if [[ -n "$GRAPH_OFFSET_OVERRIDE" ]]; then
    graph_offset="$GRAPH_OFFSET_OVERRIDE"
  else
    graph_offset="$((SHARD_INDEX * SHARD_SIZE))"
  fi

  if [[ -z "$USER_LOG_FILE" ]]; then
    NODE_DIR="$node_dir"
    log_file="$(normalize_log_file_path "$LOG_FILE_TEMPLATE")"
  else
    NODE_DIR="$node_dir"
    log_file="$(normalize_log_file_path "$USER_LOG_FILE")"
  fi
  if [[ "$log_file" == *"{"* || "$log_file" == *"}"* ]]; then
    echo "ERROR: LOG_FILE contains unresolved braces after normalization: $log_file" >&2
    echo "       Use a template like Logs_highc/{nodes}/logfile.csv" >&2
    exit 1
  fi

  log_stem="${log_file%.csv}"
  log_ext="${log_file##*.}"
  if [[ "$log_ext" == "$log_file" ]]; then
    log_file="${log_file}_shard${SHARD_INDEX}"
  else
    log_file="${log_stem}_shard${SHARD_INDEX}.${log_ext}"
  fi

  cmd=(
    python3 -u "$DENOISING_SCRIPT"
    --data-dir "$data_dir"
    --connectivities "${C_ARGS[@]}"
    --split "$SPLIT"
    --num-iterations "${ITER_ARGS[@]}"
    --alpha-min "${ALPHA_MIN_ARGS[@]}"
    --alpha-max "${ALPHA_MAX_ARGS[@]}"
    --checkpoint "$CHECKPOINT"
    --log-file "$log_file"
    --batch-size "$BATCH_SIZE"
    --num-workers "$NUM_WORKERS"
    --cuda-streams "$CUDA_STREAMS"
    --aggregation-mode "$AGGREGATION_MODE"
    --phi-input-mode "$PHI_INPUT_MODE"
    --profile-warmup "$PROFILE_WARMUP"
    --profile-steps "$PROFILE_STEPS"
    --check-every "$CHECK_EVERY"
    --progress-every "$PROGRESS_EVERY"
    --num-colors "$NUM_COLORS"
    --latent-dim "$LATENT_DIM"
    --num-layers "$NUM_LAYERS"
    --dropout-rate "$DROPOUT_RATE"
    --temp "$TEMP"
    --seed "$SEED"
  )

  if [[ "$COMPILE_MODEL" == "1" ]]; then
    cmd+=(--compile-model --compile-mode "$COMPILE_MODE")
  fi

  if [[ "$DROP_SOLVED_GRAPHS" == "0" ]]; then
    cmd+=(--no-drop-solved-graphs)
  else
    cmd+=(--drop-solved-graphs)
  fi

  if [[ -n "$PROFILE_TRACE" ]]; then
    cmd+=(--profile-trace "$PROFILE_TRACE")
  fi

  if [[ -n "$PROFILE_BATCH_SIZES" ]]; then
    local -a profile_batch_args
    read -r -a profile_batch_args <<< "$PROFILE_BATCH_SIZES"
    cmd+=(--profile-batch-sizes "${profile_batch_args[@]}")
  fi

  if [[ -n "$MAX_GRAPHS" ]]; then
    cmd+=(--max-graphs "$MAX_GRAPHS" --graph-offset "$graph_offset")
  fi

  echo "Denoising parameters"
  echo "PROJECT_DIR   : $PROJECT_DIR"
  echo "DATA_ROOT     : $DATA_ROOT"
  echo "GRAPH_TYPE_DIR: $GRAPH_TYPE_DIR"
  echo "NODE_LIST     : $NODE_LIST"
  echo "NODE_INDEX    : $node_index"
  echo "NODE_VALUE    : $node_value"
  echo "NODE_DIR      : $node_dir"
  echo "SHARD_COUNT   : $SHARD_COUNT"
  echo "SHARD_INDEX   : $SHARD_INDEX"
  echo "DATA_DIR      : $data_dir"
  echo "C_LIST        : $C_LIST"
  echo "SPLIT         : $SPLIT"
  echo "SCRIPT        : $DENOISING_SCRIPT"
  echo "NUM_ITERATIONS: $NUM_ITERATIONS"
  echo "ALPHA_MIN/MAX : $ALPHA_MIN / $ALPHA_MAX"
  echo "CHECKPOINT    : $CHECKPOINT"
  echo "LOG_FILE      : $log_file"
  echo "BATCH_SIZE    : $BATCH_SIZE"
  echo "NUM_WORKERS   : $NUM_WORKERS"
  echo "CUDA_STREAMS  : $CUDA_STREAMS"
  echo "AGGREGATION   : $AGGREGATION_MODE"
  echo "PHI_INPUT     : $PHI_INPUT_MODE"
  echo "DROP_SOLVED   : $DROP_SOLVED_GRAPHS"
  echo "PROFILE_WARMUP: $PROFILE_WARMUP"
  echo "PROFILE_STEPS : $PROFILE_STEPS"
  echo "PROFILE_TRACE : ${PROFILE_TRACE:-derived from LOG_FILE}"
  echo "PROFILE_BATCH : ${PROFILE_BATCH_SIZES:-single batch}"
  echo "CHECK_EVERY   : $CHECK_EVERY"
  echo "PROGRESS_EVERY: $PROGRESS_EVERY"
  echo "SHARD_SIZE    : $SHARD_SIZE"
  echo "GRAPH_OFFSET  : $graph_offset"
  echo "COMPILE_MODEL : $COMPILE_MODEL"
  echo "COMPILE_MODE  : $COMPILE_MODE"
  echo "MAX_GRAPHS    : ${MAX_GRAPHS:-none}"
  echo "=========================================="

  srun "${cmd[@]}"
}

if [[ -n "$EXPLICIT_DATA_DIR" ]]; then
  run_denoising_for_node 0 "${NODE_ARGS[0]}"
else
  for NODE_INDEX in "${!NODE_ARGS[@]}"; do
    run_denoising_for_node "$NODE_INDEX" "${NODE_ARGS[$NODE_INDEX]}"
  done
fi

echo ""
echo "=========================================="
echo "Job completed: $(date)"
echo "=========================================="
