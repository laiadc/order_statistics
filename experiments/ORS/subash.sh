#!/bin/bash
#
# HTCondor Job Execution Script (with diagnostics)
#

set -x
set -euo pipefail

# Limit threads to match Condor request
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export SCIPY_OPENBLAS_NUM_THREADS=1   # ← add this

echo "========== JOB START =========="
date
hostname
echo "User: $(whoami)"
echo "PWD: $(pwd)"
echo "HOME: $HOME"
echo "_CONDOR_SCRATCH_DIR: $_CONDOR_SCRATCH_DIR"
echo "================================"

# Extract command-line arguments
ns=$1
gates_name=$2
degree=$3
num_gates=$4
n_layers=$5

echo "========== INPUT PARAMETERS =========="
echo "ns=$ns"
echo "gates_name=$gates_name"
echo "degree=$degree"
echo "num_gates=$num_gates"
echo "n_layers=$n_layers"
echo "======================================"

# Navigate to project directory (use absolute path for safety)
PROJECT_DIR="/nfs/pic.es/user/l/ldomingo/complexity/HPC"

echo "=== Changing to project directory: $PROJECT_DIR ==="
cd "$PROJECT_DIR" || { echo "ERROR: Failed to cd to $PROJECT_DIR"; exit 1; }

# Check Python environment path
ENV_PATH="/nfs/pic.es/user/l/ldomingo/quantumenv/bin/activate"

echo "=== Activating environment: $ENV_PATH ==="
if [ -f "$ENV_PATH" ]; then
    source "$ENV_PATH"
else
    echo "ERROR: Virtual environment not found at $ENV_PATH"
    exit 1
fi

echo "=== Python info ==="
which python3
python --version

# Check script exists
echo "=== Checking Python script ==="
if [ ! -f "run_ORS_noisy_multibasis.py" ]; then
    echo "ERROR: run_ORS_noisy_multibasis.py not found!"
    exit 1
fi

echo "========== EXECUTING PYTHON =========="

# Run Python unbuffered to ensure logs appear
taskset -c 0 python run_ORS_noisy_multibasis.py \
    --ns "$ns" \
    --gates-name "$gates_name" \
    --degree "$degree" \
    --num-gates "$num_gates" \
    --n-layers "$n_layers"

exit_code=$?

echo "========== PYTHON FINISHED =========="
echo "Exit code: $exit_code"

# Check exit status
if [ $exit_code -eq 0 ]; then
    echo "Job completed successfully"
else
    echo "Job failed with exit code $exit_code"
    deactivate
    exit $exit_code
fi

# Deactivate virtual environment
echo "=== Deactivating environment ==="
deactivate

echo "========== JOB END =========="
date