#!/bin/bash
#
# HTCondor Job Execution Script (with diagnostics)
#
# MAKE THIS EXECUTABLE!!!! chmod +x /nfs/pic.es/user/l/ldomingo/time_series/HPC/subash.sh
set -x
set -euo pipefail

# Limit threads to match Condor request
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1

echo "========== JOB START =========="
date
hostname
echo "User: $(whoami)"
echo "PWD: $(pwd)"
echo "HOME: $HOME"
echo "_CONDOR_SCRATCH_DIR: $_CONDOR_SCRATCH_DIR"
echo "================================"

# Extract command-line arguments
family_gates=$1
num_gates=$2
num_experiments=$3
degree=$4
K=$5
L=$6

echo "========== INPUT PARAMETERS =========="
echo "family_gates=$family_gates"
echo "num_gates=$num_gates"
echo "num_experiments=$num_experiments"
echo "degree=$degree"
echo "K=$K"
echo "L=$L"
echo "======================================"

# Navigate to project directory (use absolute path for safety)
PROJECT_DIR="/nfs/pic.es/user/l/ldomingo/time_series/HPC"

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
if [ ! -f "narma_QRC.py" ]; then
    echo "ERROR: narma_QRC.py not found!"
    exit 1
fi

echo "========== EXECUTING PYTHON =========="

# Run Python unbuffered to ensure logs appear
taskset -c 0 python -u narma_QRC.py \
    --family-gates "$family_gates" \
    --num-experiments "$num_experiments" \
    --num-gates "$num_gates" \
    --degree "$degree" \
    --K "$K" \
    --L "$L"

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