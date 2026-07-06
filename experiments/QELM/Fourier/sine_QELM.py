import os
# ── CPU throttling — must be set BEFORE numpy/scipy/qiskit are imported ──────
os.environ["OMP_NUM_THREADS"]        = "1"
os.environ["MKL_NUM_THREADS"]        = "1"
os.environ["OPENBLAS_NUM_THREADS"]   = "1"
os.environ["NUMEXPR_NUM_THREADS"]    = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
# ─────────────────────────────────────────────────────────────────────────────

import sys
import argparse
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from tqdm import tqdm
from qiskit.quantum_info import SparsePauliOp

# Import QuantumCircQiskit from the local copy in this folder
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from QELM import QuantumCircQiskit
from metrics import (
    lorentz_curves,
    order_statistics_score,
)
from qelm_utils import run_qelm_experiment, generate_target

# ── Output path ────────────────────────────────────────────────────────────────
def _out_path():
    if family_gates == 'Haar':
        tag = 'Haar'
    elif num_gates is not None:
        tag = f'{family_gates}_ng{num_gates}'
    else:
        tag = f'{family_gates}_deg{int(degree)}'
    return os.path.join(RESULTS_FOLDER, f'{tag}_6q_{ns}runs.npz')


# ── Argument parsing ──────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(
    description="QELM sine regression: Pauli vs Exponential encoding sweep")
parser.add_argument('--family-gates',      type=str,   required=True,
                    help="Circuit family: Haar, D2_random, G1, G2, G3, scrambling_ansatz")
parser.add_argument('--num-gates',         type=int,   default=-1,
                    help="Number of gates for G1/G2/G3 (ignored for Haar/D2_random/scrambling)")
parser.add_argument('--num-experiments',   type=int,   default=100,
                    help="Number of random reservoir instances (ns)")
parser.add_argument('--degree',            type=float, default=-1,
                    help="Expected graph degree for D2_random (-1 = not used)")
parser.add_argument('--K',                 type=int,   default=-1,
                    dest='scrambler_depth',
                    help="Scrambler depth K for scrambling_ansatz (-1 = not used)")
parser.add_argument('--L',                 type=int,   default=-1,
                    dest='n_layers',
                    help="Number of layers L for scrambling_ansatz (-1 = not used)")
args = parser.parse_args()

# Map -1 sentinels → None
family_gates    = args.family_gates
ns              = args.num_experiments
num_gates       = None if args.num_gates       == -1 else args.num_gates
degree          = None if args.degree          == -1 else args.degree
scrambler_depth = None if args.scrambler_depth == -1 else args.scrambler_depth
n_layers        = None if args.n_layers        == -1 else args.n_layers

RESULTS_FOLDER = "/data/cvcqml/common/laia/time_series/results/sine/"
os.makedirs(RESULTS_FOLDER, exist_ok=True)

# ── Data ──────────────────────────────────────────────────────────────────────
N_QUBITS  = 6
N_A       = 6        # Fourier parameter (matches N_QUBITS)
N_SAMPLES = 5000

x_all = np.linspace(0, 2 * np.pi, N_SAMPLES, endpoint=False)
y_all = generate_target(x_all, n_a=N_A, seed=0)
idx_tr, idx_te = train_test_split(np.arange(N_SAMPLES), test_size=0.3, random_state=42)
x_train, x_test = x_all[idx_tr], x_all[idx_te]
y_train, y_test = y_all[idx_tr], y_all[idx_te]

# ── Circuit kwargs ─────────────────────────────────────────────────────────────
circuit_kwargs = {}
if num_gates is not None:
    circuit_kwargs['num_gates'] = num_gates
if degree is not None:
    circuit_kwargs['degree'] = degree

# ── Run ───────────────────────────────────────────────────────────────────────
path = _out_path()

print(f'Running  {family_gates}  num_gates={num_gates}  degree={degree}  ns={ns}')
print(f'Output → {path}')

ors, mse, reff, nobs_list = run_qelm_experiment(
    family_gates, N_QUBITS, ns, x_train, x_test, y_train, y_test,
    **circuit_kwargs)

np.savez_compressed(path, ors=ors, mse=mse, reff=reff, n_obs_list=nobs_list)

print(f'Saved  → {path}')
print(f'ORS:   {ors.mean():.4f} ± {ors.std():.4f}')
print(f'MSE:   {mse[:, -1].mean():.4e}')
print(f'R_eff: {reff[:, -1].mean():.2f}')