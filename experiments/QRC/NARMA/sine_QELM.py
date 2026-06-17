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


# ── Target function ───────────────────────────────────────────────────────────
def generate_target(x, n_a=6, seed=0):
    rng = np.random.default_rng(seed)
    K = (3 ** n_a - 1) // 2
    a = rng.uniform(-1, 1, size=K + 1)
    b = rng.uniform(-1, 1, size=K + 1)
    y = np.zeros_like(x)
    for k in range(K + 1):
        y += a[k] * np.cos(k * x) + b[k] * np.sin(k * x)
    return y


# ── Random Pauli observables ──────────────────────────────────────────────────
def random_pauli_observables(n_obs, n_qubits, seed=42):
    rng = np.random.default_rng(seed)
    paulis = np.array(['I', 'X', 'Y', 'Z'])
    return [SparsePauliOp(''.join(rng.choice(paulis, size=n_qubits)))
            for _ in range(n_obs)]


# ── Persistence helpers ───────────────────────────────────────────────────────
def _sine_path(results_dir, gates_name, nqbits, ns,
               degree=None, depth=None, n_layers=None, scrambler_depth=None):
    tag = gates_name
    if degree          is not None: tag += f"_deg{int(degree)}"
    if depth           is not None: tag += f"_depth{int(depth)}"
    if n_layers        is not None: tag += f"_L{n_layers}"
    if scrambler_depth is not None: tag += f"_K{scrambler_depth}"
    return os.path.join(results_dir, f"{tag}_{nqbits}q_{ns}runs.npz")


def _save_sine(path, mse_p, mse_train_p, mse_e, mse_train_e, yp, ye, n_obs_list):
    np.savez_compressed(path,
                        mse_pauli=np.array(mse_p),
                        mse_train_pauli=np.array(mse_train_p),
                        mse_exp=np.array(mse_e),
                        mse_train_exp=np.array(mse_train_e),
                        y_pred_pauli=np.array(yp),
                        y_pred_exp=np.array(ye),
                        n_obs_list=np.array(n_obs_list))
    print(f"  Saved → {path}")


def _load_sine(path):
    d = np.load(path)
    return (d['mse_pauli'], d['mse_train_pauli'],
            d['mse_exp'],   d['mse_train_exp'],
            d['y_pred_pauli'], d['y_pred_exp'])


# ── Core experiment ───────────────────────────────────────────────────────────
def run_qelm_experiment(gates_name, ns, nqbits, x, y_train, y_test,
                         idx_train, idx_test, n_obs_list, all_observables,
                         **circuit_kwargs):
    """Run ns random-reservoir instances for the sine regression task.

    Both Pauli and Exponential encodings share the same random circuit for
    each run (reproduced by setting the same numpy seed before each
    QuantumCircQiskit call).  Features are computed once per encoding at
    max_n_obs observables and then column-subsetted for the n_obs sweep.

    Parameters
    ----------
    gates_name      : str   — circuit family (Haar, D2_random, G1/G2/G3, …)
    ns              : int   — number of random reservoir instances
    nqbits          : int   — circuit size (= k_exp for exponential encoding)
    x               : 1-D array of input values (all n_samples)
    y_train / y_test: 1-D target arrays
    idx_train / idx_test : index arrays into x / y
    n_obs_list      : 1-D array of observable counts to sweep
    all_observables : list of SparsePauliOp (length = max(n_obs_list))
    **circuit_kwargs: forwarded to QuantumCircQiskit (degree, num_gates, …)

    Returns
    -------
    mse_test_pauli, mse_train_pauli : ndarray (ns, len(n_obs_list))
    mse_test_exp,   mse_train_exp   : ndarray (ns, len(n_obs_list))
    y_pred_pauli, y_pred_exp        : ndarray (ns, len(n_obs_list), n_test)
    """
    n_test  = len(idx_test)
    n_steps = len(n_obs_list)

    mse_test_p  = np.zeros((ns, n_steps))
    mse_train_p = np.zeros((ns, n_steps))
    mse_test_e  = np.zeros((ns, n_steps))
    mse_train_e = np.zeros((ns, n_steps))
    yp_all      = np.zeros((ns, n_steps, n_test))
    ye_all      = np.zeros((ns, n_steps, n_test))

    for run in tqdm(range(ns), desc=gates_name, leave=True):
        seed = run * 2

        # Pauli reservoir
        np.random.seed(seed)
        qrc_p = QuantumCircQiskit(
            gates_name, nqbits=nqbits, encoding='pauli',
            theta_scale=1.0, statevector=True,
            observables_type=all_observables, **circuit_kwargs)

        # Exponential reservoir — same circuit via same seed
        np.random.seed(seed)
        qrc_e = QuantumCircQiskit(
            gates_name, nqbits=nqbits, encoding='exponential',
            k_exp=nqbits, theta_scale=1.0, statevector=True,
            observables_type=all_observables, **circuit_kwargs)

        # Compute all features once at max_n_obs
        feat_p = np.array([qrc_p.run_circuit(xi) for xi in x])
        feat_e = np.array([qrc_e.run_circuit(xi) for xi in x])

        for i, n_obs in enumerate(n_obs_list):
            fp_tr = feat_p[idx_train, :n_obs];  fp_te = feat_p[idx_test, :n_obs]
            fe_tr = feat_e[idx_train, :n_obs];  fe_te = feat_e[idx_test, :n_obs]

            lm_p = LinearRegression().fit(fp_tr, y_train)
            lm_e = LinearRegression().fit(fe_tr, y_train)

            yp_te = lm_p.predict(fp_te);  ye_te = lm_e.predict(fe_te)
            yp_tr = lm_p.predict(fp_tr);  ye_tr = lm_e.predict(fe_tr)

            yp_all[run, i]      = yp_te
            ye_all[run, i]      = ye_te
            mse_test_p[run, i]  = np.mean((yp_te - y_test)  ** 2)
            mse_train_p[run, i] = np.mean((yp_tr - y_train) ** 2)
            mse_test_e[run, i]  = np.mean((ye_te - y_test)  ** 2)
            mse_train_e[run, i] = np.mean((ye_tr - y_train) ** 2)

    return mse_test_p, mse_train_p, mse_test_e, mse_train_e, yp_all, ye_all


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

# ── Experiment parameters ─────────────────────────────────────────────────────
nqbits      = 6          # circuit size / Fourier parameter
n_samples   = 5000
n_obs_list  = np.arange(10, 800, 50)
max_n_obs   = int(n_obs_list.max())

RESULTS_FOLDER = "/data/cvcqml/common/laia/time_series/results/sine/"
os.makedirs(RESULTS_FOLDER, exist_ok=True)

# ── Data ─────────────────────────────────────────────────────────────────────
x = np.linspace(0, 2 * np.pi, n_samples, endpoint=False)
y = generate_target(x, n_a=nqbits, seed=0)

idx_train, idx_test = train_test_split(
    np.arange(n_samples), test_size=0.3, random_state=42)
y_train, y_test = y[idx_train], y[idx_test]

# ── Observables ───────────────────────────────────────────────────────────────
all_observables = random_pauli_observables(max_n_obs, nqbits, seed=42)

# ── Build circuit kwargs (only pass non-None values) ─────────────────────────
circuit_kwargs = {}
if num_gates       is not None: circuit_kwargs['num_gates']       = num_gates
if degree          is not None: circuit_kwargs['degree']          = degree
if scrambler_depth is not None: circuit_kwargs['scrambler_depth'] = scrambler_depth
if n_layers        is not None: circuit_kwargs['n_layers']        = n_layers

# ── Output path ───────────────────────────────────────────────────────────────
path = _sine_path(RESULTS_FOLDER, family_gates, nqbits, ns,
                  degree=degree, depth=num_gates,
                  n_layers=n_layers, scrambler_depth=scrambler_depth)

# ── Run ───────────────────────────────────────────────────────────────────────
print(f"Running {family_gates} ns={ns} | kwargs={circuit_kwargs}")
print(f"Output → {path}")

mse_test_p, mse_train_p, mse_test_e, mse_train_e, y_pred_p, y_pred_e = \
    run_qelm_experiment(family_gates, ns, nqbits, x, y_train, y_test,
                        idx_train, idx_test, n_obs_list, all_observables,
                        **circuit_kwargs)

_save_sine(path, mse_test_p, mse_train_p, mse_test_e, mse_train_e,
           y_pred_p, y_pred_e, n_obs_list)

print(f"{family_gates}  "
      f"Pauli MSE={mse_test_p.mean():.6f}±{mse_test_p.std():.6f}  "
      f"Exp   MSE={mse_test_e.mean():.6f}±{mse_test_e.std():.6f}")
