import numpy as np
# Sklearn imports
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.preprocessing import MinMaxScaler
from quantum_reservoir import QuantumCircQiskit, _qrc_path, _save_qrc
import os
import argparse
from tqdm import tqdm 
from metrics import (
    lorentz_curves,
    order_statistics_score,
    renyi2_entropy,
)

# ── Experiment config ────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
#parser.add_argument('--data-folder', default='data/LiH/', help='Path to the data folder')
parser.add_argument('--qrc-dir', default='results/chemistry/', help='Path to the QRC directory results')
parser.add_argument('--nqbits', type=int, default=8, help='Number of qubits in the quantum reservoir')
parser.add_argument('--ns', type=int, default=10, help='Number of independent random reservoir instances per config')
parser.add_argument('--alpha', type=float, default=1e-7, help='Ridge regularisation parameter')
parser.add_argument('--gates-name', type=str, default='G1', help='Circuit family name passed to QuantumCircQiskit')
parser.add_argument('--degrees', type=list, default=None, help='Degree for IQP circuits (if applicable)')
parser.add_argument('--depths', type=list, default=[], help='Depth of the layered circuits (if applicable)')
parser.add_argument('--scrambler-K', type=list, default=[], help='Depth of the scrambler block (if applicable)')
parser.add_argument('--scrambler-layers', type=list, default=[], help='Number of layers of the scrambler block (if applicable)')
args = parser.parse_args()

DATA_FOLDER = args.data_folder
QRC_DIR = args.qrc_dir
os.makedirs(QRC_DIR, exist_ok=True)

alpha_exp = args.alpha
nqbits_exp = args.nqbits
ns = args.ns
gates_name = args.gates_name
degrees = args.degrees
depths = args.depths
K = args.scrambler_K
L = args.scrambler_layers
nshots = 10000
n_shadows = 10000

# Load data
with open(f'{DATA_FOLDER}/spectrums_LiH.npy', 'rb') as f:
            spectrums = np.load(f)
with open(f'{DATA_FOLDER}/bond_lengths_LiH.npy', 'rb') as f:
            bond_lengths = np.load(f)
with open(f'{DATA_FOLDER}/ground_states_LiH.npy', 'rb') as f:
            ground_states = np.load(f)

# Target: excitation energies
y_exp = np.zeros((ground_states.shape[0], 2))
y_exp[:, 0] = spectrums[:, 1] - spectrums[:, 0]
y_exp[:, 1] = spectrums[:, 2] - spectrums[:, 0]

# Fixed train/test split — identical indices used for all families
_all_idx = np.arange(ground_states.shape[0])
idx_train_exp, idx_test_exp = train_test_split(_all_idx, test_size=0.33, random_state=42)

# Scaler fitted on train targets once (reused across all runs)
_scaler_exp = MinMaxScaler()
y_train_exp_scaled = _scaler_exp.fit_transform(y_exp[idx_train_exp])

def run_qrc_experiment(gates_name, ns=10, **circuit_kwargs):
    """Run ns independent QRC experiments for one circuit family/config.

    Each run:
      1. Builds a new random QuantumCircQiskit reservoir.
      2. Computes quantum features (obs_res) for all 300 ground states.
      3. Fits Ridge regression on the train split.
      4. Evaluates test-set MSE.

    Parameters
    ----------
    gates_name     : str   — circuit family passed to QuantumCircQiskit
    ns             : int   — number of random reservoir instances
    **circuit_kwargs       — forwarded to QuantumCircQiskit (degree, num_gates, etc.)

    Returns
    -------
    obs_res_all : ndarray (ns, 300, n_features)
    y_hat_all   : ndarray (ns, n_test, 2)
    mse_all     : ndarray (ns,)
    """
    obs_res_all, y_hat_all, mse_all = [], [], []
    ors_list, delta_s_list, var_x_list = [], [], []
    for _ in tqdm(range(ns), desc=gates_name, leave=True):
        qrc = QuantumCircQiskit(gates_name, nqbits=nqbits_exp, **circuit_kwargs)

        # LAMBDA
        _, sp, _ = lorentz_curves(qrc.get_reservoir, nqbits_exp, nshots, 1, verbose=False)
        ors = order_statistics_score(sp, D=2**nqbits_exp, K=K)
        #ors = ors[np.isfinite(ors)]
        ors_list.append(ors)
        # DeltaS
        _, delta_s, _ = renyi2_entropy(qrc.get_reservoir, nqbits_exp, n_shadows, 1,
                                            subsystem_size=2, use_statevector=True,
                                            verbose=False)
        delta_s_list.append(delta_s)

        # Quantum features for all 300 ground states
        obs_res = np.array([qrc.run_circuit(gs) for gs in ground_states])
        var_x = np.std(obs_res, axis=0)
        var_x_list.append(var_x)

        obs_train = obs_res[idx_train_exp]
        obs_test  = obs_res[idx_test_exp]

        # Ridge regression on scaled targets
        lm = Ridge(alpha=alpha_exp)
        lm.fit(obs_train, y_train_exp_scaled)

        y_hat = _scaler_exp.inverse_transform(lm.predict(obs_test))
        mse   = np.mean(np.square(y_hat - y_exp[idx_test_exp]))

        obs_res_all.append(obs_res)
        y_hat_all.append(y_hat)
        mse_all.append(mse)

    return np.array(obs_res_all), np.array(y_hat_all), np.array(mse_all), np.array(ors_list), np.array(delta_s_list), np.array(var_x_list)

if gates_name == 'Haar':
    path = _qrc_path("Haar", nqbits_exp, ns)
    obs, y_hat, mse = run_qrc_experiment("Haar", ns=ns)
    _save_qrc(path, obs, y_hat, mse)

elif gates_name == 'IQP':
    for deg in degrees:
        path = _qrc_path("IQP", nqbits_exp, ns, degree=deg)
        obs, y_hat, mse = run_qrc_experiment("IQP", ns=ns, degree=deg)
        _save_qrc(path, obs, y_hat, mse)
        print(f"  deg={deg:2d}  MSE = {np.mean(mse):.8f} ± {np.std(mse):.8f}")

elif gates_name in ['G1', 'G2', 'G3']:
    for depth in depths:
        path = _qrc_path(gates_name, nqbits_exp, ns, depth=depth)
        obs, y_hat, mse = run_qrc_experiment(gates_name, ns=ns, num_gates=depth)
        _save_qrc(path, obs, y_hat, mse)
        print(f"  {gates_name} depth={depth:4d}  MSE = {np.mean(mse):.4f} ± {np.std(mse):.4f}")

elif gates_name == 'Scrambling':
    for L_val in L:
        for K_val in K:
            path = _qrc_path("scrambling_ansatz", nqbits_exp, ns,
                            n_layers=L_val, scrambler_depth=K_val)
            obs, y_hat, mse = run_qrc_experiment(
                    "scrambling_ansatz", ns=ns,
                    n_layers=L_val, scrambler_depth=K_val
                )
            _save_qrc(path, obs, y_hat, mse)
            print(f"  L={L_val} K={K_val}  MSE = {np.mean(mse):.4f} ± {np.std(mse):.4f}")
else:
      print(f"Unknown gates_name: {gates_name}")