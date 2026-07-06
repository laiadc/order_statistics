"""Shared utilities for QELM Fourier regression experiments.

Imported by sine_QELM.py (same directory) and by Exp2_ORS_Reff.ipynb
(which appends this directory to sys.path before importing).
"""

import numpy as np
from sklearn.linear_model import LinearRegression
from tqdm import tqdm
from qiskit.quantum_info import SparsePauliOp

from metrics import lorentz_curves, order_statistics_score
from QELM import QuantumCircQiskit


def effective_rank(F, eps=1e-12):
    """Participation ratio of singular values of the centred feature matrix."""
    F = F - F.mean(axis=0, keepdims=True)
    try:
        sv = np.linalg.svd(F, compute_uv=False)
    except np.linalg.LinAlgError:
        from scipy.linalg import svd as sp_svd
        sv = sp_svd(F, compute_uv=False, lapack_driver='gesvd')
    sv = sv[sv > eps]
    if len(sv) == 0:
        return 0.0
    return float(sv.sum()**2 / (sv**2).sum())


def random_pauli_observables(n_obs, n_qubits, seed=42):
    rng = np.random.default_rng(seed)
    paulis = np.array(['I', 'X', 'Y', 'Z'])
    return [SparsePauliOp(''.join(rng.choice(paulis, size=n_qubits)))
            for _ in range(n_obs)]


def generate_target(x, n_a=6, seed=0):
    """Random Fourier series target with (3^n_a - 1)/2 frequency components."""
    rng = np.random.default_rng(seed)
    K = (3**n_a - 1) // 2
    a = rng.uniform(-1, 1, size=K + 1)
    b = rng.uniform(-1, 1, size=K + 1)
    y = np.zeros_like(x)
    for k in range(K + 1):
        y += a[k] * np.cos(k * x) + b[k] * np.sin(k * x)
    return y


def run_qelm_experiment(gates_name, n, ns, x_train, x_test, y_train, y_test,
                        K=4, nobs_list=None, **circuit_kwargs):
    """QELM Fourier regression: ns random reservoirs, exponential encoding.

    For each reservoir instance:
      - ORS is measured on the bare (un-encoded) circuit
      - Feature matrices are built with exponential encoding
      - MSE and R_eff are computed across a sweep of n_obs values

    Parameters
    ----------
    gates_name : str
    n          : int  — number of qubits
    ns         : int  — number of independent reservoir instances
    x_train, x_test, y_train, y_test : arrays
    K          : int  — ORS top-K (default 4)
    nobs_list  : array of ints — observable counts to sweep (default arange(10,1000,50))
    **circuit_kwargs : passed to QuantumCircQiskit (num_gates, degree, …)

    Returns
    -------
    ors       : (ns,)
    mse       : (ns, n_obs)
    reff      : (ns, n_obs)
    nobs_list : (n_obs,)
    """
    if nobs_list is None:
        nobs_list = np.arange(10, 1000, 50)

    all_obs   = random_pauli_observables(int(nobs_list.max()), n, seed=42)
    ors_list, mse_list, reff_list = [], [], []

    for run in tqdm(range(ns), desc=gates_name):
        seed = run * 2

        # ORS: bare reservoir (no encoding)
        np.random.seed(seed)
        _qrc = QuantumCircQiskit(gates_name, nqbits=n, statevector=True, **circuit_kwargs)
        _, sp, _ = lorentz_curves(_qrc.get_reservoir, n, 2000, 1, verbose=False)
        ors_list.append(order_statistics_score(sp, 2**n, K=K)[0])

        # Feature matrices: exponential encoding
        np.random.seed(seed)
        qrc = QuantumCircQiskit(
            gates_name, nqbits=n, theta_scale=1.0,
            observables_type=all_obs, encoding='exponential',
            k_exp=n, statevector=True, **circuit_kwargs)
        F_tr = qrc.feature_matrix(x_train)
        F_te = qrc.feature_matrix(x_test)

        mse_run, reff_run = [], []
        for nobs in nobs_list:
            Ft, Fv = F_tr[:, :nobs], F_te[:, :nobs]
            y_hat  = LinearRegression().fit(Ft, y_train).predict(Fv)
            mse_run.append(float(np.mean((y_hat - y_test)**2)))
            reff_run.append(effective_rank(Ft))
        mse_list.append(mse_run)
        reff_list.append(reff_run)

    return np.array(ors_list), np.array(mse_list), np.array(reff_list), nobs_list
