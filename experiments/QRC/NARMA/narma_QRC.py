import os
# ── CPU throttling — must be set BEFORE numpy/scipy/qiskit are imported ──────
os.environ["OMP_NUM_THREADS"]        = "1"
os.environ["MKL_NUM_THREADS"]        = "1"
os.environ["OPENBLAS_NUM_THREADS"]   = "1"
os.environ["NUMEXPR_NUM_THREADS"]    = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
# ─────────────────────────────────────────────────────────────────────────────

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn import metrics
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm
from technical_analysis_indicators import get_regressor_vars
from QRC import QuantumRC_Circuit
import argparse
from metrics import (
    lorentz_curves,
    order_statistics_score,
    renyi2_entropy,
)
from sklearn.linear_model import Ridge
from tqdm import tqdm

# ── Gate-family comparison experiment ────────────────────────────────────────
import os
import numpy as np
from tqdm import tqdm

QRC_DIR = os.path.join("results", "time_series", "NARMA")
os.makedirs(QRC_DIR, exist_ok=True)


# ── Persistence helpers ───────────────────────────────────────────────────────
def _qrc_path(RESULTS_FOLDER, gates_name, nqbits, ns, num_gates=None, degree=None,
              n_layers=None, scrambler_depth=None):
    tag = gates_name
    if num_gates       is not None: tag += f"_ng{num_gates}"
    if degree          is not None: tag += f"_deg{float(degree)}"
    if n_layers        is not None: tag += f"_L{n_layers}"
    if scrambler_depth is not None: tag += f"_K{scrambler_depth}"
    return os.path.join(RESULTS_FOLDER, f"{tag}_{nqbits}q_{ns}runs.npz")


def _save_qrc(path, res):
    """Save the dict returned by run_narma_experiment.

    Pass the result dict directly:
        res = run_narma_experiment(...)
        _save_qrc(path, res)
    """
    np.savez_compressed(
        path,
        mse_p=np.asarray(res['mse_p']),
        mse_e=np.asarray(res['mse_e']),
        y_pred_p=np.asarray(res['y_pred_p']),
        y_pred_e=np.asarray(res['y_pred_e']),
        ors=np.asarray(res['ors']),
        delta_s=np.asarray(res['delta_s']),
        var_x_p=np.asarray(res['var_x_p']),
        var_x_e=np.asarray(res['var_x_e']),
        X_train_p=np.asarray(res['X_train_p']),
        X_test_p=np.asarray(res['X_test_p']),
        X_train_e=np.asarray(res['X_train_e']),
        X_test_e=np.asarray(res['X_test_e']),
    )
    print(f"  Saved → {path}")
    
def generate_narma_fourier_task(
    T=2000, n_train=1400, narma_order=5,
    n_a=6, seed=0, normalize=True, use_g = True,
):
    """
    NARMA-n task with Fourier-rich external input.
    
    The input to the QRC is u_t ∈ [0, 2π] (uniform random).
    The driver s_t = g(u_t) where g has frequencies matching the
    exponential encoding's budget for n_a qubits.
    The target y_{t+1} follows NARMA dynamics in s_t and past y.
    
    Parameters
    ----------
    T : int
        Total length of the time series.
    n_train : int
        Length of training set.
    narma_order : int
        NARMA memory depth. 5 is standard; 10 tests longer memory.
    n_a : int
        Encoding qubit count — determines g's frequency content.
    """
    rng = np.random.default_rng(seed)
    K = (3 ** n_a - 1) // 2  # 364 for n_a=6
    
    # Random Fourier coefficients for g
    a = rng.uniform(-1, 1, size=K + 1)
    b = rng.uniform(-1, 1, size=K + 1)
    
    def g(u):
        """Fourier-rich driver function: g: [0, 2π] → R, with frequencies up to K."""
        u = np.atleast_1d(u)
        out = np.zeros_like(u, dtype=float)
        for k in range(K + 1):
            out += a[k] * np.cos(k * u) + b[k] * np.sin(k * u)
        return out
    
    
    # Normalize g to [0, 0.5] (NARMA expects s_t ∈ [0, 0.5])
    u_probe = np.linspace(0, 2 * np.pi, 5000)
    g_vals = g(u_probe) if use_g else u_probe
    g_min, g_max = g_vals.min(), g_vals.max()
    
    def g_normalized(u):
        if use_g:
            return 0.5 * (g(u) - g_min) / (g_max - g_min)
        return 0.5 * (u - g_min) / (g_max - g_min)
    
    # Generate input sequence u_t and driver s_t
    u = rng.uniform(0, 2 * np.pi, size=T)
    s = g_normalized(u)
    
    # Run NARMA dynamics: need narma_order burn-in steps
    y = np.zeros(T)
    n = narma_order
    for t in range(n - 1, T - 1):
        y[t + 1] = (0.3 * y[t] 
                    + 0.05 * y[t] * np.sum(y[t - n + 1:t + 1])
                    + 1.5 * s[t - n + 1] * s[t] 
                    + 0.1)
    
    # Drop burn-in
    u = u[n:]
    y = y[n:]
    T_eff = len(y)
    assert n_train < T_eff
    
    # QRC input: u_t (the reservoir sees u_t and must predict y_{t+1})
    # Target: y itself
    u = u.reshape(-1, 1)
    y = y.reshape(-1, 1)
    u_train, u_test = u[:n_train], u[n_train:]
    y_train, y_test = y[:n_train], y[n_train:]
    
    scaler = None
    if normalize:
        # Scale the INPUT u to [0, 1] so the QRC encoding sees a standard range
        u_scaler = MinMaxScaler(feature_range=(0, 1))
        u_train = u_scaler.fit_transform(u_train)
        u_test = u_scaler.transform(u_test)
        # Scale the target separately
        y_scaler = MinMaxScaler(feature_range=(0, 1))
        y_train = y_scaler.fit_transform(y_train)
        y_test = y_scaler.transform(y_test)
        scaler = (u_scaler, y_scaler)
    
    return {
        'u_train': u_train, 'u_test': u_test,   # QRC inputs
        'y_train': y_train, 'y_test': y_test,   # targets
        'g': g_normalized,
        'K': K,
        'narma_order': narma_order,
        'scaler': scaler,
    }


def run_narma_experiment(
    gates_name,
    data,
    n_obs_list,
    ns=10,
    nqbits=3,
    k_exp=3,
    theta_scale=2*np.pi,
    n_measure=6,
    alpha=1e-10,
    # Mode controls
    input_mode='exogenous',      # 'exogenous' | 'autoregressive' | 'hybrid'
    feedback='teacher',          # 'teacher' | 'closed_loop'
    dim_s=1,
    dim_y=1,
    # Complexity-metric config
    nshots_ors=2000,
    K_ors=None,
    n_shadows=200,
    subsystem_size=2,
    t_dismiss=0,                  # washout steps to exclude from var_x
    # Training options
    retrain_on_test=False,
    verbose=False,
    **circuit_kwargs,
):
    """
    NARMA analogue of run_sine_experiment for QuantumRC_Circuit.

    Single-pass (fast) when the reservoir trajectory is independent of the
    readout (exogenous OR teacher-forced); rebuilds per n_obs otherwise.
    """
    s_train = data.get('u_train')
    s_test  = data.get('u_test')
    y_train = data['y_train']
    y_test  = data['y_test']
    n_test = y_test.shape[0]
    max_n_obs = n_obs_list[-1]
    if K_ors is None:
        K_ors = 2**nqbits

    single_pass = (input_mode == 'exogenous') or (feedback == 'teacher')

    # ── Result containers ────────────────────────────────────────────────
    mse_p = np.zeros((ns, len(n_obs_list)))
    mse_e = np.zeros((ns, len(n_obs_list)))
    yp_all = np.zeros((ns, len(n_obs_list), n_test))
    ye_all = np.zeros((ns, len(n_obs_list), n_test))
    ors_list, delta_s_list = [], []
    var_x_p_list, var_x_e_list = [], []
    # Keep only the last run's feature matrices (for sanity checks/plotting)
    Xp_tr_last = Xp_te_last = Xe_tr_last = Xe_te_last = None

    def _make_qrc(encoding, seed, n_obs=None):
        np.random.seed(seed)
        qrc = QuantumRC_Circuit(
            nqbits=nqbits, gates_set=gates_name, encoding=encoding,
            k_exp=k_exp, theta_scale=theta_scale, n_measure=n_measure,
            observables_type=[1, 2, 3], alpha=alpha,
            input_mode=input_mode, feedback=feedback,
            dim_s=dim_s, dim_y=dim_y,
            **circuit_kwargs,
        )
        trunc_to = max_n_obs if single_pass else n_obs
        qrc.observables = qrc.observables[:trunc_to]
        qrc.obs_array = np.array([o.to_matrix() for o in qrc.observables])
        return qrc

    def _call_train(qrc, cache=False):
        if input_mode == 'autoregressive':
            return qrc.train(y=y_train, cache_dm_traj=cache)
        return qrc.train(s=s_train, y=y_train, cache_dm_traj=cache)

    def _call_test(qrc, cache=False):
        kwargs = dict(retrain=retrain_on_test, cache_dm_traj=cache)
        if input_mode == 'autoregressive':
            return qrc.test(y=y_test, **kwargs)
        return qrc.test(s=s_test, y=y_test, **kwargs)

    def _project(dm_traj, obs):
        f = np.tensordot(obs, dm_traj, axes=[[2, 1], [1, 2]]).real
        return f.T[:-1]

    def _fit_and_predict(X_tr, y_tr, X_te, y_te, retrain):
        lm = Ridge(alpha=alpha).fit(X_tr, y_tr)
        preds, X_aug, y_aug = [], X_tr.copy(), y_tr.copy()
        for i in range(X_te.shape[0]):
            row = X_te[i:i+1]
            preds.append(lm.predict(row).ravel())
            if retrain:
                X_aug = np.vstack([X_aug, row])
                y_aug = np.vstack([y_aug, y_te[i].reshape(1, -1)])
                lm = Ridge(alpha=alpha).fit(X_aug, y_aug)
        return np.array(preds).ravel()

    desc = f"NARMA [{gates_name}] {input_mode}/{feedback}"
    for run in tqdm(range(ns), desc=desc):
        seed = run * 2

        if single_pass:
            qrc_p = _make_qrc('pauli', seed)
            qrc_e = _make_qrc('exponential', seed)

            _call_train(qrc_p, cache=True); dm_p_tr = qrc_p.dm_traj
            _call_test(qrc_p,  cache=True); dm_p_te = qrc_p.dm_traj
            _call_train(qrc_e, cache=True); dm_e_tr = qrc_e.dm_traj
            _call_test(qrc_e,  cache=True); dm_e_te = qrc_e.dm_traj

            full_obs = qrc_p.obs_array
            Xp_tr = _project(dm_p_tr, full_obs)
            Xp_te = _project(dm_p_te, full_obs)
            Xe_tr = _project(dm_e_tr, full_obs)
            Xe_te = _project(dm_e_te, full_obs)

            # Complexity metrics (reservoir-only; same for both encodings)
            _, sp, _ = lorentz_curves(
                qrc_p.get_reservoir, qrc_p.n_total, nshots_ors, 1, verbose=False)
            ors_list.append(
                order_statistics_score(sp, D=2**qrc_p.n_total, K=K_ors))
            _, delta_s_val, _ = renyi2_entropy(
                qrc_p.get_reservoir, qrc_p.n_total, n_shadows, 1,
                subsystem_size=subsystem_size, use_statevector=True, verbose=False)
            delta_s_list.append(delta_s_val)

            # Input sensitivity per encoding: variance over time
            # (excluding washout transient if specified)
            var_x_p_list.append(np.std(Xp_tr[t_dismiss:], axis=0))
            var_x_e_list.append(np.std(Xe_tr[t_dismiss:], axis=0))

            # Sweep n_obs
            for i, n_obs in enumerate(n_obs_list):
                yp = _fit_and_predict(Xp_tr[:, :n_obs], y_train,
                                      Xp_te[:, :n_obs], y_test, retrain_on_test)
                ye = _fit_and_predict(Xe_tr[:, :n_obs], y_train,
                                      Xe_te[:, :n_obs], y_test, retrain_on_test)
                yp_all[run, i] = yp;  ye_all[run, i] = ye
                mse_p[run, i] = np.mean((yp - y_test.ravel())**2)
                mse_e[run, i] = np.mean((ye - y_test.ravel())**2)
                if verbose:
                    print(f"  run={run} n_obs={n_obs:3d}  "
                          f"pauli={mse_p[run,i]:.4e}  exp={mse_e[run,i]:.4e}")

            # Save last run's feature matrices for inspection
            Xp_tr_last, Xp_te_last = Xp_tr, Xp_te
            Xe_tr_last, Xe_te_last = Xe_tr, Xe_te

        else:
            # Closed-loop: rebuild per n_obs
            qrc_ref = _make_qrc('pauli', seed, n_obs=max_n_obs)
            _, sp, _ = lorentz_curves(
                qrc_ref.get_reservoir, qrc_ref.n_total, nshots_ors, 1, verbose=False)
            ors_list.append(
                order_statistics_score(sp, D=2**qrc_ref.n_total, K=K_ors))
            _, delta_s_val, _ = renyi2_entropy(
                qrc_ref.get_reservoir, qrc_ref.n_total, n_shadows, 1,
                subsystem_size=subsystem_size, use_statevector=True, verbose=False)
            delta_s_list.append(delta_s_val)
            _call_train(qrc_ref)
            # In closed-loop we only have one trajectory available cheaply (Pauli ref)
            var_x_p_list.append(np.std(qrc_ref.X_train[t_dismiss:], axis=0))
            var_x_e_list.append(np.full_like(var_x_p_list[-1], np.nan))

            for i, n_obs in enumerate(n_obs_list):
                qrc_p = _make_qrc('pauli', seed, n_obs=n_obs)
                qrc_e = _make_qrc('exponential', seed, n_obs=n_obs)
                _call_train(qrc_p); yp = _call_test(qrc_p).ravel()
                _call_train(qrc_e); ye = _call_test(qrc_e).ravel()
                yp_all[run, i] = yp; ye_all[run, i] = ye
                mse_p[run, i] = np.mean((yp - y_test.ravel())**2)
                mse_e[run, i] = np.mean((ye - y_test.ravel())**2)
                if verbose:
                    print(f"  run={run} n_obs={n_obs:3d}  "
                          f"pauli={mse_p[run,i]:.4e}  exp={mse_e[run,i]:.4e}")

    # Stack var_x lists; rows of unequal length only happen if reservoirs differ,
    # which they don't here (same nqbits/n_measure across runs)
    return {
        'mse_p': mse_p, 'mse_e': mse_e,
        'y_pred_p': yp_all, 'y_pred_e': ye_all,
        'ors': np.array(ors_list),
        'delta_s': np.array(delta_s_list),
        'var_x_p': np.array(var_x_p_list),
        'var_x_e': np.array(var_x_e_list),
        'X_train_p': Xp_tr_last, 'X_test_p': Xp_te_last,
        'X_train_e': Xe_tr_last, 'X_test_e': Xe_te_last,
    }
    
# ── Experiment config ────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument('--family-gates', type=str, required=True)
parser.add_argument('--num-gates', type=int, default=100)
parser.add_argument('--num-experiments', type=int, default=100)
parser.add_argument('--degree',          type=float, default=None)
parser.add_argument('--K',               type=int,   default=1, dest='scrambler_depth')
parser.add_argument('--L',               type=int,   default=1, dest='n_layers')
args = parser.parse_args()

family_gates = args.family_gates
num_gates = args.num_gates
ns = args.num_experiments
degree          = args.degree
scrambler_depth = args.scrambler_depth
n_layers        = args.n_layers

if num_gates==-1:
    num_gates = None
if degree==-1:
    degree = None
if scrambler_depth==-1:
    scrambler_depth = None
if n_layers==-1:
    n_layers = None
    
t_dismiss = 100
nqbits = 3
k_exp  = 3
nv = 1
n_shadows = 10000
nshots = 10000
K_ors = 4
n_obs_list  = np.arange(10, 1000, 50) 
alpha = 1e-10

RESULTS_FOLDER = "/data/cvcqml/common/laia/time_series/results/NARMA/"
os.makedirs(RESULTS_FOLDER, exist_ok=True)

data = generate_narma_fourier_task(T=5000, n_train=4000, narma_order=2, n_a=3)
u_train, y_train = data['u_train'], data['y_train']
u_test, y_test = data['u_test'], data['y_test']
t_train = np.arange(len(y_train))
t_test = np.arange(len(y_train), len(y_train) + len(y_test))


path = _qrc_path(RESULTS_FOLDER, family_gates, nqbits, ns, num_gates=num_gates, degree= degree, scrambler_depth = scrambler_depth, n_layers = n_layers)

res = run_narma_experiment(family_gates, data, n_obs_list, ns=ns, nqbits=nqbits, k_exp=k_exp, n_measure=nqbits + k_exp, alpha=alpha, nshots_ors=nshots, K_ors=K_ors, n_shadows=n_shadows, num_gates=num_gates, degree= degree,  scrambler_depth = scrambler_depth, n_layers = n_layers)

_save_qrc(path, res)

print(f"  MSE Pauli= {np.mean(res['mse_p']):.8f} ± {np.std(res['mse_p']):.8f}, "
      f"MSE Exp= {np.mean(res['mse_e']):.8f} ± {np.std(res['mse_e']):.8f}, "
      f"ORS = {np.mean(res['ors']):.4f} ± {np.std(res['ors']):.4f}, "
      f"ΔS = {np.mean(res['delta_s']):.4f} ± {np.std(res['delta_s']):.4f}, "
      f"Var_p = {np.mean(res['var_x_p']):.4f} ± {np.std(res['var_x_p']):.4f}, "
      f"Var_e = {np.mean(res['var_x_e']):.4f} ± {np.std(res['var_x_e']):.4f}")