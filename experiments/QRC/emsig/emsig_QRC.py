import os
# ── CPU throttling — must be set BEFORE numpy/scipy/qiskit are imported ──────
os.environ["OMP_NUM_THREADS"]        = "1"
os.environ["MKL_NUM_THREADS"]        = "1"
os.environ["OPENBLAS_NUM_THREADS"]   = "1"
os.environ["NUMEXPR_NUM_THREADS"]    = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
# ─────────────────────────────────────────────────────────────────────────────

import argparse
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import MinMaxScaler

from QRC import QuantumRC_Circuit
from metrics import (
    lorentz_curves,
    order_statistics_score,
)

# ─────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────
def effective_rank(F, eps=1e-12):
    """Participation ratio of singular values of the centred feature matrix."""
    F = F - F.mean(axis=0, keepdims=True)
    try:
        sv = np.linalg.svd(F, compute_uv=False)
    except np.linalg.LinAlgError:
        from scipy.linalg import svd
        sv = svd(F, compute_uv=False, lapack_driver='gesvd')
    sv = sv[sv > eps]
    if len(sv) == 0:
        return 0.0
    return float(sv.sum() ** 2 / (sv ** 2).sum())


# ─────────────────────────────────────────────────────────────────────
# EMSIG loader (Brücke et al. 2024)
# ─────────────────────────────────────────────────────────────────────
TARGET = "_sum_consumption_active_power"

def load_ems_series(csv_path,
                    ems_id="ems_3",
                    start="2019-01-01",
                    end="2020-12-31 23:59:59",
                    resample="15min"):
    df = pd.read_csv(csv_path, usecols=["ems_id", "time", TARGET])
    df = df[df["ems_id"] == ems_id].copy()
    df["time"] = pd.to_datetime(df["time"], unit="ms", utc=True)
    df[TARGET] = pd.to_numeric(df[TARGET], errors="coerce")

    df = (df.sort_values("time")
            .set_index("time")[[TARGET]]
            .loc[start:end])

    # Regular 15-min grid, interpolate small gaps (≤4 h)
    idx = pd.date_range(start, end, freq="15min", tz="UTC")
    df  = df.reindex(idx)
    df[TARGET] = df[TARGET].interpolate(method="time", limit=16)
    df = df.dropna()

    if resample is not None and resample != "15min":
        df = df.resample(resample).mean().dropna()

    return (df[TARGET].values / 1000.0).astype(float)        # W → kW


def compute_metrics(y_true, y_pred):
    """Return [MAE, RMSE, NRMSE, R2]."""
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    err    = y_true - y_pred
    mae    = float(np.mean(np.abs(err)))
    rmse   = float(np.sqrt(np.mean(err ** 2)))
    rng    = y_true.max() - y_true.min()
    nrmse  = float(rmse / rng) if rng > 0 else np.nan
    ss_res = float(np.sum(err ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    r2     = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
    return [mae, rmse, nrmse, r2]
    
# ─────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument('--family-gates',    type=str,   required=True)
parser.add_argument('--num-gates',       type=int,   default=100)
parser.add_argument('--num-experiments', type=int,   default=100)
parser.add_argument('--degree',          type=float, default=None)
parser.add_argument('--K',               type=int,   default=1, dest='scrambler_depth')
parser.add_argument('--L',               type=int,   default=1, dest='n_layers')
parser.add_argument('--ems',             type=str,   default='ems_3')
parser.add_argument('--N',               type=int,   default=5000,
                    help='Number of contiguous 15-min samples to use')
parser.add_argument('--resample',        type=str,   default='15min',
                    help='"15min" (paper) or e.g. "1h"')
args = parser.parse_args()

family_gates    = args.family_gates
num_gates       = args.num_gates
ns              = args.num_experiments
degree          = args.degree
scrambler_depth = args.scrambler_depth
n_layers        = args.n_layers
ems_id          = args.ems
N               = args.N
resample        = args.resample

# ── Fixed experiment config (matches weather script style) ───────────────────
t_dismiss   = 100
nqbits      = 3
k_exp       = 1
encoding    = 'pauli'
n_measure   = 4
Nv          = 1
theta_scale = np.pi
nshots_ors  = 2000
K_ors       = 4

DATA_FOLDER    = "/data/cvcqml/common/laia/time_series/"
RESULTS_FOLDER = "/data/cvcqml/common/laia/time_series/results/emsig/"
os.makedirs(RESULTS_FOLDER, exist_ok=True)
CSV_PATH = os.path.join(DATA_FOLDER, "emsig_energy_data_by_ems.csv")


# ─────────────────────────────────────────────────────────────────────
# 1. Load + prepare data
# ─────────────────────────────────────────────────────────────────────
series_full = load_ems_series(CSV_PATH, ems_id=ems_id, resample=resample)
print(f"EMS {ems_id} full series ({resample}): {len(series_full)} samples")

series = series_full[:N]

n_train = int(0.70 * N)
n_val   = int(0.15 * N)
y_train = series[:n_train]
y_val   = series[n_train : n_train + n_val]
y_test  = series[n_train + n_val :]
print(f"train/val/test sizes: {len(y_train)}/{len(y_val)}/{len(y_test)}")

scaler_y = MinMaxScaler(feature_range=(0, 1))
y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1))
y_val_scaled   = scaler_y.transform   (y_val  .reshape(-1, 1))
y_test_scaled  = scaler_y.transform   (y_test .reshape(-1, 1))
y_val_scaled   = np.clip(y_val_scaled,  0, 1)
y_test_scaled  = np.clip(y_test_scaled, 0, 1)

X_train_scaled = None
X_val_scaled   = None
X_test_scaled  = None


# ─────────────────────────────────────────────────────────────────────
# 2. Gate config
# ─────────────────────────────────────────────────────────────────────
if family_gates == 'G1':
    gates_set, alpha = ['CNOT', 'X', 'H'], 1e-6
elif family_gates == 'G2':
    gates_set, alpha = ['CNOT', 'S', 'H'], 1e-6
elif family_gates == 'G3':
    gates_set, alpha = ['CNOT', 'T', 'H'], 1e-12
elif family_gates == 'Ising':
    gates_set, alpha = 'Ising', 1e-12
else:
    gates_set, alpha = family_gates, 1e-12


# ─────────────────────────────────────────────────────────────────────
# 3. Run experiments
# ─────────────────────────────────────────────────────────────────────
all_y_train_pred = []
all_y_val_pred   = []
all_y_test_pred  = []
all_metrics      = []
ors_list         = []
var_x_list       = []
r_eff_list       = []
all_x_train_list = []

for i in range(ns):
    qrc = QuantumRC_Circuit(
        gates_set=gates_set, nqbits=nqbits, k_exp=k_exp, encoding=encoding,
        dim_y=1, input_mode='autoregressive', theta_scale=theta_scale,
        alpha=alpha, Nv=Nv, t_dismiss=t_dismiss, n_measure=n_measure,
        observables_type=[1,2], degree=degree,
        num_gates=num_gates, n_layers=n_layers, scrambler_depth=scrambler_depth,
    )

    y_train_pred = qrc.train(y=y_train_scaled, scaler=scaler_y,
                             X_train_scaled=X_train_scaled)
    y_val_pred   = qrc.test(y=y_val_scaled,   scaler=scaler_y,
                             X_test_scaled=X_val_scaled,  retrain=True)
    y_test_pred  = qrc.test(y=y_test_scaled,  scaler=scaler_y,
                             X_test_scaled=X_test_scaled, retrain=True)

    m_train = compute_metrics(y_train[t_dismiss:], y_train_pred)
    m_val   = compute_metrics(y_val,  y_val_pred)
    m_test  = compute_metrics(y_test, y_test_pred)
    all_metrics.append(m_train + m_val + m_test)   # 12 values per run

    all_y_train_pred.append(y_train_pred)
    all_y_val_pred.append(y_val_pred)
    all_y_test_pred.append(y_test_pred)

    # Complexity metrics (reservoir-only)
    _, sp, _ = lorentz_curves(qrc.get_reservoir, qrc.n_total,
                              nshots_ors, 1, verbose=False)
    ors_list  .append(order_statistics_score(sp, D=2**qrc.n_total, K=K_ors))
    var_x_list.append(np.std(qrc.X_train, axis=0))
    r_eff_list.append(effective_rank(qrc.X_train))
    all_x_train_list.append(qrc.X_train)

    print(f"[{i+1}/{ns}] "
          f"Test  MAE={m_test[0]:.4f}  RMSE={m_test[1]:.4f}  "
          f"NRMSE={m_test[2]:.4f}  R²={m_test[3]:.3f} | "
          f"ORS: {np.mean(ors_list):.4f}  Var_x: {np.mean(var_x_list):.4f}  "
          f"R_eff: {np.mean(r_eff_list):.2f}")



# ─────────────────────────────────────────────────────────────────────
# 4. Save
# ─────────────────────────────────────────────────────────────────────
metrics_array = np.array(all_metrics)        # (ns, 3)

fname = (f"results_{ems_id}_{family_gates}_{num_gates}"
         f"_{degree}_{scrambler_depth}_{n_layers}.npz")
path  = os.path.join(RESULTS_FOLDER, fname)

metrics_cols = np.array([
    'train_mae', 'train_rmse', 'train_nrmse', 'train_r2',
    'val_mae',   'val_rmse',   'val_nrmse',   'val_r2',
    'test_mae',  'test_rmse',  'test_nrmse',  'test_r2',
])

np.savez_compressed(
    path,
    y_train_pred = np.array(all_y_train_pred),
    y_val_pred   = np.array(all_y_val_pred),
    y_test_pred  = np.array(all_y_test_pred),
    X_train      = np.array(all_x_train_list),
    var_x        = np.array(var_x_list),
    ors          = np.array(ors_list),
    r_eff        = np.array(r_eff_list),
    metrics      = np.array(all_metrics),          # (ns, 12)
    metrics_cols = metrics_cols,
)
print(f"\nSaved all results → {path}")