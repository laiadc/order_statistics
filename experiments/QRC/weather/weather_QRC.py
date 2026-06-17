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

from technical_analysis_indicators import get_regressor_vars
from QRC import QuantumRC_Circuit
from metrics import (
    lorentz_curves,
    order_statistics_score,
    renyi2_entropy
)

import argparse

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
t_dismiss = 100
nqbits = 5
k_exp = 1
encoding = 'pauli'
n_measure = 3
Nv = 2
nshots_ors = 2000
K_ors = 4

DATA_FOLDER    = "/data/cvcqml/common/laia/time_series/"
RESULTS_FOLDER = "/data/cvcqml/common/laia/time_series/results/weather/"
os.makedirs(RESULTS_FOLDER, exist_ok=True)

# 1. READ DATA
data_train = pd.read_csv(DATA_FOLDER + 'weather_train.csv')
data_test = pd.read_csv(DATA_FOLDER + 'weather_test.csv')
data = pd.concat([data_train, data_test], ignore_index=True)

y = data[['meantemp']]
cols_pred = y.shape[1]
cols = ['humidity','wind_speed', 'meanpressure']
X = data[cols].shift(1).bfill()

data.date = pd.to_datetime(data.date)
dateIn_val = np.datetime64("2015-09-28")
dateFin_val = np.datetime64("2016-07-24")

X_train = X[data.date<=dateIn_val]
X_val = X[(data.date>dateIn_val) & (data.date<dateFin_val)]
X_test = X[data.date>dateFin_val]

y_train = y[data.date<=dateIn_val].values
y_val = y[(data.date>dateIn_val) & (data.date<dateFin_val)].values
y_test = y[data.date>dateFin_val].values

date_train = data.date[data.date<=dateIn_val]
date_val = data.date[(data.date>dateIn_val) & (data.date<dateFin_val)]
date_test = data.date[data.date>dateFin_val]

scaler_x = MinMaxScaler(feature_range=(-1,1))

# Fit scaler with train data
X_train_scaled = scaler_x.fit_transform(X_train)
# Transform validation data
X_val_scaled = scaler_x.transform(X_val)
X_test_scaled = scaler_x.transform(X_test)

scaler_y = MinMaxScaler(feature_range=(0,1))
# Fit scaler with train data
y_train_scaled = scaler_y.fit_transform(np.array(y_train).reshape(-1,cols_pred))

# Transform validation data
y_val_scaled = scaler_y.transform(np.array(y_val).reshape(-1,cols_pred))
y_test_scaled = scaler_y.transform(np.array(y_test).reshape(-1,cols_pred))
y_val_scaled = np.clip(y_val_scaled, 0, 1)
y_test_scaled = np.clip(y_test_scaled, 0, 1)

# 3. GATE CONFIG
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

# 4. RUN EXPERIMENTS — accumulate results across all ns runs
all_y_train_pred = []
all_y_val_pred   = []
all_y_test_pred  = []
all_metrics      = []   # one dict per run
ors_list = []
var_x_list = []
all_x_train_list = []

for i in range(ns):
    qrc = QuantumRC_Circuit(gates_set=gates_set, nqbits=nqbits, k_exp = k_exp, encoding=encoding, dim_y=1, input_mode = 'autoregressive',
                            alpha=alpha, Nv=Nv, t_dismiss=t_dismiss, n_measure=n_measure,  observables_type = [1], degree=degree,
                            num_gates=num_gates, n_layers=n_layers, scrambler_depth=scrambler_depth)
    
    y_train_pred = qrc.train(y = y_train_scaled, scaler = scaler_y,
                              X_train_scaled=X_train_scaled) 
    y_val_pred   = qrc.test(y = y_val_scaled, scaler = scaler_y,
                             X_test_scaled=X_val_scaled,  retrain=True)
    y_test_pred  = qrc.test(y = y_test_scaled, scaler = scaler_y,
                             X_test_scaled=X_test_scaled, retrain=True)

    train_mae = mean_absolute_error(y_train[t_dismiss:,], y_train_pred)
    val_mae   = mean_absolute_error(y_val,   y_val_pred)
    test_mae  = mean_absolute_error(y_test,  y_test_pred)

    all_y_train_pred.append(y_train_pred)
    all_y_val_pred.append(y_val_pred)
    all_y_test_pred.append(y_test_pred)
    all_metrics.append([train_mae, val_mae, test_mae]) 

    # Complexity metrics (reservoir-only)
    _, sp, _ = lorentz_curves(qrc.get_reservoir, qrc.n_total, nshots_ors, 1, verbose=False)
    ors_list.append(order_statistics_score(sp, D=2**qrc.n_total, K=K_ors))
    all_x_train_list.append(qrc.X_train)
    var_x_list.append(np.std(qrc.X_train, axis=0))
    
    print(f"[{i+1}/{ns}] Train MAE: {train_mae:.4f}  Val MAE: {val_mae:.4f}  Test MAE: {test_mae:.4f} | "
          f"ORS: {np.mean(ors_list):.4f}  Var_x: {np.mean(var_x_list):.4f}  " )

# 5. SAVE — single compressed file
# shapes: (ns, len_split, ...) for predictions; (ns, 6) for metrics
metrics_array = np.array(all_metrics)   # (ns, 6): mae_train/val/test, acc_train/val/test

path = RESULTS_FOLDER +  f'results_{family_gates}_{num_gates}_{degree}_{scrambler_depth}_{n_layers}.npz'
np.savez_compressed(
    path,
    y_train_pred = np.array(all_y_train_pred),  # (ns, n_train, ...)
    y_val_pred   = np.array(all_y_val_pred),    # (ns, n_val,   ...)
    y_test_pred  = np.array(all_y_test_pred),   # (ns, n_test,  ...)
    X_train = np.array(all_x_train_list),
    var_x = np.array(var_x_list),
    ors = np.array(ors_list),
    metrics      = metrics_array,               # (ns, 3)
    metrics_cols = np.array(['train_mae', 'val_mae', 'test_mae']),
)
print(f"\nSaved all results → {path}")
