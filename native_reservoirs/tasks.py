"""Real-data tasks for Exp4, with the Exp3 readout protocols.

LiH (QELM)
    Targets, split, scaling, readout and metric are exactly those of Exp3 (cell 5):
    y = [E1−E0, E2−E0] (Ha), ``train_test_split(test_size=0.33, random_state=42)``,
    ``MinMaxScaler`` on y, ``Ridge(alpha=1e-12)``, MSE in Ha².

    Input representation (Exp4): the symmetry-reduced *configuration weights* of the ground
    state instead of the full 256-amplitude state (not loadable natively on any of the three
    QPUs):

    * particle-number conservation (2 electrons) → only Hamming-weight-2 strings; 10 of the
      28 are populated: 3, 12, 18, 33, 48, 66, 96, 129, 144, 192;
    * singlet spin symmetry: a18 = −a33, a66 = −a129, a96 = −a144 (exact) →
      7 configuration weights W = (P3, P12, P48, P192, P18+P33, P66+P129, P96+P144), ΣW = 1;
    * dropping the dominant closed-shell weight P192 (fixed by normalisation) leaves
      **6 independent weights** w = (P3, P12, P48, P18+P33, P66+P129, P96+P144);
    * weights |a|² are used rather than amplitudes: the global sign of every stored state is
      arbitrary and the open-shell sector flips sign at R ≈ 2.49 Å.

    The circuit therefore receives classical numbers derived from the state, not the state.

Energy (QRC)
    Exactly Exp3's series: ``data/time_series/emsig_energy_data_by_ems.csv`` (gitignored, large),
    EMS ``ems_3``, 2019-01-01 … 2020-12-31 on a 15-min grid, time interpolation of gaps up to
    16 steps (4 h), dropna, kW (``load_ems_series``, copied from
    ``experiments/QRC/emsig/emsig_QRC.py``); first N = 5000 samples, contiguous 70/15/15 split,
    MinMax on train, clip val / test to [0, 1]; autoregressive horizon-1 readout with washout
    100, Ridge 1e-12, teacher forcing and online retraining (``retrain=True``). This reproduces
    Exp3's stored EMSIG results exactly. ``load_energy_series`` (hourly aggregate file) is kept
    only as a fallback.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from .hardware_specs import REPO_ROOT

DATA_DIR = REPO_ROOT / "data"

# ── LiH ───────────────────────────────────────────────────────────────────────
LIH_SUPPORT = (3, 12, 18, 33, 48, 66, 96, 129, 144, 192)
LIH_SINGLET_PAIRS = ((18, 33), (66, 129), (96, 144))
LIH_CLOSED_SHELL = (3, 12, 48)
LIH_DOMINANT = 192
LIH_WEIGHT_NAMES = ("P3", "P12", "P48", "P18+P33", "P66+P129", "P96+P144")


def load_lih(data_dir: Path | str = DATA_DIR / "LiH") -> dict:
    """Load the LiH dataset used in Exp3 and its targets y = [E1-E0, E2-E0]."""
    d = Path(data_dir)
    spectra = np.load(d / "spectrums_LiH.npy")
    gs = np.load(d / "ground_states_LiH.npy")
    bl = np.load(d / "bond_lengths_LiH.npy")
    y = np.stack([spectra[:, 1] - spectra[:, 0], spectra[:, 2] - spectra[:, 0]], axis=1)
    return {"bond_lengths": bl, "spectra": spectra, "ground_states": gs, "y": y}


def lih_configuration_weights(ground_states, check: bool = True, tol: float = 1e-10) -> dict:
    """Symmetry-reduced configuration weights (see module docstring).

    Returns dict with ``w6`` (N, 6) — the encoded features —, ``W7`` (N, 7) and a small
    report of the symmetry checks.
    """
    gs = np.asarray(ground_states)
    n_q = int(np.log2(gs.shape[1]))
    hw2 = [i for i in range(gs.shape[1]) if bin(i).count("1") == 2]
    outside = np.delete(np.abs(gs), list(LIH_SUPPORT), axis=1).max()
    pair_err = max(np.abs(gs[:, a] + gs[:, b]).max() for a, b in LIH_SINGLET_PAIRS)
    P = np.abs(gs) ** 2
    W7 = np.stack([P[:, 3], P[:, 12], P[:, 48], P[:, 192],
                   P[:, 18] + P[:, 33], P[:, 66] + P[:, 129], P[:, 96] + P[:, 144]], axis=1)
    norm_err = np.abs(W7.sum(axis=1) - 1).max()
    report = {"n_qubits": n_q, "hamming_weight_2_strings": len(hw2),
              "populated_strings": len(LIH_SUPPORT), "max_amplitude_outside_support": float(outside),
              "max_singlet_pair_violation": float(pair_err), "max_normalisation_error": float(norm_err),
              "min_dominant_weight": float(W7[:, 3].min())}
    if check:
        assert set(LIH_SUPPORT) <= set(hw2), "support is not particle-number conserving"
        assert outside < tol, f"amplitude outside the 10-string support: {outside}"
        assert pair_err < tol, f"singlet symmetry violated: {pair_err}"
        assert norm_err < tol, f"weights do not sum to 1: {norm_err}"
    w6 = W7[:, [0, 1, 2, 4, 5, 6]]
    return {"w6": w6, "W7": W7, "names": LIH_WEIGHT_NAMES, "report": report}


def lih_split_indices(n: int = 300, test_size: float = 0.33, random_state: int = 42):
    """Train / test indices identical to Exp3's ``train_test_split`` call."""
    idx = np.arange(n)
    tr, te = train_test_split(idx, test_size=test_size, random_state=random_state)
    return tr, te


def lih_train_subset(n_train: int | None, bond_lengths=None, variant: int = 0, n: int = 300) -> np.ndarray:
    """``n_train`` training geometries on an (optionally jittered) bond-length grid.

    Chosen among Exp3's 201 training indices as the points closest to a grid spanning the
    training bond-length range; ``variant > 0`` jitters every node by up to ±half a spacing
    (deterministic in ``(n_train, variant)``). The test set is unchanged.
    """
    tr, _ = lih_split_indices(n)
    if n_train is None or n_train >= len(tr):
        return tr
    if bond_lengths is None:
        bond_lengths = np.load(DATA_DIR / "LiH" / "bond_lengths_LiH.npy")
    R = np.asarray(bond_lengths)[tr]
    lo, hi = R.min(), R.max()
    grid = np.linspace(lo, hi, n_train)
    if variant:
        h = (hi - lo) / max(n_train - 1, 1)
        grid = np.clip(grid + np.random.default_rng(1000 * n_train + variant).uniform(-h / 2, h / 2, n_train), lo, hi)
    chosen = []
    for g in grid:                                   # nearest unused training point per grid node
        order = np.argsort(np.abs(R - g))
        chosen.append(next(tr[i] for i in order if tr[i] not in chosen))
    return np.sort(np.array(chosen))


CHEMICAL_ACCURACY_HA = 1.594e-3          # 1 kcal/mol


def run_lih_readout(F, y, alpha=1e-12, test_size: float = 0.33, random_state: int = 42,
                    train_idx=None, bond_lengths=None, _split=None) -> dict:
    """Exp3 LiH readout: split → MinMax(y_tr) → Ridge → inverse transform → MSE (Ha²).

    ``alpha='cv'`` selects the ridge strength by efficient leave-one-out CV on the *training*
    set only (``RidgeCV``, α ∈ [1e-12, 1]).
    ``train_idx`` restricts training to a subset of Exp3's training indices (learning curves);
    the test set is Exp3's 99 test points — restricted, if ``bond_lengths`` is given, to those
    inside the bond-length range of the training subset (interpolation only). Also returns RMSE
    / max error (Ha), the fraction of test points within chemical accuracy (1.6 mHa) and the
    number of test points used.
    """
    if _split is not None:                          # explicit (train rows, test rows)
        i_tr, i_te = _split
        X_tr, X_te, y_tr, y_te = F[i_tr], F[i_te], y[i_tr], y[i_te]
    else:
        X_tr, X_te, y_tr, y_te, i_tr, i_te = train_test_split(
            F, y, np.arange(len(y)), test_size=test_size, random_state=random_state)
    if train_idx is not None:
        keep = np.isin(i_tr, train_idx)
        X_tr, y_tr = X_tr[keep], y_tr[keep]
        if bond_lengths is not None:
            R = np.asarray(bond_lengths)
            lo, hi = R[train_idx].min(), R[train_idx].max()
            inside = (R[i_te] >= lo) & (R[i_te] <= hi)
            X_te, y_te, i_te = X_te[inside], y_te[inside], i_te[inside]
    scaler = MinMaxScaler()
    if alpha == "cv":
        from sklearn.linear_model import RidgeCV
        model = RidgeCV(alphas=np.logspace(-12, 0, 25)).fit(X_tr, scaler.fit_transform(y_tr))
    else:
        model = Ridge(alpha=alpha).fit(X_tr, scaler.fit_transform(y_tr))
    y_hat = scaler.inverse_transform(model.predict(X_te))
    err = np.abs(y_hat - y_te)
    return {"mse": float(np.mean(err ** 2)), "rmse": float(np.sqrt(np.mean(err ** 2))),
            "max_err": float(err.max()), "frac_chem": float(np.mean(err < CHEMICAL_ACCURACY_HA)),
            "n_test": int(len(y_te)), "y_hat": y_hat, "y_test": y_te, "idx_test": i_te}


# ── Energy ────────────────────────────────────────────────────────────────────
ENERGY_CSV = DATA_DIR / "time_series" / "emsig_energy_data_hourly.csv"
EMS_CSV = DATA_DIR / "time_series" / "emsig_energy_data_by_ems.csv"
ENERGY_TARGET = "_sum_consumption_active_power"
_EMS_CACHE = {}


def load_ems_series(csv_path: Path | str = EMS_CSV, ems_id: str = "ems_3", start: str = "2019-01-01",
                    end: str = "2020-12-31 23:59:59", resample: str = "15min") -> np.ndarray:
    """Per-EMS consumption in kW — copy of ``experiments/QRC/emsig/emsig_QRC.py::load_ems_series``."""
    key = (str(csv_path), ems_id, start, end, resample)
    if key in _EMS_CACHE:
        return _EMS_CACHE[key]
    if not Path(csv_path).exists():
        raise FileNotFoundError(f"{csv_path} not found (large, gitignored EMSIG export used by Exp3)")
    df = pd.read_csv(csv_path, usecols=["ems_id", "time", ENERGY_TARGET])
    df = df[df["ems_id"] == ems_id].copy()
    df["time"] = pd.to_datetime(df["time"], unit="ms", utc=True)
    df[ENERGY_TARGET] = pd.to_numeric(df[ENERGY_TARGET], errors="coerce")
    df = (df.sort_values("time").set_index("time")[[ENERGY_TARGET]].loc[start:end])
    idx = pd.date_range(start, end, freq="15min", tz="UTC")
    df = df.reindex(idx)
    df[ENERGY_TARGET] = df[ENERGY_TARGET].interpolate(method="time", limit=16)
    df = df.dropna()
    if resample is not None and resample != "15min":
        df = df.resample(resample).mean().dropna()
    out = (df[ENERGY_TARGET].values / 1000.0).astype(float)
    _EMS_CACHE[key] = out
    return out


def load_energy_series(csv_path: Path | str = ENERGY_CSV, start: str = "2019-01-01",
                       end: str = "2020-12-31 23:59:59", interp_limit: int = 4) -> np.ndarray:
    """Hourly consumption in kW, following ``emsig_QRC.load_ems_series`` (hourly grid)."""
    df = pd.read_csv(csv_path)
    df["time"] = pd.to_datetime(df["time"], utc=True)
    s = pd.to_numeric(df.set_index("time").sort_index()[ENERGY_TARGET], errors="coerce")
    s = s.loc[start:end]
    idx = pd.date_range(start, end, freq="1h", tz="UTC")
    s = s.reindex(idx).interpolate(method="time", limit=interp_limit).dropna()
    return (s.values / 1000.0).astype(float)


def energy_splits(series, N: int = 5000, frac=(0.70, 0.15)) -> dict:
    """Contiguous train / val / test split and MinMax scaling as in ``emsig_QRC.py``."""
    series = np.asarray(series)[:N]
    n_train, n_val = int(frac[0] * N), int(frac[1] * N)
    y_train, y_val, y_test = series[:n_train], series[n_train:n_train + n_val], series[n_train + n_val:]
    scaler = MinMaxScaler(feature_range=(0, 1))
    return {
        "y_train": y_train, "y_val": y_val, "y_test": y_test, "scaler": scaler,
        "y_train_scaled": scaler.fit_transform(y_train.reshape(-1, 1)),
        "y_val_scaled": np.clip(scaler.transform(y_val.reshape(-1, 1)), 0, 1),
        "y_test_scaled": np.clip(scaler.transform(y_test.reshape(-1, 1)), 0, 1),
    }


def compute_metrics(y_true, y_pred):
    """Return [MAE, RMSE, NRMSE, R2] (copied from experiments/QRC/emsig/emsig_QRC.py)."""
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


METRIC_NAMES = ["mae", "rmse", "nrmse", "r2"]


def run_energy_qrc(qrc, data: dict, t_dismiss: int = 100, retrain: bool = True) -> dict:
    """Train / val / test flow of ``experiments/QRC/emsig/emsig_QRC.py`` for any QRC object.

    Works with ``NativeQRC`` and ``PhotonicQRC`` (both inherit ``QuantumRC_Circuit.train/test``).
    """
    from .diagnostics import effective_rank
    sc = data["scaler"]
    y_train_pred = qrc.train(y=data["y_train_scaled"], scaler=sc, X_train_scaled=None)
    X_train_only = qrc.X_train.copy()
    y_val_pred = qrc.test(y=data["y_val_scaled"], scaler=sc, X_test_scaled=None, retrain=retrain)
    y_test_pred = qrc.test(y=data["y_test_scaled"], scaler=sc, X_test_scaled=None, retrain=retrain)
    out = {}
    for split, yt, yp in (("train", data["y_train"][t_dismiss:], y_train_pred),
                          ("val", data["y_val"], y_val_pred), ("test", data["y_test"], y_test_pred)):
        for name, v in zip(METRIC_NAMES, compute_metrics(yt, yp)):
            out[f"{split}_{name}"] = v
    out["reff"] = effective_rank(qrc.X_train)                 # Exp3 convention (after retrain)
    out["reff_train_only"] = effective_rank(X_train_only)
    out["n_features"] = int(qrc.X_train.shape[1])
    out["y_test_pred"] = np.asarray(y_test_pred).ravel()
    return out
