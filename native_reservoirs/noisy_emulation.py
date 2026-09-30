"""Task performance under realistic noise (Exp4 §6).

Gate-based (IQM, AQT): the LiH and energy task circuits are exported as JSON
(:func:`export_task_circuits`), run on the vendors' noisy emulators by
``experiments/native_hardware/vendor_tasks.py`` (vendor environment, qiskit 1.x), and the counts
are turned back into 1-local Pauli features here (3 settings: X, Y or Z on every qubit).

Photonic (Lucy): the same tasks with Perceval's ``NoiseModel`` (partial distinguishability,
multi-photon emission, loss), post-selected on 3 detected photons. Ascella's published source
figures stand in for Lucy's.

Every platform and task is scored four ways (all with the §5 readout protocols):

* ``noise_free``      — exact features for training and test;
* ``twin``            — the noise-aware digital twin of §5 for training and test (its prediction);
* ``emulator_twin``   — readout trained on the twin, test inputs from the noisy emulator
                        (the inference-only hardware plan);
* ``emulator_calibrated`` — as ``emulator_twin``, but the twin's noise parameters are first fitted
                        to the measured test features against their simulated values (no target
                        labels, no extra circuits): per-feature scale and offset on the gate-based
                        machines; on Lucy the weight c of ideal vs fully distinguishable photons;
* ``emulator``        — readout trained and tested on noisy-emulator data.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .base import derive_seed
from .experiment import (Exp4Config, _gate_window_resources, _pauli1_from_samples, get_data,
                         make_reservoir, qrc_reservoir)

SHOTS = 10_000                                   # shots per setting (gate) / 3-photon events (Lucy)
REPS = 3                                         # shot-noise draws averaged (twin; Lucy also its emulator)
SETTINGS = ("X", "Y", "Z")
# native R(θ, φ) applied to every qubit before a Z measurement: RY(-π/2) → X, RX(π/2) → Y
BASIS_OPS = {"X": [["r", np.pi / 2, 3 * np.pi / 2]], "Y": [["r", np.pi / 2, 0.0]], "Z": []}
# Ascella (arXiv:2306.00874): M_s = 0.944, g2(0) = 7.3e-3, total efficiency ≈ 8 %
LUCY_NOISE = {"indistinguishability": 0.94, "g2": 0.0073, "transmittance": 0.08}
EMULATOR = {"iqm": "IQM fake Crystal-54 (sample noise model)", "aqt": "AQT offline noisy simulator",
            "quandela": "Perceval noise model (Ascella source figures)"}


def _vendor_dir(cfg: Exp4Config) -> Path:
    return cfg.path / "vendor"


def counts_path(cfg: Exp4Config, platform: str, task: str) -> Path:
    return _vendor_dir(cfg) / f"task_counts_{platform}_{task}.npz"


# ── features ──────────────────────────────────────────────────────────────────
def pauli1_exact(psi: np.ndarray, n: int) -> np.ndarray:
    """All 1-local Paulis [X_0..X_{n-1}, Y_0.., Z_0..] of output states (qiskit little-endian)."""
    return _pauli1_from_samples(psi, n, None, None)


def pauli1_from_counts(counts: np.ndarray, n: int) -> np.ndarray:
    """Same features from measured counts of shape (N, 3 settings, 2^n), settings in SETTINGS order."""
    bits = (np.arange(2 ** n)[:, None] >> np.arange(n)[None, :]) & 1
    p = counts / counts.sum(axis=-1, keepdims=True)
    return np.hstack([p[:, b] @ (1 - 2 * bits) for b in range(len(SETTINGS))])


def _twin(X: np.ndarray, f, S: int, rng, offset=0.0) -> np.ndarray:
    """Gate-based digital twin: depolarising factor f (global or per feature) and Gaussian shot noise (§5)."""
    X = f * X + offset
    return X + rng.normal(size=X.shape) * np.sqrt(np.clip(1 - X ** 2, 0, None) / S)


def calibrate(X_exact: np.ndarray, X_measured: np.ndarray):
    """Per-feature affine map measured ≈ a·exact + b, fitted by least squares on the same inputs."""
    a, b = np.empty(X_exact.shape[1]), np.empty(X_exact.shape[1])
    for k in range(X_exact.shape[1]):
        A = np.vstack([X_exact[:, k], np.ones(len(X_exact))]).T
        (a[k], b[k]), *_ = np.linalg.lstsq(A, X_measured[:, k], rcond=None)
    return a, b


def calibrate_mixture(X_ind: np.ndarray, X_dist: np.ndarray, X_measured: np.ndarray) -> float:
    """Photonic: weight c in measured ≈ c·(indistinguishable) + (1 − c)·(distinguishable), least squares."""
    D = (X_ind - X_dist).ravel()
    return float(np.clip(D @ (X_measured - X_dist).ravel() / (D @ D), 0.0, 1.0))


def _sample(P: np.ndarray, S: int, rng) -> np.ndarray:
    return np.array([rng.multinomial(S, np.clip(p, 0, None) / p.sum()) / S for p in P])


# ── inputs of each task ───────────────────────────────────────────────────────
def _lih_angles(cfg):
    from .adapters import weights_to_angles
    d = get_data(cfg)
    return weights_to_angles(d["w6"], cfg.lih_angle_map, d["w_range"])


class _WindowSource:
    """Feature provider for ``WindowedQRC``: features from a function of the windows (records them)."""

    def __init__(self, fn):
        self.fn, self.seen = fn, []

    def window_features(self, windows, **kw):
        windows = np.asarray(windows, dtype=float)
        self.seen.append(windows)
        return self.fn(windows)


def _energy_r2(cfg, fit_fn, test_fn, alphas=(1e-12, 1e-8, 1e-6, 1e-4, 1e-2)):
    """Window-2 QRC readout as in §5 (teacher forcing, α chosen on validation); R² on the full test set.

    ``fit_fn`` supplies the training / validation features, ``test_fn`` the test features.
    Returns (R², all windows requested)."""
    from . import tasks
    from .adapters import WindowedQRC
    d = get_data(cfg)["energy"]
    src = _WindowSource(fit_fn)
    qrc = WindowedQRC(src, window=cfg.qrc_window, alpha=cfg.alpha, t_dismiss=cfg.t_dismiss)
    best = None
    for a in alphas:
        qrc.alpha = a
        qrc.train(y=d["y_train_scaled"], scaler=d["scaler"])
        r2v = tasks.compute_metrics(d["y_val"], qrc.test(y=d["y_val_scaled"], scaler=d["scaler"], retrain=False))[3]
        best = max(best or (r2v, a), (r2v, a))
    qrc.alpha = best[1]
    qrc.train(y=d["y_train_scaled"], scaler=d["scaler"])
    qrc.test(y=d["y_val_scaled"], scaler=d["scaler"], retrain=False)
    src.fn = test_fn
    pred = qrc.test(y=d["y_test_scaled"], scaler=d["scaler"], retrain=False)
    return tasks.compute_metrics(d["y_test"], pred)[3], np.vstack(src.seen)


def energy_windows(cfg: Exp4Config) -> np.ndarray:
    """Unique input windows the energy readout requests (train, validation and test), in first-seen order."""
    _, seen = _energy_r2(cfg, lambda w: w, lambda w: w, alphas=(cfg.alpha,))
    _, idx = np.unique(np.round(seen, 12), axis=0, return_index=True)
    return seen[np.sort(idx)]


def _lookup(windows: np.ndarray, F: np.ndarray):
    """windows → features, by exact window match."""
    table = {tuple(np.round(w, 12)): i for i, w in enumerate(windows)}
    return lambda W: F[[table[tuple(np.round(w, 12))] for w in W]]


def _lih_rmse(F_fit, F_test, cfg):
    from . import tasks
    y = get_data(cfg)["lih"]["y"]
    tr, te = tasks.lih_split_indices()
    o = tasks.run_lih_readout(np.vstack([F_fit[tr], F_test[te]]), np.vstack([y[tr], y[te]]), alpha="cv",
                              test_size=len(te), random_state=None,
                              _split=(np.arange(len(tr)), len(tr) + np.arange(len(te))))
    return 1e3 * o["rmse"]


# ── export for the vendor emulators ──────────────────────────────────────────
def _ops(ops) -> list:
    return [[o["name"], o["qubits"], [float(x) for x in o["params"]]] for o in ops]


def export_task_circuits(cfg: Exp4Config, configs: dict, path: Path | None = None) -> Path:
    """Write the gate-based task circuits for ``vendor_tasks.py``.

    ``configs = {"lih": {platform: (level, n_layers)}, "energy": {...}}``. Each circuit is a list of
    segments — a data-encoding op list, ``"R"`` (the reservoir) or ``"P"`` (hidden qubits in |+⟩) —
    followed on the vendor side by the basis rotation of each setting and a final measurement.
    """
    path = path or (_vendor_dir(cfg) / "task_circuits.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    entries = []
    for task, per in configs.items():
        for platform, (level, n_layers) in per.items():
            if platform not in ("iqm", "aqt"):
                continue
            if task == "lih":
                res = make_reservoir(platform, level, 0, cfg)
                inputs = [[[["r", [j], [float(t), np.pi / 2]] for j, t in enumerate(th)], "R"] for th in _lih_angles(cfg)]
            else:
                res = qrc_reservoir(cfg, platform, level, n_layers)
                inputs = []
                for row in energy_windows(cfg):
                    segs = ["P"]                                # hidden qubits in |+⟩
                    for u in row:
                        segs += [[["r", [0], [np.pi / 2, float((np.pi / 2 - np.pi * u) % (2 * np.pi))]]], "R"]
                    inputs.append(segs)
            entries.append({"platform": platform, "task": task, "level": str(level), "n_layers": n_layers,
                            "n_qubits": res.n_qubits, "layout": list(res.layout),
                            "reservoir_ops": _ops(res.export_ops()),
                            "prep_ops": [["r", [h], [np.pi / 2, np.pi / 2]] for h in range(1, res.n_qubits)],
                            "inputs": inputs})
    path.write_text(json.dumps({"config_hash": cfg.hash(), "shots": SHOTS, "settings": list(SETTINGS),
                                "basis_ops": BASIS_OPS, "tasks": entries}))
    return path


# ── Perceval (Lucy) ───────────────────────────────────────────────────────────
def _perceval_chunk(cfg, task, level, items, noise_kw):
    import perceval as pcvl
    res = make_reservoir("quandela", level, 0, cfg)
    nm = pcvl.NoiseModel(**noise_kw)
    Vs = ([res.circuit_unitary([t]) for t in items] if task == "lih" else list(res.window_unitaries(items)))
    return np.array([res.noisy_probabilities(V, nm) for V in Vs])


def perceval_probabilities(cfg, task, level, items, noise_kw, n_jobs=None) -> np.ndarray:
    """Post-selected 3-photon distributions under a Perceval noise model, one per input."""
    from joblib import Parallel, delayed
    chunks = np.array_split(np.asarray(items), max(1, min(len(items) // 50, 64)))
    out = Parallel(n_jobs=n_jobs or cfg.n_jobs)(delayed(_perceval_chunk)(cfg, task, level, c, noise_kw)
                                                for c in chunks)
    return np.vstack(out)


def _ideal_probabilities(res, task, items):
    Vs = [res.circuit_unitary([t]) for t in items] if task == "lih" else res.window_unitaries(items)
    return np.array([res.probabilities(V) for V in Vs])


# ── scoring ───────────────────────────────────────────────────────────────────
def _score_lih(cfg, X, X_twin, X_emu, calibrated_twin):
    from . import tasks
    out = {"noise_free": _lih_rmse(X, X, cfg), "twin": _lih_rmse(X_twin, X_twin, cfg),
           "emulator_twin": np.nan, "emulator_calibrated": np.nan, "emulator": np.nan}
    if X_emu is not None:
        _, te = tasks.lih_split_indices()
        X_cal = calibrated_twin(te, X_emu)
        out.update(emulator_twin=_lih_rmse(X_twin, X_emu, cfg), emulator_calibrated=_lih_rmse(X_cal, X_emu, cfg),
                   emulator=_lih_rmse(X_emu, X_emu, cfg))
    return out


def _test_windows(cfg, W):
    """Rows of W that the energy readout uses as test inputs."""
    seen = {}

    def rec(windows):
        seen["last"] = windows
        return windows
    _energy_r2(cfg, lambda w: w, rec, alphas=(cfg.alpha,))
    table = {tuple(np.round(w, 12)): i for i, w in enumerate(W)}
    return np.array([table[tuple(np.round(w, 12))] for w in seen["last"]])


def _score_energy(cfg, W, X, X_twin, X_emu, calibrated_twin):
    ex, tw = _lookup(W, X), _lookup(W, X_twin)
    out = {"noise_free": _energy_r2(cfg, ex, ex)[0], "twin": _energy_r2(cfg, tw, tw)[0],
           "emulator_twin": np.nan, "emulator_calibrated": np.nan, "emulator": np.nan}
    if X_emu is not None:
        em = _lookup(W, X_emu)
        cal = _lookup(W, calibrated_twin(_test_windows(cfg, W), X_emu))
        out.update(emulator_twin=_energy_r2(cfg, tw, em)[0], emulator_calibrated=_energy_r2(cfg, cal, em)[0],
                   emulator=_energy_r2(cfg, em, em)[0])
    return out


def _mean(scores: list[dict]) -> dict:
    """Mean over shot-noise draws, with the spread as ``<key>_std``."""
    out = {}
    for k in scores[0]:
        v = np.array([d[k] for d in scores], dtype=float)
        out[k], out[f"{k}_std"] = float(np.mean(v)), float(np.std(v))
    return out


class _PhotonicTask:
    """Lucy task inputs with ideal and fully distinguishable output distributions, scored under a noise model."""

    def __init__(self, cfg, task, level, W=None):
        self.cfg, self.task, self.level, self.W = cfg, task, level, W
        self.res = make_reservoir("quandela", level, 0, cfg)
        self.items = _lih_angles(cfg) if task == "lih" else W
        self.P = _ideal_probabilities(self.res, task, self.items)
        self.P_dist = perceval_probabilities(cfg, task, level, self.items, {"indistinguishability": 0.0})
        self.X, self.X_dist = self.res.features_from_probs(self.P), self.res.features_from_probs(self.P_dist)

    def score(self, noise_kw: dict, S: int, rng, reps: int = REPS) -> dict:
        feats = self.res.features_from_probs
        Pn = perceval_probabilities(self.cfg, self.task, self.level, self.items, noise_kw)

        def calibrated_twin(rows, X_meas):
            c = calibrate_mixture(self.X[rows], self.X_dist[rows], X_meas[rows])
            return feats(_sample(c * self.P + (1 - c) * self.P_dist, S, rng))
        scores = []
        for _ in range(reps):
            X_twin, X_emu = feats(_sample(self.P, S, rng)), feats(_sample(Pn, S, rng))
            scores.append(_score_lih(self.cfg, self.X, X_twin, X_emu, calibrated_twin) if self.task == "lih"
                          else _score_energy(self.cfg, self.W, self.X, X_twin, X_emu, calibrated_twin))
        return _mean(scores)


def noisy_task_performance(cfg: Exp4Config, configs: dict, S: int = SHOTS, reps: int = REPS) -> pd.DataFrame:
    """LiH RMSE (mHa) and energy test R², noise-free / twin / noisy emulator, for every platform.

    Scores are means over ``reps`` shot-noise draws of the twin (the vendor-emulator counts are one
    run; Lucy's emulated events are redrawn too)."""
    from .adapters import GateWindowFeatures
    rows = []
    for task, per in configs.items():
        W = energy_windows(cfg) if task == "energy" else None
        for platform, (level, n_layers) in per.items():
            rng = np.random.default_rng(derive_seed(cfg.base_seed, "noisy_tasks", platform, task))
            if platform == "quandela":
                ph = _PhotonicTask(cfg, task, level, W)
                rows.append({"platform": platform, "task": task, "level": str(level), "n_layers": n_layers,
                             "shots": S, "f_est": np.nan, "emulator_name": EMULATOR[platform],
                             **ph.score(LUCY_NOISE, S, rng, reps)})
                continue
            n = cfg.n_sites
            if task == "lih":
                res = make_reservoir(platform, level, 0, cfg)
                th = _lih_angles(cfg)
                single = np.stack([np.cos(th / 2), np.sin(th / 2)], -1)
                psi = single[:, n - 1]
                for q in range(n - 2, -1, -1):
                    psi = np.einsum("ia,ib->iab", psi, single[:, q]).reshape(len(th), -1)
                psi = psi @ res.unitary().T
                f = res.resource_metrics()["f_est"]
            else:
                res = qrc_reservoir(cfg, platform, level, n_layers)
                psi = GateWindowFeatures(res.circuit, mode="reupload", observables_type=(1,)).window_states(W)
                f = _gate_window_resources(res, cfg.qrc_window, "reupload", "x")["x_f_est"]
            X = pauli1_exact(psi, n)

            def calibrated_twin(rows, X_meas, X=X, rng=rng):
                a, b = calibrate(X[rows], X_meas[rows])
                return _twin(X, a, S, rng, offset=b)
            cp = counts_path(cfg, platform, task)
            X_emu = pauli1_from_counts(np.load(cp)["counts"], n) if cp.exists() else None
            scores =[(_score_lih(cfg, X, _twin(X, f, S, rng), X_emu, calibrated_twin) if task == "lih"
                       else _score_energy(cfg, W, X, _twin(X, f, S, rng), X_emu, calibrated_twin))
                      for _ in range(reps)]
            rows.append({"platform": platform, "task": task, "level": str(level), "n_layers": n_layers,
                         "shots": S, "f_est": f, "emulator_name": EMULATOR[platform], **_mean(scores)})
    return pd.DataFrame(rows)


def perceval_noise_sweep(cfg: Exp4Config, configs: dict, S: int = SHOTS, reps: int = REPS,
                         indistinguishabilities=(1.0, 0.98, 0.96, 0.94, 0.92, 0.9, 0.85, 0.8)) -> pd.DataFrame:
    """Lucy: the §6 scores vs photon indistinguishability (g2 and loss at Ascella's values), S events per input."""
    rows = []
    for task in ("lih", "energy"):
        level = configs[task]["quandela"][0]
        ph = _PhotonicTask(cfg, task, level, energy_windows(cfg) if task == "energy" else None)
        for ind in indistinguishabilities:
            rng = np.random.default_rng(derive_seed(cfg.base_seed, "perceval_sweep", task, int(round(1e4 * ind))))
            rows.append({"task": task, "level": str(level), "indistinguishability": ind, "shots": S,
                         "g2": LUCY_NOISE["g2"], "transmittance": LUCY_NOISE["transmittance"],
                         **ph.score({**LUCY_NOISE, "indistinguishability": ind}, S, rng, reps)})
    return pd.DataFrame(rows)
