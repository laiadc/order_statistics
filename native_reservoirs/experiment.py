"""Exp4 experiment runner: one reservoir instance → ORS + LiH QELM + energy QRC + resources.

Every native reservoir instance is identified by (platform, level, instance) and gets its own
seed ``derive_seed(base_seed, platform, level, instance)``. The same instance is used for the
task-independent diagnostic (ORS / ORS_LO on the bare reservoir) and for both tasks, so the
three quantities are measured on literally the same reservoir.

Results (``results/native_hardware/``):
    records.csv     one row per (platform, level, instance): seed, tasks, R_eff, resources
    ors.csv         one row per (platform, level): ORS / ORS_LO cell gap ± SEM
    ors_instances.csv  per-instance ORS gaps
    metadata.json   configuration, seeding scheme, package versions, git SHA, timestamps
"""

from __future__ import annotations

import dataclasses
import datetime as _dt
import hashlib
import importlib.metadata as md
import json
import platform as _platform
import subprocess
import time
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np
import pandas as pd

from .base import GateNativeReservoir, derive_seed
from .hardware_specs import REPO_ROOT


# ── Configuration ─────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class Exp4Config:
    base_seed: int = 2026
    n_sites: int = 6
    ns_task: int = 10
    ns_ors: int = 50
    K: int = 4
    B: int = 5
    iqm_layers: tuple = (1, 2, 4, 8, 16)
    aqt_densities: tuple = (0.25, 0.5, 0.75, 1.0)
    aqt_n_layers: int = 4
    aqt_alphas: tuple = ()                # supplementary power-law sweep (off by default)
    photonic_modes: int = 12
    photonic_photons: int = 3
    photonic_encoding: str = "phases"
    photonic_depths: tuple = (1, 2, 4, 6, 12)
    photonic_mesh_sampling: str = "haar"
    photonic_window: int = 2
    qrc_window: int = 2                  # gate-based, no reset, measure at the end (main)
    qrc_window_reset: int = 8            # gate-based, reset between steps (secondary)
    run_continuous_qrc: bool = True      # Exp3 density-matrix QRC (idealised reference)
    lih_angle_map: str = "minmax_pi"
    lih_train_sizes: tuple = (6, 8, 10, 12, 15, 20, 30, 40, 60, 100, 201)
    lih_grid_variants: int = 5
    include_haar: bool = True
    energy_source: str = "ems_3"         # Exp3 series (per-EMS file); 'hourly' = fallback
    energy_N: int = 5000
    t_dismiss: int = 100
    alpha: float = 1e-12
    retrain: bool = True
    n_ref_lo: int = 2000
    n_holdout_lo: int = 1000
    n_jobs: int = -1
    results_dir: str = "results/native_hardware"

    def quick(self) -> "Exp4Config":
        """Small configuration for smoke tests."""
        return replace(self, ns_task=2, ns_ors=6, energy_N=1500, n_ref_lo=300, n_holdout_lo=150,
                       iqm_layers=(1, 4), aqt_densities=(0.25, 1.0), aqt_alphas=(0.0, 2.0),
                       photonic_depths=(1, 12), lih_train_sizes=(10, 50, 201), lih_grid_variants=2,
                       results_dir=self.results_dir + "_quick")

    def as_dict(self) -> dict:
        return dataclasses.asdict(self)

    def hash(self) -> str:
        """Hash of the result-determining fields (``n_jobs`` does not change results)."""
        d = {k: v for k, v in self.as_dict().items() if k != "n_jobs"}
        return hashlib.sha256(json.dumps(d, sort_keys=True).encode()).hexdigest()[:12]

    @property
    def path(self) -> Path:
        return REPO_ROOT / self.results_dir


PLATFORM_LABELS = {
    "iqm": "IQM Euro-Q-Exa (lattice)",
    "aqt": "AQT Piast-Q (all-to-all)",
    "aqt_powerlaw": "AQT Piast-Q (power-law RXX)",
    "quandela": "Quandela Lucy (photonic)",
    "haar_qubit": "Haar U(2^6) reference (not native)",
}


def levels(cfg: Exp4Config, supplementary: bool = True) -> list[tuple[str, object]]:
    """(platform, level) cells of the main sweep (+ the AQT power-law supplement)."""
    cells = [("iqm", L) for L in cfg.iqm_layers]
    cells += [("aqt", r) for r in cfg.aqt_densities]
    cells += [("quandela", D) for D in cfg.photonic_depths]
    if cfg.include_haar:
        cells += [("quandela", "haar"), ("haar_qubit", "haar")]
    if supplementary and cfg.aqt_alphas:
        cells += [("aqt_powerlaw", a) for a in cfg.aqt_alphas]
    return cells


# ── Non-native Haar reference on 6 qubits ─────────────────────────────────────
class HaarQubitReference(GateNativeReservoir):
    """Haar-random U(2^n) — the Exp1–3 reference family; NOT hardware-native (reference only)."""

    platform_key = "haar_qubit"
    NATIVE_OPS = ("unitary",)
    complexity_name = "haar"

    def _build_circuit(self, rng):
        from qiskit import QuantumCircuit
        from qiskit.circuit.library import UnitaryGate
        from qiskit.quantum_info import random_unitary
        qc = QuantumCircuit(self.n_qubits, name="Haar")
        qc.append(UnitaryGate(random_unitary(2 ** self.n_qubits, seed=int(rng.integers(2**31)))),
                  range(self.n_qubits))
        return qc

    @property
    def complexity_value(self):
        return "haar"

    def resource_metrics(self) -> dict:
        return {"platform": "haar_qubit", "qubits": self.n_qubits, "note": "not hardware-native"}


def make_reservoir(platform: str, level, instance: int, cfg: Exp4Config, window: int = 1):
    """Build the reservoir of one (platform, level, instance) cell."""
    from .aqt_iontrap import AQTIonTrapReservoir
    from .iqm_lattice import IQMLatticeReservoir
    from .quandela_photonic import QuandelaPhotonicReservoir

    seed = derive_seed(cfg.base_seed, platform, str(level), instance)
    n = cfg.n_sites
    if platform == "iqm":
        r = IQMLatticeReservoir(n, n_layers=int(level), seed=seed)
    elif platform == "aqt":
        r = AQTIonTrapReservoir(n, density=float(level), n_layers=cfg.aqt_n_layers, seed=seed)
    elif platform == "aqt_powerlaw":
        r = AQTIonTrapReservoir(n, n_layers=cfg.aqt_n_layers, pattern="power_law", alpha=float(level), seed=seed)
    elif platform == "quandela":
        haar = level == "haar"
        r = QuandelaPhotonicReservoir(cfg.photonic_modes, cfg.photonic_photons,
                                      mesh_depth=cfg.photonic_modes if haar else int(level), window=window,
                                      haar=haar, mesh_sampling=cfg.photonic_mesh_sampling,
                                      encoding=cfg.photonic_encoding, n_data=n, seed=seed)
    elif platform == "haar_qubit":
        r = HaarQubitReference(n, seed=seed)
    else:
        raise ValueError(platform)
    return r.build()


# ── Data (loaded once per process) ────────────────────────────────────────────
_DATA = {}


def get_data(cfg: Exp4Config) -> dict:
    from . import tasks
    key = (cfg.energy_N, cfg.energy_source)
    if key not in _DATA:
        lih = tasks.load_lih()
        W = tasks.lih_configuration_weights(lih["ground_states"])
        tr, _ = tasks.lih_split_indices(len(lih["y"]))
        w_range = (W["w6"][tr].min(axis=0), W["w6"][tr].max(axis=0))
        series = tasks.load_ems_series() if cfg.energy_source == "ems_3" else tasks.load_energy_series()
        energy = tasks.energy_splits(series, N=cfg.energy_N)
        _DATA[key] = {"lih": lih, "w6": W["w6"], "w_range": w_range, "lih_report": W["report"],
                      "energy": energy}
    return _DATA[key]


# ── One instance ──────────────────────────────────────────────────────────────
def _lih_features(res, w6, angle_map, w_range, kind="correlators"):
    if res.modality == "photonic":
        return res.features_from_probs(res.run(w6, task="lih", angle_map=angle_map, w_range=w_range), kind)
    return res.extract_features(w6, angle_map=angle_map, w_range=w_range)


def _lih_block(res, data, cfg) -> dict:
    """LiH: main encoding, angle-map ablation, photonic feature variants, learning curve."""
    from . import tasks
    from .diagnostics import effective_rank
    y, w6, am = data["lih"]["y"], data["w6"], cfg.lih_angle_map
    rec = {}
    if res.modality == "photonic":
        P = res.run(w6, task="lih", angle_map=am, w_range=data["w_range"])
        variants = {"": "correlators", "_patterns": "patterns", "_first_moments": "first_moments"}
        for suffix, kind in variants.items():
            F = res.features_from_probs(P, kind)
            out = tasks.run_lih_readout(F, y, alpha=cfg.alpha)
            rec[f"lih_mse{suffix}"] = out["mse"]
            rec[f"lih_rmse{suffix}"] = out["rmse"]
            rec[f"lih_reff{suffix}"] = effective_rank(F)
            rec[f"lih_nfeat{suffix}"] = F.shape[1]
            if suffix == "":
                rec.update(lih_maxerr=out["max_err"], lih_frac_chem=out["frac_chem"])
    else:
        F = _lih_features(res, w6, am, data["w_range"])
        out = tasks.run_lih_readout(F, y, alpha=cfg.alpha)
        rec.update(lih_mse=out["mse"], lih_rmse=out["rmse"], lih_maxerr=out["max_err"],
                   lih_frac_chem=out["frac_chem"], lih_reff=effective_rank(F), lih_nfeat=F.shape[1])
    other = "arcsin_sqrt" if am != "arcsin_sqrt" else "minmax_pi"
    F_o = _lih_features(res, w6, other, data["w_range"])
    rec[f"lih_mse_{other}"] = tasks.run_lih_readout(F_o, y, alpha=cfg.alpha)["mse"]
    # learning curve (interpolation only): training geometries on jittered bond-length grids,
    # test points restricted to the scanned range, ridge strength by leave-one-out CV on the
    # training points only (alpha=1e-12 is ill-posed for n_train ≲ n_features), angle range
    # fitted on the subset; median over grid variants
    R = data["lih"]["bond_lengths"]
    for n_tr in cfg.lih_train_sizes:
        rm, mx, fc = [], [], []
        for v in range(1 if n_tr >= 201 else cfg.lih_grid_variants):
            idx = tasks.lih_train_subset(n_tr, bond_lengths=R, variant=v)
            wr = (w6[idx].min(axis=0), w6[idx].max(axis=0))
            o = tasks.run_lih_readout(_lih_features(res, w6, am, wr), y, alpha="cv", train_idx=idx, bond_lengths=R)
            rm.append(o["rmse"]); mx.append(o["max_err"]); fc.append(o["frac_chem"])
        rec[f"lih_lc_rmse_{n_tr}"] = float(np.median(rm))
        rec[f"lih_lc_maxerr_{n_tr}"] = float(np.median(mx))
        rec[f"lih_lc_fracchem_{n_tr}"] = float(np.median(fc))
    return rec


def _gate_window_resources(res, window, mode, prefix) -> dict:
    """Per-time-step circuit of a windowed gate QRC: depth, 1Q/2Q ops, resets, fidelity estimate."""
    from .adapters import GateWindowFeatures
    from .resources import circuit_fidelity_estimate, pauli_measurement_settings
    qc = GateWindowFeatures(res.circuit, mode=mode).window_circuit(np.zeros(window), measure=False)
    ops = dict(qc.count_ops())
    n2 = sum(v for k, v in ops.items() if k in res.TWO_QUBIT_OPS)
    n1 = sum(v for k, v in ops.items() if k in res.ONE_QUBIT_OPS)
    return {f"{prefix}_window": window, f"{prefix}_depth": qc.depth(), f"{prefix}_ops_1q": n1,
            f"{prefix}_ops_2q": n2, f"{prefix}_resets": int(ops.get("reset", 0)),
            f"{prefix}_settings": pauli_measurement_settings(res.n_qubits, (1, 2)),
            f"{prefix}_f_est": circuit_fidelity_estimate(res, qc)}


def _qrc_block(res, data, cfg) -> dict:
    """Energy QRC: windowed protocols (hardware-executable) + Exp3 continuous reference."""
    from . import tasks
    from .adapters import PhotonicQRC
    rec = {}
    run = lambda q: tasks.run_energy_qrc(q, data["energy"], t_dismiss=cfg.t_dismiss, retrain=cfg.retrain)
    keep = lambda m, pre: {f"{pre}_{k}": v for k, v in m.items() if k != "y_test_pred"}
    if res.modality == "photonic":
        rec.update(keep(run(PhotonicQRC(res, window=cfg.photonic_window, alpha=cfg.alpha,
                                        t_dismiss=cfg.t_dismiss)), "qrc"))
        res.window = cfg.photonic_window
        rq = res.resource_metrics()
        rec.update({f"resqrc_{k}": rq[k] for k in ("beam_splitters", "phase_shifters_fixed",
                                                    "phase_shifters_data", "optical_layers", "window")})
        res.window = 1
        return rec
    for pre, W, mode in (("qrc", cfg.qrc_window, "reupload"), ("qrcR", cfg.qrc_window_reset, "reset")):
        q = res.as_windowed_qrc(window=W, mode=mode, alpha=cfg.alpha, t_dismiss=cfg.t_dismiss)
        try:
            rec.update(keep(run(q), pre))
        except np.linalg.LinAlgError:              # degenerate feature matrix
            rec[f"{pre}_test_r2"] = np.nan
        if res.platform_key != "haar_qubit":
            rec.update(_gate_window_resources(res, W, mode, "res" + pre))
    if cfg.run_continuous_qrc:
        rec.update(keep(run(res.as_qrc(alpha=cfg.alpha, t_dismiss=cfg.t_dismiss)), "qrcC"))
    return rec


def run_instance(platform: str, level, instance: int, cfg: Exp4Config, tasks_to_run=("lih", "qrc")) -> dict:
    """Run LiH QELM, energy QRC and resource counting on one reservoir instance."""
    t0 = time.time()
    data = get_data(cfg)
    res = make_reservoir(platform, level, instance, cfg)
    rec = {"platform": platform, "level": str(level), "instance": instance, "seed": res.seed,
           "complexity_name": res.complexity_name}
    if "lih" in tasks_to_run:
        rec.update(_lih_block(res, data, cfg))

    if "qrc" in tasks_to_run and platform != "aqt_powerlaw":
        rec.update(_qrc_block(res, data, cfg))

    if platform != "haar_qubit":
        rm = res.resource_metrics()
        rec.update({f"res_{k}": v for k, v in rm.items() if not isinstance(v, (dict, list))})
    rec["runtime_s"] = time.time() - t0
    return rec


# ── ORS per cell ──────────────────────────────────────────────────────────────
def run_ors_cell(platform: str, level, cfg: Exp4Config, lo_reference=None) -> tuple[dict, list[dict]]:
    """ORS (gate) or ORS_LO (photonic) for one (platform, level) cell."""
    from .diagnostics import gate_ors

    insts = [make_reservoir(platform, level, i, cfg) for i in range(cfg.ns_ors)]
    if insts[0].modality == "gate":
        o = gate_ors([r.circuit_factory() for r in insts], cfg.n_sites, K=cfg.K, B=cfg.B)
        cell = {"platform": platform, "level": str(level), "metric": "ORS (gate, Haar U(2^n) ref.)",
                "gap": o["gap"], "gap_sem": o["gap_sem"], "ns": cfg.ns_ors, "K": cfg.K, "B": cfg.B,
                "lambda_ref": o["lambda_ref"]}
        per = o["per_instance_gap"]
    else:
        P = np.array([r.run() for r in insts])
        o = lo_reference.gap(P)
        cell = {"platform": platform, "level": str(level), "metric": "ORS_LO (photonic, Haar U(m) ref.)",
                "gap": o["gap"], "gap_sem": o["gap_sem"], "ns": cfg.ns_ors, "K": cfg.K, "B": 1,
                "lambda_ref": o["lambda_ref"], "D_eff": o["D_eff"]}
        per = o["per_instance_gap"]
    rows = [{"platform": platform, "level": str(level), "instance": i, "seed": r.seed, "ors_gap": float(g)}
            for i, (r, g) in enumerate(zip(insts, per))]
    return cell, rows


# ── Full sweep ────────────────────────────────────────────────────────────────
def _versions():
    pk = ["numpy", "scipy", "qiskit", "qiskit-aer", "scikit-learn", "pandas", "perceval-quandela",
          "networkx", "joblib"]
    out = {"python": _platform.python_version()}
    for p in pk:
        try:
            out[p] = md.version(p)
        except md.PackageNotFoundError:
            out[p] = None
    return out


def _git_sha():
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, capture_output=True,
                              text=True, check=True).stdout.strip()
    except Exception:
        return None


def run_all(cfg: Exp4Config = Exp4Config(), force: bool = False, verbose: bool = True) -> dict:
    """Run (or load cached) main sweep + ORS; returns dict of DataFrames."""
    from joblib import Parallel, delayed
    from .diagnostics import photonic_lo_reference

    out_dir = cfg.path
    out_dir.mkdir(parents=True, exist_ok=True)
    meta_path = out_dir / "metadata.json"
    files = {k: out_dir / f"{k}.csv" for k in ("records", "ors", "ors_instances")}
    if not force and meta_path.exists() and all(p.exists() for p in files.values()):
        meta = json.loads(meta_path.read_text())
        if meta.get("config_hash") == cfg.hash():
            if verbose:
                print(f"loaded cached Exp4 results ({cfg.hash()}) from {out_dir}")
            return {k: pd.read_csv(p) for k, p in files.items()} | {"metadata": meta}

    t0 = time.time()
    cells = levels(cfg)
    jobs = [(p, lv, i) for p, lv in cells for i in range(cfg.ns_task)]
    if verbose:
        print(f"Exp4: {len(jobs)} reservoir instances, {len(cells)} cells, n_jobs={cfg.n_jobs}")
    records = Parallel(n_jobs=cfg.n_jobs, verbose=5 if verbose else 0)(
        delayed(run_instance)(p, lv, i, cfg) for p, lv, i in jobs)

    lo_ref = photonic_lo_reference(cfg.photonic_modes, cfg.photonic_photons, K=cfg.K, n_ref=cfg.n_ref_lo,
                                   n_holdout=cfg.n_holdout_lo, seed=derive_seed(cfg.base_seed, "lo_ref"))
    ors = Parallel(n_jobs=cfg.n_jobs, verbose=5 if verbose else 0)(
        delayed(run_ors_cell)(p, lv, cfg, lo_ref) for p, lv in cells)

    df = pd.DataFrame(records)
    df_ors = pd.DataFrame([c for c, _ in ors])
    df_ors_i = pd.DataFrame([r for _, rows in ors for r in rows])
    df = df.merge(df_ors_i[["platform", "level", "instance", "ors_gap"]], on=["platform", "level", "instance"], how="left")
    df.to_csv(files["records"], index=False)
    df_ors.to_csv(files["ors"], index=False)
    df_ors_i.to_csv(files["ors_instances"], index=False)

    meta = {"config": cfg.as_dict(), "config_hash": cfg.hash(),
            "seeding": "seed = derive_seed(base_seed, platform, str(level), instance) (numpy SeedSequence)",
            "lo_reference": {"label": lo_ref.label, "n_ref": lo_ref.n_ref, "n_holdout": lo_ref.n_holdout,
                             "lambda_ref": lo_ref.lambda_ref, "D_eff": lo_ref.D_eff,
                             "lambda_ref_analytic_at_D_eff": lo_ref.lambda_ref_analytic, "seed": lo_ref.seed},
            "lih_symmetry_report": get_data(cfg)["lih_report"],
            "versions": _versions(), "git_sha": _git_sha(),
            "finished_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
            "wall_time_s": time.time() - t0}
    meta_path.write_text(json.dumps(meta, indent=1, default=str))
    if verbose:
        print(f"done in {time.time() - t0:.0f}s → {out_dir}")
    return {"records": df, "ors": df_ors, "ors_instances": df_ors_i, "metadata": meta}


# ── Vendor exchange ───────────────────────────────────────────────────────────
def export_vendor_circuits(cfg: Exp4Config, n_instances: int = 3, path: Path | None = None) -> Path:
    """Write native gate-based circuits as JSON op lists for experiments/native_hardware/vendor_validate.py."""
    path = path or (cfg.path / "vendor" / "circuits.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    circuits = []
    for platform, lv in levels(cfg, supplementary=False):
        if platform not in ("iqm", "aqt"):
            continue
        for i in range(n_instances):
            r = make_reservoir(platform, lv, i, cfg)
            circuits.append({"platform": platform, "level": str(lv), "instance": i, "seed": r.seed,
                             "kind": "reservoir", "n_qubits": r.n_qubits, "layout": list(r.layout),
                             "ops": r.export_ops(), "f_est": r.resource_metrics().get("f_est")})
        # one hardware-executable QRC time step (no-reset window, measured at the end)
        from .adapters import GateWindowFeatures
        r = make_reservoir(platform, lv, 0, cfg)
        qc = GateWindowFeatures(r.circuit, mode="reupload").window_circuit(
            np.linspace(0.2, 0.8, cfg.qrc_window), measure=False)
        ops = [{"name": ins.operation.name, "qubits": [qc.find_bit(q).index for q in ins.qubits],
                "params": [float(x) for x in ins.operation.params]} for ins in qc.data]
        circuits.append({"platform": platform, "level": str(lv), "instance": 0, "seed": r.seed,
                         "kind": "qrc_window", "n_qubits": r.n_qubits, "layout": list(r.layout), "ops": ops,
                         "f_est": None})
    path.write_text(json.dumps({"config_hash": cfg.hash(), "circuits": circuits}))
    return path


# ── Sampling (shot) budget ────────────────────────────────────────────────────
def _pauli1_from_samples(psi_batch, n, shots, rng):
    """Finite-shot estimates of all 1-local Paulis from 3 global settings (X, Y, Z).

    ``psi_batch``: (N, 2^n) output states. shots=None → exact expectation values.
    """
    H = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
    Sdg = np.diag([1, -1j])
    rot = {"X": H, "Y": H @ Sdg, "Z": np.eye(2)}
    bits = ((np.arange(2 ** n)[:, None] >> np.arange(n)[None, :]) & 1)       # (D, n), qubit j = bit j
    signs = 1 - 2 * bits
    feats = []
    for b in ("X", "Y", "Z"):
        R = rot[b]
        for _ in range(n - 1):
            R = np.kron(rot[b], R)
        p = np.abs(psi_batch @ R.T) ** 2
        if shots is not None:
            p = np.array([rng.multinomial(shots, pi / pi.sum()) / shots for pi in p])
        feats.append(p @ signs)                                                  # (N, n)
    return np.hstack(feats)


def shot_budget(cfg: Exp4Config, representative: dict,
                shots_list=(100, 1000, 10000, 100000, 1000000, None), reps: int = 3) -> pd.DataFrame:
    """LiH MSE vs shots per input for one representative instance per platform.

    Gate-based: 3 measurement settings (all-X/Y/Z) × ``shots``; photonic: ``shots``
    post-selected n-photon events (raw shots = shots × η^-n, see resources).
    """
    from . import tasks
    from .adapters import weights_to_angles
    data = get_data(cfg)
    y, w6 = data["lih"]["y"], data["w6"]
    rows = []
    for platform, level in representative.items():
        res = make_reservoir(platform, level, 0, cfg)
        th = weights_to_angles(w6, cfg.lih_angle_map, data["w_range"])
        if res.modality == "gate":
            single = np.stack([np.cos(th / 2), np.sin(th / 2)], axis=-1)        # (N, n, 2)
            psi = single[:, -1]
            for q in range(res.n_qubits - 2, -1, -1):
                psi = np.einsum("ia,ib->iab", psi, single[:, q]).reshape(len(w6), -1)
            psi = psi @ res.unitary().T
        else:
            P = res.run(w6, task="lih", angle_map=cfg.lih_angle_map, w_range=data["w_range"])
        for S in shots_list:
            for r in range(reps if S is not None else 1):
                rng = np.random.default_rng(derive_seed(cfg.base_seed, "shots", platform, S or 0, r))
                if res.modality == "gate":
                    F = _pauli1_from_samples(psi, res.n_qubits, S, rng)
                    total = None if S is None else 3 * S
                else:
                    Ph = P if S is None else np.array([rng.multinomial(S, p / p.sum()) / S for p in P])
                    F = res.features_from_probs(Ph, "correlators")
                    total = S
                o, ocv = tasks.run_lih_readout(F, y, alpha=cfg.alpha), tasks.run_lih_readout(F, y, alpha="cv")
                rows.append({"platform": platform, "level": str(level), "shots": S if S else np.inf,
                             "shots_per_input_total": total if total else np.inf, "rep": r,
                             "lih_mse": o["mse"], "lih_rmse": o["rmse"],
                             "lih_mse_cv": ocv["mse"], "lih_rmse_cv": ocv["rmse"]})
    return pd.DataFrame(rows)


# ── Noise-aware checks ────────────────────────────────────────────────────────
def photonic_noise_sweep(cfg: Exp4Config, level, indistinguishabilities=(1.0, 0.98, 0.95, 0.9, 0.8),
                         n_instances: int = 3) -> pd.DataFrame:
    """LiH MSE / R_eff vs photon indistinguishability (Perceval NoiseModel, assumed values)."""
    import perceval as pcvl
    from . import tasks
    from .adapters import weights_to_angles
    from .diagnostics import effective_rank
    data = get_data(cfg)
    rows = []
    for i in range(n_instances):
        res = make_reservoir("quandela", level, i, cfg)
        th = weights_to_angles(data["w6"], cfg.lih_angle_map, data["w_range"])
        Vs = [res.circuit_unitary([t]) for t in th]
        for ind in indistinguishabilities:
            nm = pcvl.NoiseModel(indistinguishability=ind)
            P = np.array([res.probabilities(V) if ind == 1.0 else res.noisy_probabilities(V, nm) for V in Vs])
            F = res.features_from_probs(P, "correlators")
            rows.append({"level": str(level), "instance": i, "indistinguishability": ind,
                         "lih_mse": tasks.run_lih_readout(F, data["lih"]["y"], alpha=cfg.alpha)["mse"],
                         "lih_reff": effective_rank(F)})
    return pd.DataFrame(rows)


def vendor_noise_crosscheck(cfg: Exp4Config, counts_path: Path | None = None) -> pd.DataFrame | None:
    """Ideal vs vendor-emulator (noisy) ORS gap for the exported gate-based circuits (Z basis).

    The noisy ORS uses ``order_statistics_score_noisy`` / ``lambda_haar_largeD_noisy`` with
    the resource-based fidelity estimate f_est (Exp-hardware convention) and with f = 1.
    """
    from .diagnostics import gate_ors_from_probs
    counts_path = counts_path or (cfg.path / "vendor" / "noisy_counts.json")
    if not counts_path.exists():
        return None
    runs = [r for r in json.loads(counts_path.read_text())["runs"] if r.get("kind", "reservoir") == "reservoir"]
    n = cfg.n_sites
    D = 2 ** n
    rows = []
    for r in runs:
        p = np.zeros(D)
        for bitstr, c in r["counts"].items():
            p[int(bitstr.replace(" ", ""), 2)] += c
        p /= p.sum()
        res = make_reservoir(r["platform"], r["level"] if r["platform"] != "iqm" else int(r["level"]), r["instance"], cfg)
        ideal = np.abs(res.run()) ** 2
        f = r.get("f_est") or 1.0
        rows.append({"platform": r["platform"], "level": r["level"], "instance": r["instance"],
                     "backend": r["backend"], "shots": r["shots"], "f_est": f,
                     "tvd_ideal_vs_emulator": 0.5 * np.abs(p - ideal).sum(),
                     "ors_gap_ideal": gate_ors_from_probs(ideal[None], D, cfg.K)["gap"],
                     "ors_gap_emulator_f1": gate_ors_from_probs(p[None], D, cfg.K, 1.0)["gap"],
                     "ors_gap_emulator_fest": gate_ors_from_probs(p[None], D, cfg.K, min(1.0, max(f, 0.05)))["gap"]})
    return pd.DataFrame(rows)


# ── Energy QRC sampling budget ────────────────────────────────────────────────
def qrc_shot_budget(cfg: Exp4Config, configs, shots_list=(100, 1000, 10000, 100000, None),
                    reps: int = 2, n_instances: int = 2) -> pd.DataFrame:
    """Energy test R² vs shots per measurement setting per time step.

    ``configs``: list of (platform, level, protocol, window) with protocol in
    {'reupload', 'reset', 'photonic'}. Gate-based: every 1+2-local Pauli estimated from S shots
    of its product-basis setting (Gaussian approximation, variance (1-<P>²)/S). Photonic: S
    post-selected 6-photon events per time step drawn from the exact pattern distribution.
    """
    from joblib import Parallel, delayed
    jobs = [(c, i, S, r) for c in configs for i in range(n_instances) for S in shots_list
            for r in range(reps if S else 1)]
    rows = Parallel(n_jobs=cfg.n_jobs)(delayed(_qrc_shot_job)(c, i, S, r, cfg) for c, i, S, r in jobs)
    return pd.DataFrame(rows)


def _qrc_shot_job(conf, inst, S, rep, cfg):
    from . import tasks
    from .adapters import GateWindowFeatures, PhotonicQRC, WindowedQRC
    platform, level, protocol, W = conf
    data = get_data(cfg)
    rng = np.random.default_rng(derive_seed(cfg.base_seed, "qrc_shots", platform, str(level), protocol, W,
                                            inst, S or 0, rep))
    res = make_reservoir(platform, level, inst, cfg)
    if protocol == "photonic":
        exact = res.probabilities
        if S is not None:
            def sampled(V):
                p = np.clip(exact(V), 0, None)
                return rng.multinomial(S, p / p.sum()) / S
            res.probabilities = sampled
        qrc = PhotonicQRC(res, window=W, alpha=cfg.alpha, t_dismiss=cfg.t_dismiss)
    else:
        prov = GateWindowFeatures(res.circuit, mode=protocol)
        exact_f = prov.window_features
        if S is not None:
            def noisy(*a, **k):
                x = exact_f(*a, **k)
                return x + rng.normal(size=x.shape) * np.sqrt(np.clip(1 - x ** 2, 0, None) / S)
            prov.window_features = noisy
        qrc = WindowedQRC(prov, window=W, alpha=cfg.alpha, t_dismiss=cfg.t_dismiss)
    try:
        r2 = tasks.run_energy_qrc(qrc, data["energy"], t_dismiss=cfg.t_dismiss, retrain=cfg.retrain)["test_r2"]
    except np.linalg.LinAlgError:
        r2 = np.nan
    return {"platform": platform, "level": str(level), "protocol": protocol, "window": W,
            "shots": S if S else np.inf, "instance": inst, "rep": rep, "test_r2": r2}


# ── Classical baselines (same data, same readout protocols) ───────────────────
def classical_baselines(cfg: Exp4Config) -> dict:
    """LiH: ridge on the 6 weights (linear / quadratic / cubic features), incl. learning curve.
    Energy: persistence and AR(W) baselines through the SAME windowed readout protocol."""
    from sklearn.preprocessing import PolynomialFeatures
    from . import tasks
    from .adapters import ClassicalWindowFeatures, WindowedQRC
    data = get_data(cfg)
    y, w6 = data["lih"]["y"], data["w6"]
    lih_rows = []
    for name, deg in (("linear", 1), ("quadratic", 2), ("cubic", 3)):
        F = PolynomialFeatures(deg, include_bias=False).fit_transform(w6)
        o = tasks.run_lih_readout(F, y, alpha=cfg.alpha)
        row = {"model": f"classical ridge, {name} in w", "n_features": F.shape[1], "lih_mse": o["mse"],
               "lih_rmse": o["rmse"], "lih_maxerr": o["max_err"], "lih_frac_chem": o["frac_chem"]}
        R = data["lih"]["bond_lengths"]
        for n_tr in cfg.lih_train_sizes:
            ol = [tasks.run_lih_readout(F, y, alpha="cv", bond_lengths=R,
                                        train_idx=tasks.lih_train_subset(n_tr, bond_lengths=R, variant=v))
                  for v in range(1 if n_tr >= 201 else cfg.lih_grid_variants)]
            row[f"lih_lc_rmse_{n_tr}"] = float(np.median([o["rmse"] for o in ol]))
            row[f"lih_lc_maxerr_{n_tr}"] = float(np.median([o["max_err"] for o in ol]))
            row[f"lih_lc_fracchem_{n_tr}"] = float(np.median([o["frac_chem"] for o in ol]))
        lih_rows.append(row)
    e = data["energy"]
    prev = np.concatenate([[e["y_val"][-1]], e["y_test"][:-1]])
    en_rows = [{"model": "persistence (ŷ_t = y_t-1)", "window": 1,
                "test_r2": tasks.compute_metrics(e["y_test"], prev)[3]}]
    for kind in ("linear", "quadratic"):
        for W in (1, 2, 4, 8, 16):
            q = WindowedQRC(ClassicalWindowFeatures(kind), window=W, alpha=cfg.alpha, t_dismiss=cfg.t_dismiss)
            en_rows.append({"model": f"classical {kind} AR({W})", "window": W,
                            "test_r2": tasks.run_energy_qrc(q, e, t_dismiss=cfg.t_dismiss,
                                                            retrain=cfg.retrain)["test_r2"]})
    return {"lih": pd.DataFrame(lih_rows), "energy": pd.DataFrame(en_rows)}


# ── Inference-only on hardware (readout trained on a noise-aware digital twin) ─
def _hardware_features(res, F_exact, P_exact, S, rng, f):
    """Emulated hardware features: gate — depolarising factor f + shot noise (Gaussian, per Pauli);
    photonic — S post-selected events drawn from the exact distribution."""
    if res.modality == "photonic":
        P = P_exact if S is None else np.array([rng.multinomial(S, p / p.sum()) / S for p in P_exact])
        return res.features_from_probs(P)
    X = f * F_exact
    return X if S is None else X + rng.normal(size=X.shape) * np.sqrt(np.clip(1 - X ** 2, 0, None) / S)


def lih_inference_budget(cfg: Exp4Config, representative: dict,
                         shots_list=(100, 300, 1000, 3000, 10000, 30000, 100000, 300000, 1000000, None),
                         n_test: int = 20, reps: int = 2, instances=range(5)) -> pd.DataFrame:
    """LiH, inference only: the readout is trained on simulated features with the SAME noise model
    and shot count (noise-aware twin, ridge α by leave-one-out CV); only ``n_test`` test geometries
    would be measured on hardware. Shots are per measurement setting (gate: 3 settings) or
    post-selected events (photonic). Repeated over several reservoir instances (report the median)."""
    from . import tasks
    data = get_data(cfg)
    y, w6 = data["lih"]["y"], data["w6"]
    tr, te_all = tasks.lih_split_indices()
    te = np.sort(np.random.default_rng(derive_seed(cfg.base_seed, "lih_test_subset")).choice(te_all, n_test, replace=False))
    rows = []
    for platform, level in representative.items():
        for inst in instances:
            res = make_reservoir(platform, level, inst, cfg)
            if res.modality == "photonic":
                P = res.run(w6, task="lih", angle_map=cfg.lih_angle_map, w_range=data["w_range"])
                F, f = res.features_from_probs(P), np.nan
            else:
                P, F = None, _lih_features(res, w6, cfg.lih_angle_map, data["w_range"])
                f = res.resource_metrics()["f_est"]
            for S in shots_list:
                for r in range(reps if S else 1):
                    rng = np.random.default_rng(derive_seed(cfg.base_seed, "lih_inf", platform, inst, S or 0, r))
                    H_tr = _hardware_features(res, F[tr], None if P is None else P[tr], S, rng, f)
                    H_te = _hardware_features(res, F[te], None if P is None else P[te], S, rng, f)
                    o = tasks.run_lih_readout(np.vstack([H_tr, H_te]), np.vstack([y[tr], y[te]]), alpha="cv",
                                              test_size=len(te), random_state=None,
                                              _split=(np.arange(len(tr)), len(tr) + np.arange(len(te))))
                    rows.append({"platform": platform, "level": str(level), "instance": inst, "f_est": f,
                                 "shots": S if S else np.inf, "rep": r, "rmse": o["rmse"],
                                 "r2": 1 - o["mse"] / y[te].var(axis=0).mean()})
    return pd.DataFrame(rows)


def qrc_reservoir(cfg: Exp4Config, platform: str, level, n_layers=None):
    """Reservoir used for the energy-task hardware plan (AQT may use fewer layers than the sweep)."""
    from .aqt_iontrap import AQTIonTrapReservoir
    if platform == "aqt" and n_layers:
        return AQTIonTrapReservoir(cfg.n_sites, density=float(level), n_layers=n_layers,
                                   seed=derive_seed(cfg.base_seed, "aqt_qrc", level, n_layers)).build()
    return make_reservoir(platform, level, 0, cfg)


def _qrc_inference_job(conf, S, rep, cfg, T_test):
    from . import tasks
    from .adapters import GateWindowFeatures, PhotonicQRC, WindowedQRC
    platform, level, n_layers = conf
    d = get_data(cfg)["energy"]
    rng = np.random.default_rng(derive_seed(cfg.base_seed, "qrc_inf", platform, str(level), S or 0, rep))
    res = qrc_reservoir(cfg, platform, level, n_layers)
    if res.modality == "photonic":
        exact = res.probabilities
        if S is not None:
            res.probabilities = lambda V: rng.multinomial(S, np.clip(exact(V), 0, None) / exact(V).sum()) / S
        qrc, f = PhotonicQRC(res, window=cfg.photonic_window, alpha=cfg.alpha, t_dismiss=cfg.t_dismiss), np.nan
    else:
        f = _gate_window_resources(res, cfg.qrc_window, "reupload", "x")["x_f_est"]
        prov = GateWindowFeatures(res.circuit, mode="reupload", observables_type=(1,))   # 3 settings
        exact_f = prov.window_features

        def noisy(*a, **k):
            x = f * exact_f(*a, **k)
            return x if S is None else x + rng.normal(size=x.shape) * np.sqrt(np.clip(1 - x ** 2, 0, None) / S)
        prov.window_features = noisy
        qrc = WindowedQRC(prov, window=cfg.qrc_window, alpha=cfg.alpha, t_dismiss=cfg.t_dismiss)
    best = None                                     # ridge α chosen on the simulated validation segment
    for a in (1e-12, 1e-8, 1e-6, 1e-4, 1e-2):
        qrc.alpha = a
        qrc.train(y=d["y_train_scaled"], scaler=d["scaler"])
        r2v = tasks.compute_metrics(d["y_val"], qrc.test(y=d["y_val_scaled"], scaler=d["scaler"], retrain=False))[3]
        best = max(best or (r2v, a), (r2v, a))
    qrc.alpha = best[1]
    qrc.train(y=d["y_train_scaled"], scaler=d["scaler"])
    qrc.test(y=d["y_val_scaled"], scaler=d["scaler"], retrain=False)
    pt = qrc.test(y=d["y_test_scaled"][:T_test], scaler=d["scaler"], retrain=False)
    return {"platform": platform, "level": str(level), "shots": S if S else np.inf, "rep": rep, "f_est": f,
            "test_r2": tasks.compute_metrics(d["y_test"][:T_test], pt)[3]}


def qrc_inference_budget(cfg: Exp4Config, configs, shots_list=(30, 100, 300, 1000, 3000, 10000, 30000, 100000, None),
                         T_test: int = 200, reps: int = 2) -> pd.DataFrame:
    """Energy QRC, inference only: readout trained on the noise-aware twin (no retraining on hardware),
    ``T_test`` test time steps measured on hardware. Gate: window-2 circuits, 1-local Paulis (3 settings)."""
    from joblib import Parallel, delayed
    jobs = [(c, S, r) for c in configs for S in shots_list for r in range(reps if S else 1)]
    return pd.DataFrame(Parallel(n_jobs=cfg.n_jobs)(delayed(_qrc_inference_job)(c, S, r, cfg, T_test) for c, S, r in jobs))
