"""Export vendor hardware targets to JSON (Exp4, hardware-native reservoirs).

Run in the *vendor* environment (qiskit 1.x + iqm-client[qiskit] + qiskit-aqt-provider),
see ``requirements-vendor.txt``:

    ../.venv-vendor/bin/python experiments/native_hardware/vendor_export.py

Writes
------
data/hardware/iqm_crystal54.json
    IQM Crystal-54 reference architecture (``IQMFakeAphrodite``): qubit names, coupling
    map, native operations from ``backend.target``, the SDK's sample error profile
    (T1/T2, PRX/CZ depolarising parameters, readout errors, gate durations) and whether
    the target exposes mid-circuit ``reset``.
data/hardware/aqt_offline20.json
    AQT offline simulators (``offline_simulator_no_noise`` / ``_noise``): qubit count,
    connectivity, native operations, the API angle constraints (read from the
    ``aqt_connector`` pydantic models) and the measurement restrictions enforced by
    ``qiskit_aqt_provider.circuit_to_aqt``.

This script deliberately imports nothing from ``native_reservoirs`` (which runs on qiskit 2.x);
the two environments only exchange JSON.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import importlib.metadata as md
import json
import math
import platform
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "data" / "hardware"


def _versions(pkgs):
    out = {"python": platform.python_version()}
    for p in pkgs:
        try:
            out[p] = md.version(p)
        except md.PackageNotFoundError:
            out[p] = None
    return out


def _stamp():
    return _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")


# ── IQM ───────────────────────────────────────────────────────────────────────
def export_iqm(out_dir: Path) -> Path:
    from iqm.qiskit_iqm import IQMFakeAphrodite

    b = IQMFakeAphrodite()
    n = b.num_qubits
    names = [b.index_to_qubit_name(i) for i in range(n)]
    edges = sorted({tuple(sorted(e)) for e in b.coupling_map.get_edges()})

    target_ops = {}
    for op in b.target.operation_names:
        qargs = b.target.qargs_for_operation_name(op)
        target_ops[op] = None if qargs is None else sorted(list(q) for q in qargs)

    ep = b.error_profile

    def _pairs(d):
        return {f"{a}-{c}": v for (a, c), v in d.items()}

    error_profile = {
        "name": ep.name,
        "t1s_ns": ep.t1s,
        "t2s_ns": ep.t2s,
        "prx_depolarizing": ep.single_qubit_gate_depolarizing_error_parameters.get("prx", {}),
        "cz_depolarizing": _pairs(ep.two_qubit_gate_depolarizing_error_parameters.get("cz", {})),
        "readout_errors": ep.readout_errors,
        "single_qubit_gate_durations_ns": ep.single_qubit_gate_durations,
        "two_qubit_gate_durations_ns": ep.two_qubit_gate_durations,
    }

    payload = {
        "platform": "IQM Crystal-54 reference architecture (proxy for Euro-Q-Exa, IQM Radiance 54q)",
        "source": "iqm.qiskit_iqm.IQMFakeAphrodite (iqm-client[qiskit])",
        "backend_name": b.name,
        "exported_utc": _stamp(),
        "versions": _versions(["qiskit", "qiskit-aer", "iqm-client"]),
        "num_qubits": n,
        "qubit_names": names,
        "coupling_edges": [list(e) for e in edges],
        "num_couplers": len(edges),
        "native_operations": sorted(b.target.operation_names),
        "target_qargs": target_ops,
        "mid_circuit_reset_in_target": "reset" in b.target.operation_names,
        "notes": [
            "Error profile is the SDK's *sample* noise model for the Crystal-54 facade, "
            "not a calibration of Euro-Q-Exa.",
            "iqm_target.py adds 'reset' (and if_else) to the target only when the QPU "
            "reports cc_prx loci (dynamic circuits); this facade does not.",
        ],
        "error_profile": error_profile,
    }
    path = out_dir / "iqm_crystal54.json"
    path.write_text(json.dumps(payload, indent=1))
    return path


# ── AQT ───────────────────────────────────────────────────────────────────────
def _field_bounds(model, field):
    """Return (lower, upper, lower_inclusive, upper_inclusive) for a pydantic field."""
    lo = hi = None
    lo_inc = hi_inc = True
    for m in model.model_fields[field].metadata:
        for attr, inc, is_lo in (("ge", True, True), ("gt", False, True),
                                 ("le", True, False), ("lt", False, False)):
            v = getattr(m, attr, None)
            if v is not None:
                if is_lo:
                    lo, lo_inc = v, inc
                else:
                    hi, hi_inc = v, inc
    fin = lambda x: None if x is None or not math.isfinite(x) else x
    return fin(lo), fin(hi), lo_inc, hi_inc


def export_aqt(out_dir: Path) -> Path:
    from qiskit import QuantumCircuit
    from qiskit_aqt_provider import AQTProvider
    from qiskit_aqt_provider.circuit_to_aqt import qiskit_to_aqt_circuit
    from aqt_connector.models.operations import GateR, GateRXX, GateRZ

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        provider = AQTProvider()          # no token: offline 'default' workspace only

    backends = {}
    for name in ("offline_simulator_no_noise", "offline_simulator_noise"):
        b = provider.get_backend(name)
        backends[name] = {
            "num_qubits": b.num_qubits,
            "native_operations": sorted(b.target.operation_names),
            "coupling_map": None if b.coupling_map is None else [list(e) for e in b.coupling_map.get_edges()],
            "resource_name": getattr(getattr(b, "resource_id", None), "resource_name", None),
        }

    # API angle constraints (stored by the API in units of pi)
    def bounds_rad(model, field):
        lo, hi, li, hinc = _field_bounds(model, field)
        return {"min_units_of_pi": lo, "max_units_of_pi": hi,
                "min_rad": None if lo is None else lo * math.pi,
                "max_rad": None if hi is None else hi * math.pi,
                "min_inclusive": li, "max_inclusive": hinc}

    constraints = {
        "r.theta": bounds_rad(GateR, "theta"),
        "r.phi": bounds_rad(GateR, "phi"),
        "rxx.theta": bounds_rad(GateRXX, "theta"),
        "rz.phi": bounds_rad(GateRZ, "phi"),
    }

    # Measurement / reset restrictions, probed through the provider's own converter
    probes = {}
    qc = QuantumCircuit(2, 2)
    qc.r(0.3, 0.1, 0)
    qc.measure(0, 0)
    qc.r(0.3, 0.1, 1)
    qc.measure(1, 1)
    try:
        qiskit_to_aqt_circuit(qc)
        probes["mid_circuit_measurement"] = {"accepted": True}
    except ValueError as exc:
        probes["mid_circuit_measurement"] = {"accepted": False, "error": str(exc)}
    qc = QuantumCircuit(1, 1)
    qc.r(0.3, 0.1, 0)
    qc.reset(0)
    qc.measure(0, 0)
    try:
        qiskit_to_aqt_circuit(qc)
        probes["reset"] = {"accepted": True}
    except ValueError as exc:
        probes["reset"] = {"accepted": False, "error": str(exc)}

    payload = {
        "platform": "AQT trapped-ion offline simulators (proxy for Piast-Q, AQT 20 qubits)",
        "source": "qiskit_aqt_provider.AQTProvider() default workspace (no token)",
        "exported_utc": _stamp(),
        "versions": _versions(["qiskit", "qiskit-aer", "qiskit-aqt-provider", "aqt-connector"]),
        "backends": backends,
        "connectivity": "all-to-all (target has no coupling map)",
        "api_angle_constraints": constraints,
        "api_probes": probes,
        "notes": [
            "Angle constraints are read from aqt_connector.models.operations (GateR, GateRXX, GateRZ).",
            "qiskit_aqt_provider.circuit_to_aqt rejects measurements that are not at the end of the "
            "circuit and any operation outside {rz, r, rxx}.",
        ],
    }
    path = out_dir / "aqt_offline20.json"
    path.write_text(json.dumps(payload, indent=1))
    return path


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for fn in (export_iqm, export_aqt):
        path = fn(args.out_dir)
        print(f"wrote {path.relative_to(ROOT) if path.is_relative_to(ROOT) else path}")


if __name__ == "__main__":
    main()
