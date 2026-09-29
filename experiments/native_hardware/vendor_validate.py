"""Validate Exp4 native circuits against the vendor SDK targets and emulators.

Run in the *vendor* environment (see ``requirements-vendor.txt``):

    ../.venv-vendor/bin/python experiments/native_hardware/vendor_validate.py \
        --circuits results/native_hardware/vendor/circuits.json

Input  : JSON op lists written by ``native_reservoirs.experiment.export_vendor_circuits``
         (main environment, qiskit 2.x). Circuits are rebuilt here with standard gates.
Checks :
  IQM  — transpile against ``IQMFakeAphrodite`` (Crystal-54) with the recorded physical
         layout at optimisation level 0: no SWAPs, CZ count unchanged, ops ⊆ {r, cz,
         measure, barrier}. Noise-aware run on the fake backend's sample noise model.
  AQT  — transpile against ``offline_simulator_no_noise``: RXX count unchanged, ops ⊆
         {r, rz, rxx, measure}; conversion to the AQT API payload (pydantic range checks);
         execution on the ideal and noisy offline simulators.
Output : ``validation.json`` (per-circuit checks) and ``noisy_counts.json`` (Z-basis counts)
         next to the input file.
"""

from __future__ import annotations

import argparse
import importlib.metadata as md
import json
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def rebuild(entry, measure=True):
    from qiskit import QuantumCircuit
    n = entry["n_qubits"]
    qc = QuantumCircuit(n, n if measure else 0)
    for op in entry["ops"]:
        name, q, p = op["name"], op["qubits"], op["params"]
        if name == "r":
            qc.r(p[0], p[1], q[0])
        elif name == "cz":
            qc.cz(q[0], q[1])
        elif name == "rz":
            qc.rz(p[0], q[0])
        elif name == "rxx":
            qc.rxx(p[0], q[0], q[1])
        elif name == "barrier":
            qc.barrier()
        else:
            raise ValueError(f"unexpected op {name}")
    if measure:
        qc.measure(range(n), range(n))
    return qc


def _counts2q(qc):
    return sum(1 for i in qc.data if i.operation.num_qubits == 2)


def validate_iqm(entries, shots):
    from qiskit import transpile
    from iqm.qiskit_iqm import IQMFakeAphrodite

    backend = IQMFakeAphrodite()
    allowed = {"r", "cz", "measure", "barrier"}
    rows, counts = [], []
    for e in entries:
        qc = rebuild(e)
        tq = transpile(qc, backend=backend, initial_layout=e["layout"], optimization_level=0,
                       seed_transpiler=0)
        ops = dict(tq.count_ops())
        phys_2q = sorted({tuple(sorted(tq.find_bit(q).index for q in i.qubits))
                          for i in tq.data if i.operation.num_qubits == 2})
        cmap = {tuple(sorted(x)) for x in backend.coupling_map.get_edges()}
        row = {"platform": "iqm", "level": e["level"], "instance": e["instance"], "kind": e.get("kind", "reservoir"),
               "ops_after": ops, "swaps": int(ops.get("swap", 0)),
               "cz_before": _counts2q(qc), "cz_after": int(ops.get("cz", 0)),
               "only_native": set(ops) <= allowed,
               "all_cz_on_couplers": all(p in cmap for p in phys_2q),
               "physical_qubits": e["layout"]}
        row["ok"] = row["swaps"] == 0 and row["cz_before"] == row["cz_after"] and row["only_native"] and row["all_cz_on_couplers"]
        rows.append(row)
        if shots:
            res = backend.run(tq, shots=shots, seed_simulator=1234).result().get_counts()
            counts.append({"platform": "iqm", "level": e["level"], "instance": e["instance"],
                           "kind": e.get("kind", "reservoir"),
                           "shots": shots, "backend": backend.name, "counts": res,
                           "f_est": e.get("f_est")})
    return rows, counts


def validate_aqt(entries, shots):
    from qiskit import transpile
    from qiskit_aqt_provider import AQTProvider
    from qiskit_aqt_provider.circuit_to_aqt import qiskit_to_aqt_circuit

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        provider = AQTProvider()
    ideal = provider.get_backend("offline_simulator_no_noise")
    noisy = provider.get_backend("offline_simulator_noise")
    allowed = {"r", "rz", "rxx", "measure", "barrier"}
    rows, counts = [], []
    for e in entries:
        qc = rebuild(e)
        tq = transpile(qc, backend=ideal, optimization_level=0, seed_transpiler=0)
        ops = dict(tq.count_ops())
        row = {"platform": "aqt", "level": e["level"], "instance": e["instance"],
               "kind": e.get("kind", "reservoir"), "ops_after": ops,
               "rxx_before": _counts2q(qc), "rxx_after": int(ops.get("rxx", 0)),
               "only_native": set(ops) <= allowed, "swaps": int(ops.get("swap", 0))}
        try:
            qiskit_to_aqt_circuit(tq)
            row["api_payload_ok"] = True
        except Exception as exc:                          # pydantic ValidationError / ValueError
            row["api_payload_ok"], row["api_error"] = False, str(exc)[:300]
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                ideal.run(tq, shots=200, with_progress_bar=False).result()
            row["ideal_run_ok"] = True
        except Exception as exc:
            row["ideal_run_ok"], row["run_error"] = False, str(exc)[:300]
        row["ok"] = (row["only_native"] and row["rxx_before"] == row["rxx_after"]
                     and row["api_payload_ok"] and row["ideal_run_ok"])
        rows.append(row)
        if shots:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                res = noisy.run(tq, shots=shots, with_progress_bar=False).result().get_counts()
            counts.append({"platform": "aqt", "level": e["level"], "instance": e["instance"],
                           "kind": e.get("kind", "reservoir"),
                           "shots": shots, "backend": noisy.name, "counts": res,
                           "f_est": e.get("f_est")})
    return rows, counts


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--circuits", type=Path, default=ROOT / "results/native_hardware/vendor/circuits.json")
    ap.add_argument("--shots-iqm", type=int, default=10000)
    ap.add_argument("--shots-aqt", type=int, default=2000)
    args = ap.parse_args()
    data = json.loads(args.circuits.read_text())
    circ = data["circuits"]
    iqm_rows, iqm_counts = validate_iqm([c for c in circ if c["platform"] == "iqm"], args.shots_iqm)
    aqt_rows, aqt_counts = validate_aqt([c for c in circ if c["platform"] == "aqt"], args.shots_aqt)
    rows = iqm_rows + aqt_rows
    versions = {p: md.version(p) for p in ("qiskit", "qiskit-aer", "iqm-client", "qiskit-aqt-provider")}
    summary = {"n_circuits": len(rows), "n_ok": sum(r["ok"] for r in rows),
               "iqm_total_swaps": sum(r["swaps"] for r in iqm_rows),
               "aqt_api_payload_ok": all(r["api_payload_ok"] for r in aqt_rows) if aqt_rows else None}
    out = args.circuits.parent
    (out / "validation.json").write_text(json.dumps(
        {"config_hash": data.get("config_hash"), "versions": versions, "summary": summary, "circuits": rows},
        indent=1, default=str))
    (out / "noisy_counts.json").write_text(json.dumps(
        {"config_hash": data.get("config_hash"), "versions": versions, "runs": iqm_counts + aqt_counts}))
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
