"""Run the Exp4 LiH and energy task circuits on the vendors' noisy emulators.

Run in the *vendor* environment (see ``requirements-vendor.txt``):

    ../.venv-vendor/bin/python experiments/native_hardware/vendor_tasks.py \
        --tasks results/native_hardware/vendor/task_circuits.json [--platform iqm|aqt|all]

Input  : ``task_circuits.json`` written by ``native_reservoirs.noisy_emulation.export_task_circuits``
         (main environment). Each circuit is a list of segments (data-encoding ops, the reservoir
         ``"R"`` or the hidden-qubit preparation ``"P"``); here every circuit is completed with the
         basis rotation of each measurement setting (X, Y, Z on every qubit, native R gates) and a
         final measurement of all qubits.
Run    : IQM — transpiled against ``IQMFakeAphrodite`` (Crystal-54) on the recorded physical
         layout (optimisation level 0; asserts 0 SWAPs and native ops only), executed on the fake
         backend's sample noise model.
         AQT — transpiled against ``offline_simulator_noise`` and executed there (the API allows at
         most 2000 shots per circuit and 50 circuits per job, so shots are split into repeats;
         ``--workers`` runs slices of the circuits in parallel processes).
Output : ``task_counts_<platform>_<task>.npz`` (``counts``: inputs × settings × 2^n, bit j = qubit j)
         and a ``.json`` sidecar (backend, versions, shots, checks) next to the input file.
"""

from __future__ import annotations

import argparse
import importlib.metadata as md
import json
import time
import warnings
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]


def build(entry, segments, basis, measure=True):
    from qiskit import QuantumCircuit
    n = entry["n_qubits"]
    qc = QuantumCircuit(n, n)
    for seg in segments:
        ops = entry["reservoir_ops"] if seg == "R" else entry.get("prep_ops", []) if seg == "P" else seg
        for name, q, p in ops:
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
    for name, theta, phi in basis:
        for q in range(n):
            qc.r(theta, phi, q)
    if measure:
        qc.measure(range(n), range(n))
    return qc


def _to_array(counts: dict, n: int) -> np.ndarray:
    a = np.zeros(2 ** n, dtype=np.int64)
    for bitstr, c in counts.items():
        a[int(bitstr.replace(" ", ""), 2)] += c
    return a


def run_iqm(entry, basis_ops, settings, shots, chunk=300):
    from qiskit import transpile
    from iqm.qiskit_iqm import IQMFakeAphrodite
    backend = IQMFakeAphrodite()
    allowed = {"r", "cz", "measure", "barrier"}
    n, N = entry["n_qubits"], len(entry["inputs"])
    counts = np.zeros((N, len(settings), 2 ** n), dtype=np.int64)
    jobs = [(i, s) for i in range(N) for s in range(len(settings))]
    swaps, non_native = 0, 0
    for k in range(0, len(jobs), chunk):
        part = jobs[k:k + chunk]
        circs = [build(entry, entry["inputs"][i], basis_ops[settings[s]]) for i, s in part]
        tq = transpile(circs, backend=backend, initial_layout=entry["layout"], optimization_level=0,
                       seed_transpiler=0)
        for c in tq:
            ops = dict(c.count_ops())
            swaps += int(ops.get("swap", 0))
            non_native += int(not set(ops) <= allowed)
        res = backend.run(tq, shots=shots, seed_simulator=1234 + k).result()
        for j, (i, s) in enumerate(part):
            counts[i, s] = _to_array(res.get_counts(j), n)
    return counts, {"backend": backend.name, "swaps": swaps, "non_native_circuits": non_native}


def _aqt_part(args):
    """Run one slice of (input, setting) jobs on the AQT offline noisy simulator (own process)."""
    from qiskit import transpile
    from qiskit_aqt_provider import AQTProvider
    entry, basis_ops, settings, shots, part, per_job, max_shots = args
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        backend = AQTProvider().get_backend("offline_simulator_noise")
    allowed = {"r", "rz", "rxx", "measure", "barrier"}
    n = entry["n_qubits"]
    reps = [max_shots] * (shots // max_shots) + ([shots % max_shots] if shots % max_shots else [])
    out, non_native = np.zeros((len(part), 2 ** n), dtype=np.int64), 0
    for k in range(0, len(part), per_job):
        sub = part[k:k + per_job]
        circs = [build(entry, entry["inputs"][i], basis_ops[settings[s]]) for i, s in sub]
        tq = transpile(circs, backend=backend, optimization_level=0, seed_transpiler=0)
        non_native += sum(int(not set(c.count_ops()) <= allowed) for c in tq)
        for r in reps:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                res = backend.run(tq, shots=r, with_progress_bar=False).result()
            for j in range(len(sub)):
                out[k + j] += _to_array(res.get_counts(j), n)
        print(f"[aqt worker] {k + len(sub)}/{len(part)} circuits", flush=True)
    return out, non_native, backend.name, reps


def run_aqt(entry, basis_ops, settings, shots, per_job=50, max_shots=2000, workers=1):
    from concurrent.futures import ProcessPoolExecutor
    n, N = entry["n_qubits"], len(entry["inputs"])
    jobs = [(i, s) for i in range(N) for s in range(len(settings))]
    parts = [p for p in np.array_split(np.arange(len(jobs)), max(1, workers)) if len(p)]
    args = [(entry, basis_ops, settings, shots, [jobs[j] for j in p], per_job, max_shots) for p in parts]
    if workers > 1:
        import os
        os.environ.setdefault("OMP_NUM_THREADS", "1")       # one simulator thread per worker process
        with ProcessPoolExecutor(max_workers=workers) as ex:
            results = list(ex.map(_aqt_part, args))
    else:
        results = [_aqt_part(a) for a in args]
    counts = np.zeros((N, len(settings), 2 ** n), dtype=np.int64)
    for p, (out, _, _, _) in zip(parts, results):
        for row, j in zip(out, p):
            i, s = jobs[j]
            counts[i, s] = row
    return counts, {"backend": results[0][2], "swaps": 0, "non_native_circuits": sum(r[1] for r in results),
                    "shot_repeats": results[0][3], "workers": len(parts)}


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--tasks", type=Path, default=ROOT / "results/native_hardware/vendor/task_circuits.json")
    ap.add_argument("--platform", choices=("iqm", "aqt", "all"), default="all")
    ap.add_argument("--task", choices=("lih", "energy", "all"), default="all")
    ap.add_argument("--workers", type=int, default=16, help="parallel single-threaded AQT simulator processes")
    args = ap.parse_args()
    data = json.loads(args.tasks.read_text())
    out = args.tasks.parent
    versions = {p: md.version(p) for p in ("qiskit", "qiskit-aer", "iqm-client", "qiskit-aqt-provider")}
    for entry in data["tasks"]:
        p = entry["platform"]
        if args.platform not in ("all", p) or args.task not in ("all", entry["task"]):
            continue
        t = time.time()
        if p == "iqm":
            counts, info = run_iqm(entry, data["basis_ops"], data["settings"], data["shots"])
        else:
            counts, info = run_aqt(entry, data["basis_ops"], data["settings"], data["shots"], workers=args.workers)
        stem = f"task_counts_{p}_{entry['task']}"
        np.savez_compressed(out / f"{stem}.npz", counts=counts)
        meta = {"config_hash": data.get("config_hash"), "versions": versions, "platform": p, "task": entry["task"],
                "level": entry["level"], "n_layers": entry.get("n_layers"), "inputs": len(entry["inputs"]),
                "settings": data["settings"], "shots": data["shots"], "seconds": round(time.time() - t, 1), **info}
        (out / f"{stem}.json").write_text(json.dumps(meta, indent=1))
        print(f"{p} {entry['task']}: {len(entry['inputs'])} inputs × {len(data['settings'])} settings "
              f"in {time.time() - t:.0f}s ({info})", flush=True)


if __name__ == "__main__":
    main()
