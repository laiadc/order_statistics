"""Platform-native resource metrics.

Gate-based (IQM, AQT)
    qubits, depth (parallel), 2-qubit depth, serial length, 1Q / 2Q / virtual op counts,
    routing overhead measured by qiskit's transpiler on the hardware coupling map,
    entangling-graph density, measurement settings for the Pauli features, and a crude
    circuit-fidelity estimate from per-operation error rates.

Photonic (Quandela)
    modes, photons, beam splitters, phase shifters (fixed / data-dependent), optical layers,
    compiled universal-chip MZI count, PNR / threshold pattern counts, feature count and
    sampling requirements. Photonic "layers" are a modality-specific quantity and are never
    compared numerically with gate depth.
"""

from __future__ import annotations

from itertools import combinations
from math import comb

import numpy as np

from .hardware_specs import AQT_PUBLISHED_FIDELITY, load_iqm_target


# ── Gate-based ────────────────────────────────────────────────────────────────
def routing_overhead(qc, edges, basis_gates, n_physical=None, layout=None,
                     optimization_level=0, seed=0) -> dict:
    """Transpile ``qc`` onto a coupling map and count the routing cost.

    ``edges=None`` means all-to-all. With a native circuit on a native layout the expected
    result is 0 SWAPs and an unchanged 2-qubit count.
    """
    from qiskit import transpile
    from qiskit.transpiler import CouplingMap

    two_q = lambda c: sum(1 for i in c.data if i.operation.num_qubits == 2)
    n2_before = two_q(qc)
    if edges is None:
        tq = transpile(qc, basis_gates=list(basis_gates), optimization_level=optimization_level,
                       seed_transpiler=seed)
        return {"routing_swaps": 0, "two_qubit_before": n2_before, "two_qubit_after": two_q(tq),
                "routing_extra_2q": two_q(tq) - n2_before, "routing_method": "all-to-all"}
    cmap = CouplingMap([list(e) for e in edges] + [[b, a] for a, b in edges])
    n_phys = n_physical or (max(max(e) for e in edges) + 1)
    layout = list(layout) if layout is not None else list(range(qc.num_qubits))
    from qiskit import QuantumCircuit
    big = QuantumCircuit(n_phys)
    big.compose(qc, qubits=layout, inplace=True)
    tq = transpile(big, coupling_map=cmap, basis_gates=list(basis_gates),
                   initial_layout=list(range(n_phys)), optimization_level=optimization_level,
                   routing_method="sabre", seed_transpiler=seed)
    n2_after = two_q(tq)
    return {"routing_swaps": int(tq.count_ops().get("swap", 0)),
            "two_qubit_before": n2_before, "two_qubit_after": n2_after,
            "routing_extra_2q": n2_after - n2_before,
            "routing_method": "qiskit sabre, fixed layout, opt. level %d" % optimization_level}


def pauli_measurement_settings(n: int, localities=(1,), seed: int = 0) -> int:
    """Number of product-basis measurement settings needed to estimate all k-local Paulis.

    1-local only: 3 (all-X, all-Y, all-Z). With 2-local terms a greedy set cover over random
    product settings is used (deterministic given ``seed``).
    """
    if max(localities) == 1:
        return 3
    need = {(i, j, a, b) for i, j in combinations(range(n), 2) for a in range(3) for b in range(3)}
    rng = np.random.default_rng(seed)
    settings = [np.full(n, a) for a in range(3)]
    for s in settings:
        need -= {(i, j, s[i], s[j]) for i, j in combinations(range(n), 2)}
    while need:
        best, gain = None, -1
        for _ in range(64):
            s = rng.integers(0, 3, n)
            g = sum(1 for i, j in combinations(range(n), 2) if (i, j, s[i], s[j]) in need)
            if g > gain:
                best, gain = s, g
        settings.append(best)
        need -= {(i, j, best[i], best[j]) for i, j in combinations(range(n), 2)}
    return len(settings)


def gate_resources(res) -> dict:
    """Resource metrics for a built :class:`~native_reservoirs.base.GateNativeReservoir`."""
    qc = res.circuit
    ops = dict(qc.count_ops())
    n1 = sum(v for k, v in ops.items() if k in res.ONE_QUBIT_OPS)
    n2 = sum(v for k, v in ops.items() if k in res.TWO_QUBIT_OPS)
    nv = sum(v for k, v in ops.items() if k in res.VIRTUAL_OPS)
    physical = n1 - nv + n2
    out = {
        "platform": res.platform_key,
        "qubits": res.n_qubits,
        "depth": qc.depth(),
        "depth_2q": qc.depth(lambda inst: inst.operation.num_qubits == 2),
        "ops_1q": n1,
        "ops_1q_virtual": nv,
        "ops_2q": n2,
        "ops_total": n1 + n2,
        "serial_length": physical,
        "op_counts": ops,
    }
    if res.platform_key == "iqm":
        out.update(routing_overhead(qc, res.hardware_edges, basis_gates=("r", "cz"),
                                    n_physical=res.G_hw.number_of_nodes(), layout=res.layout))
        out["couplers_used"] = len({tuple(sorted((res.circuit.find_bit(q).index for q in i.qubits)))
                                    for i in qc.data if i.operation.num_qubits == 2})
        out["couplers_in_patch"] = len(res.edges)
        out["f_est"] = fidelity_estimate_iqm(res)
    elif res.platform_key == "aqt":
        out.update(routing_overhead(qc, None, basis_gates=("r", "rz", "rxx")))
        out["graph_density"] = res.graph_density
        out["entangling_pairs"] = len(res.graph)
        out["f_est"] = fidelity_estimate_aqt(res)
    return out


def fidelity_estimate_iqm(res, target=None, circuit=None) -> float:
    """f ≈ Π(1-p_1q) Π(1-p_cz) Π(1-p_ro) from the SDK's *sample* Crystal-54 error profile.

    Depolarising parameters are used directly as error probabilities of a global
    depolarising model (the model used by the noisy ORS). Resets are not in the error
    profile and are counted as error-free. Indicative only (see ASSUMPTIONS).
    """
    target = target or res.target or load_iqm_target()
    qc = circuit if circuit is not None else res.circuit
    if not target or "error_profile" not in target:
        return float("nan")
    ep, names = target["error_profile"], target["qubit_names"]
    logf = 0.0
    for inst in qc.data:
        qs = [res.layout[qc.find_bit(q).index] for q in inst.qubits]
        if inst.operation.name == "r":
            logf += np.log1p(-ep["prx_depolarizing"].get(names[qs[0]], 0.0))
        elif inst.operation.name == "cz":
            a, b = names[qs[0]], names[qs[1]]
            p = ep["cz_depolarizing"].get(f"{a}-{b}", ep["cz_depolarizing"].get(f"{b}-{a}", 0.0))
            logf += np.log1p(-p)
    for q in res.layout:
        ro = ep["readout_errors"].get(names[q], {"0": 0.0, "1": 0.0})
        logf += np.log1p(-0.5 * (ro["0"] + ro["1"]))
    return float(np.exp(logf))


def fidelity_estimate_aqt(res, circuit=None) -> float:
    """f ≈ F_1q^{#R} F_2q^{#RXX} with vendor-published AQT figures (IBEX Q1, not Piast-Q)."""
    ops = (circuit if circuit is not None else res.circuit).count_ops()
    f1, f2 = AQT_PUBLISHED_FIDELITY["1q"], AQT_PUBLISHED_FIDELITY["2q"]
    return float(f1 ** ops.get("r", 0) * f2 ** ops.get("rxx", 0))


def circuit_fidelity_estimate(res, circuit=None) -> float:
    """Platform-appropriate fidelity estimate of ``circuit`` (default: the bare reservoir)."""
    if res.platform_key == "iqm":
        return fidelity_estimate_iqm(res, circuit=circuit)
    if res.platform_key == "aqt":
        return fidelity_estimate_aqt(res, circuit=circuit)
    return float("nan")


def routing_contrast(n: int, layout, hardware_edges, n_physical: int, seed: int = 0,
                     num_gates: int = 100, degree: int = 2) -> list[dict]:
    """Route Exp3 families (non-native, all-to-all) onto the same IQM patch.

    Circuits are first compiled to {r, cz} with all-to-all connectivity, then routed onto the
    lattice; the difference in CZ count is the price of a non-hardware-native design.
    """
    import random
    from qiskit import QuantumCircuit, transpile
    from qiskit.circuit.library import UnitaryGate
    from qiskit.quantum_info import random_unitary
    from QELM import QuantumCircQiskit

    fams = []
    for fam, kw in (("G3", dict(num_gates=num_gates)), ("D2_random", dict(degree=degree))):
        random.seed(seed)
        np.random.seed(seed)
        fams.append((fam, QuantumCircQiskit(fam, nqbits=n, statevector=False, **kw).get_reservoir()))
    qc_h = QuantumCircuit(n)
    qc_h.append(UnitaryGate(random_unitary(2 ** n, seed=seed)), range(n))
    fams.append(("Haar", qc_h))

    rows = []
    for fam, qc in fams:
        flat = transpile(qc, basis_gates=["r", "cz"], optimization_level=1, seed_transpiler=seed)
        r = routing_overhead(flat, hardware_edges, basis_gates=("r", "cz"), n_physical=n_physical,
                             layout=layout, optimization_level=1, seed=seed)
        rows.append({"family": fam, "cz_all_to_all": r["two_qubit_before"],
                     "cz_on_lattice": r["two_qubit_after"], "extra_cz": r["routing_extra_2q"],
                     "overhead_ratio": r["two_qubit_after"] / max(1, r["two_qubit_before"])})
    return rows


# ── Photonic ──────────────────────────────────────────────────────────────────
def n_patterns_pnr(m: int, n: int) -> int:
    """Number of n-photon output patterns with photon-number resolution: C(m+n-1, n)."""
    return comb(m + n - 1, n)


def n_patterns_threshold(m: int, n: int) -> int:
    """Collision-free n-click patterns seen by threshold detectors: C(m, n)."""
    return comb(m, n)


def shots_for_features(n_features: int, var_max: float, eps: float) -> float:
    """Shots per input so that every feature has standard error ≤ eps (worst-case variance)."""
    return var_max / eps ** 2


def shots_for_distribution(n_outcomes: int, eps_l1: float) -> float:
    """Shots per input so that E‖p̂ - p‖₁ ≲ eps_l1 (≈ sqrt(D/S))."""
    return n_outcomes / eps_l1 ** 2


def postselection_overhead(n_photons: int, eta: float) -> float:
    """Raw-shot multiplier for keeping only n-photon events at per-photon transmission η."""
    return eta ** (-n_photons)


def photonic_resources(res, eps: float = 0.01, eps_l1: float = 0.1, etas=(0.5, 0.3, 0.1)) -> dict:
    """Resource metrics of one circuit of a built :class:`~native_reservoirs.quandela_photonic.QuandelaPhotonicReservoir`.

    One circuit = [U_pre] + W × (data column + U_res), with W = ``res.window``.
    """
    m, n, W = res.n_modes, res.n_photons, res.window
    mesh = res.mesh_mzi_count
    n_mesh_blocks = W + (1 if res.encoding == "phases" else 0)
    enc_mzi = 0 if res.encoding == "phases" else res.n_data          # dual-rail: one data MZI per photon
    mzis = n_mesh_blocks * mesh + W * enc_mzi
    depth = res.mesh_depth if not res.haar else m
    D_pnr, D_thr = n_patterns_pnr(m, n), n_patterns_threshold(m, n)
    n_corr = m + comb(m, 2)
    out = {
        "platform": "quandela",
        "modes": m, "photons": n,
        "input_state": res.input_state_str,
        "encoding": res.encoding,
        "window": W,
        "mesh_depth": depth,
        "mzi_reservoir": n_mesh_blocks * mesh,
        "mzi_encoding": W * enc_mzi,
        "beam_splitters": 2 * mzis,
        "phase_shifters_fixed": 2 * n_mesh_blocks * mesh + n_mesh_blocks * m,   # 2 per MZI + output screens
        "phase_shifters_data": W * (res.n_data if res.encoding == "phases" else enc_mzi),
        "optical_layers": n_mesh_blocks * depth + W,
        "chip_mzis_universal": m * (m - 1) // 2,
        "patterns_pnr": D_pnr,
        "patterns_threshold_collision_free": D_thr,
        "features_correlators": n_corr,
        "shots_correlators_eps": shots_for_features(n_corr, var_max=n ** 2 / 4, eps=eps),
        "shots_patterns_l1": shots_for_distribution(D_pnr, eps_l1),
        "eps": eps, "eps_l1": eps_l1,
    }
    for eta in etas:
        out[f"postselection_x_eta{eta:g}"] = postselection_overhead(n, eta)
    return out


def photonic_projection(configs=((12, 6), (16, 8), (20, 10), (24, 12)), eps=0.01, eps_l1=0.1, eta=0.3):
    """Resource projection of the photonic reservoir up to Lucy's 24 modes / 12 photons."""
    rows = []
    for m, n in configs:
        D = n_patterns_pnr(m, n)
        F = m + comb(m, 2)
        rows.append({"modes": m, "photons": n, "chip_mzis": m * (m - 1) // 2,
                     "patterns_pnr": D, "features_correlators": F,
                     "shots_correlators": shots_for_features(F, n ** 2 / 4, eps),
                     "shots_patterns": shots_for_distribution(D, eps_l1),
                     f"postselection_x_eta{eta:g}": postselection_overhead(n, eta)})
    return rows
