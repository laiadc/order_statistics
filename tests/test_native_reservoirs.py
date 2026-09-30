"""Tests for the Exp4 hardware-native reservoirs.

Run from the repository root with the main environment:

    ../.venv/bin/python -m pytest tests/test_native_reservoirs.py -q
"""

import importlib.util
import random
from itertools import permutations
from math import factorial

import numpy as np
import pytest
from qiskit import QuantumCircuit
from qiskit.quantum_info import Operator, Statevector

from native_reservoirs.adapters import NativeQELM, NativeQRC, PhotonicQRC
from native_reservoirs.aqt_iontrap import AQTIonTrapReservoir
from native_reservoirs.base import derive_seed
from native_reservoirs.diagnostics import (effective_rank, naive_gate_ors_gap,
                                           photonic_lo_reference, qubit_haar_reference)
from native_reservoirs.iqm_lattice import IQMLatticeReservoir, hardware_graph
from native_reservoirs.quandela_photonic import (QuandelaPhotonicReservoir, mzi_circuit,
                                                 mzi_matrix)
from native_reservoirs import tasks


def _equal_up_to_phase(a, b, tol=1e-10):
    return abs(abs(np.vdot(a, b)) - 1.0) < tol


# ── seeds ─────────────────────────────────────────────────────────────────────
def test_derive_seed_deterministic_and_distinct():
    assert derive_seed(2026, "iqm", "4", 0) == derive_seed(2026, "iqm", "4", 0)
    assert derive_seed(2026, "iqm", "4", 0) != derive_seed(2026, "iqm", "4", 1)
    assert derive_seed(2026, "iqm", "4", 0) != derive_seed(2026, "aqt", "4", 0)


# ── native encodings (IQM PRX / AQT R) ────────────────────────────────────────
@pytest.mark.parametrize("theta", [0.0, 0.3, 1.7, np.pi, 5.1])
def test_native_encoding_identities(theta):
    # Exp3 'pauli' encoding: H then Rz(-θ)  ≅  PRX(π/2, π/2 - θ)
    a, b = QuantumCircuit(1), QuantumCircuit(1)
    a.h(0); a.rz(-theta, 0)
    b.r(np.pi / 2, np.pi / 2 - theta, 0)
    assert _equal_up_to_phase(Statevector(a).data, Statevector(b).data)
    # RY(θ) == PRX(θ, π/2) as operators
    c, d = QuantumCircuit(1), QuantumCircuit(1)
    c.ry(theta, 0)
    d.r(theta, np.pi / 2, 0)
    assert np.allclose(Operator(c).data, Operator(d).data)


def test_plus_state_is_native():
    qc = QuantumCircuit(1)
    qc.r(np.pi / 2, np.pi / 2, 0)
    assert _equal_up_to_phase(Statevector(qc).data, np.array([1, 1]) / np.sqrt(2))


# ── IQM ───────────────────────────────────────────────────────────────────────
def test_iqm_native_ops_topology_and_routing():
    r = IQMLatticeReservoir(6, n_layers=8, seed=3).build()
    G, target = hardware_graph()
    assert set(r.circuit.count_ops()) <= {"r", "cz"}
    for inst in r.circuit.data:
        if inst.operation.name == "cz":
            a, b = (r.layout[r.circuit.find_bit(q).index] for q in inst.qubits)
            assert G.has_edge(a, b), "CZ on a non-existent coupler"
    assert len(r.edges) == 7                              # densest 6-qubit patch = 2x3 block
    assert r.G_hw.subgraph(r.layout).degree(r.layout[0]) == max(dict(r.G_hw.subgraph(r.layout).degree()).values())
    rm = r.resource_metrics()
    assert rm["routing_swaps"] == 0 and rm["routing_extra_2q"] == 0
    for cls in r.colour_classes:                          # colour classes are parallel layers
        qs = [q for e in cls for q in e]
        assert len(qs) == len(set(qs))


def test_iqm_determinism():
    a = IQMLatticeReservoir(6, 4, seed=11).build()
    b = IQMLatticeReservoir(6, 4, seed=11).build()
    c = IQMLatticeReservoir(6, 4, seed=12).build()
    assert np.allclose(a.unitary(), b.unitary())
    assert not np.allclose(a.unitary(), c.unitary())


# ── AQT ───────────────────────────────────────────────────────────────────────
@pytest.mark.parametrize("density", [0.25, 0.5, 0.75, 1.0])
def test_aqt_native_ops_ranges_density(density):
    r = AQTIonTrapReservoir(6, density=density, n_layers=2, seed=5).build()
    assert set(r.circuit.count_ops()) <= {"r", "rz", "rxx"}
    for inst in r.circuit.data:
        p = [float(x) for x in inst.operation.params]
        if inst.operation.name == "r":
            assert 0 <= p[0] <= np.pi and 0 <= p[1] <= 2 * np.pi
        if inst.operation.name == "rxx":
            assert 0 < p[0] <= np.pi / 2
    assert r.graph_density == pytest.approx(max(1, round(density * 15)) / 15)
    assert r.resource_metrics()["routing_swaps"] == 0


def test_aqt_power_law_pattern():
    r = AQTIonTrapReservoir(6, pattern="power_law", alpha=2.0, seed=1).build()
    assert len(r.graph) == 15
    assert r.weights[(0, 1)] == pytest.approx(1.0) and r.weights[(0, 5)] == pytest.approx(1 / 25)


# ── adapters reuse the existing classes exactly ───────────────────────────────
def test_native_qelm_reproduces_quantumcircqiskit():
    from QELM import QuantumCircQiskit
    random.seed(0); np.random.seed(0)
    ref = QuantumCircQiskit("G3", num_gates=40, nqbits=4, encoding="pauli", observables_type=[1, 2])
    nat = NativeQELM(ref.reservoir_circuit, encoding="pauli", observables_type=[1, 2])
    x = np.linspace(0, 1, 17)
    assert np.allclose(ref.feature_matrix(x), nat.feature_matrix(x))


def test_native_qrc_reproduces_quantumrc():
    from QRC import QuantumRC_Circuit
    random.seed(1); np.random.seed(1)
    ref = QuantumRC_Circuit(gates_set="G3", num_gates=30, nqbits=3, n_measure=4, observables_type=[1, 2])
    nat = NativeQRC(ref.qc_U, observables_type=[1, 2])
    y = np.random.default_rng(0).uniform(size=(25, 1))
    assert np.allclose(ref._compute_features(y), nat._compute_features(y))


def test_weights_ry_encoding():
    from qiskit.quantum_info import SparsePauliOp
    ident = QuantumCircuit(6)
    zs = [SparsePauliOp("I" * (5 - j) + "Z" + "I" * j) for j in range(6)]
    q = NativeQELM(ident, encoding="weights_ry", observables_type=zs)
    w = np.random.default_rng(3).uniform(0, 0.4, size=(5, 6))
    F = q.feature_matrix(w)
    assert np.allclose(F, 1 - 2 * w)
    # circuit (native PRX gates) and analytic product state agree
    assert np.allclose(Statevector(q._encode(w[0])).data, q._encode_state_fast(w[0]))
    assert set(q._encode(w[0]).count_ops()) == {"r"}


# ── LiH symmetry reduction ────────────────────────────────────────────────────
def test_lih_symmetry_reduction():
    lih = tasks.load_lih()
    W = tasks.lih_configuration_weights(lih["ground_states"])     # asserts internally
    assert W["w6"].shape == (300, 6)
    rep = W["report"]
    assert rep["hamming_weight_2_strings"] == 28 and rep["populated_strings"] == 10
    assert rep["min_dominant_weight"] > 0.6
    assert np.all(W["w6"] >= 0) and np.all(W["w6"].sum(1) < 1)


def test_lih_readout_matches_exp3_split():
    tr, te = tasks.lih_split_indices(300)
    assert len(tr) == 201 and len(te) == 99


# ── photonic ──────────────────────────────────────────────────────────────────
def test_mzi_matrix_matches_perceval():
    for a, b in [(0.2, 1.3), (np.pi, 0.0), (4.0, 2.2)]:
        assert np.allclose(mzi_matrix(a, b), np.array(mzi_circuit(a, b).compute_unitary()))


def test_photonic_weight_encoding_calibration():
    r = QuandelaPhotonicReservoir(12, 6, mesh_depth=2, encoding="dualrail", seed=0).build()
    w = np.array([0.05, 0.1, 0.2, 0.4, 0.6, 0.95])
    E = r.encoding_unitary(2 * np.arcsin(np.sqrt(w)))
    assert np.allclose([abs(E[2 * k + 1, 2 * k]) ** 2 for k in range(6)], w)
    assert np.allclose(E, np.array(r.encoding_circuit(2 * np.arcsin(np.sqrt(w))).compute_unitary()))


def _perm(M):
    n = M.shape[0]
    return sum(np.prod([M[i, s[i]] for i in range(n)]) for s in permutations(range(n)))


def test_slos_matches_permanents():
    r = QuandelaPhotonicReservoir(4, 2, mesh_depth=3, seed=7).build()
    U = r.U_res @ r.U_pre                                   # bare reservoir of the 'phases' design
    P = r.run()
    inp = [i for i, v in enumerate(r.input_occupation) for _ in range(v)]
    for pat, p in zip(r.patterns, P):
        out = [i for i, v in enumerate(pat) for _ in range(v)]
        sub = U[np.ix_(out, inp)]
        norm = np.prod([factorial(v) for v in pat]) * np.prod([factorial(v) for v in r.input_occupation])
        assert p == pytest.approx(abs(_perm(sub)) ** 2 / norm, abs=1e-12)
    assert P.sum() == pytest.approx(1.0)


def test_first_moments_are_interference_blind():
    r = QuandelaPhotonicReservoir(8, 4, mesh_depth=4, seed=2).build()
    V = r.circuit_unitary([np.full(r.n_data, 1.1)])
    P = r.probabilities(V)
    assert np.allclose(r.features_from_probs(P, "first_moments")[0], r.first_moments_distinguishable(V))
    # two-body correlators DO depend on quantum interference (differ from distinguishable)
    corr = r.features_from_probs(P, "correlators")[0][8:]
    Vd = np.abs(V[:, r.input_modes]) ** 2
    dist = np.array([sum(Vd[i, a] * Vd[j, b] for a in range(4) for b in range(4) if a != b)
                     for i in range(8) for j in range(i + 1, 8)])
    assert not np.allclose(corr, dist, atol=1e-3)


def test_photonic_determinism():
    a = QuandelaPhotonicReservoir(12, 6, 4, seed=9).build()
    b = QuandelaPhotonicReservoir(12, 6, 4, seed=9).build()
    assert np.allclose(a.U_res, b.U_res)


def test_photonic_qrc_window_semantics():
    r = QuandelaPhotonicReservoir(6, 3, mesh_depth=2, seed=1).build()
    q = PhotonicQRC(r, window=3, alpha=1e-8, t_dismiss=5)
    seen = {}
    r_wf = r.window_features

    def spy(windows, **kw):
        seen["w"] = np.array(windows)
        return r_wf(windows, **kw)

    r.window_features = spy
    y = np.arange(1, 8, dtype=float).reshape(-1, 1) / 10
    q._compute_features(y)
    assert np.allclose(seen["w"][0], [0, 0, 0.1]) and np.allclose(seen["w"][3], [0.2, 0.3, 0.4])
    r.window_features = r_wf
    # inherited train/test path runs and returns one prediction per step
    from sklearn.preprocessing import MinMaxScaler
    s = np.sin(np.linspace(0, 12, 120)).reshape(-1, 1)
    sc = MinMaxScaler().fit(s[:80])
    q.train(y=sc.transform(s[:80]), scaler=sc)
    pred = q.test(y=np.clip(sc.transform(s[80:]), 0, 1), scaler=sc, retrain=True)
    assert pred.shape == (40, 1)


# ── ORS_LO consistency ────────────────────────────────────────────────────────
def test_ors_lo_single_photon_equals_qubit_haar_reference():
    # 1 photon in 16 modes through Haar U(16) is exactly Dirichlet(1,…,1) on 16 outcomes,
    # i.e. the qubit Haar ensemble for n=4 qubits → same fitted D_eff and Λ_ref.
    ph = photonic_lo_reference(16, 1, n_ref=3000, n_holdout=1500, seed=1)
    qb = qubit_haar_reference(4, n_ref=3000, n_holdout=1500, seed=2)
    assert ph.D_eff == pytest.approx(qb.D_eff, rel=0.08)
    assert ph.lambda_ref == pytest.approx(qb.lambda_ref, abs=0.1)


def test_ors_lo_reference_recovers_large_D_and_orders_meshes():
    qb = qubit_haar_reference(6, seed=3)
    assert qb.D_eff == pytest.approx(64, rel=0.06)
    ref = photonic_lo_reference(6, 3, n_ref=1500, n_holdout=800, seed=4)
    g = {}
    for D in (1, 6, "haar"):
        P = np.array([QuandelaPhotonicReservoir(6, 3, mesh_depth=6 if D == "haar" else D,
                                                haar=D == "haar", seed=50 + s).build().run() for s in range(40)])
        g[D] = ref.gap(P)
    assert abs(g["haar"]["gap"]) < 3 * g["haar"]["gap_sem"] + 0.3
    assert g[1]["gap"] < g[6]["gap"] < g["haar"]["gap"] + 3 * g["haar"]["gap_sem"]
    # the naive qubit ORS does NOT give zero for the maximally expressive photonic ensemble
    # (Exp4 configuration: 3 photons in 12 modes; measured ≈ -7.0 ± 0.4)
    P = np.array([QuandelaPhotonicReservoir(12, 3, 12, haar=True, seed=90 + s).build().run() for s in range(150)])
    assert naive_gate_ors_gap(P)["gap"] < -3.0


# ── R_eff identical to the original implementation ────────────────────────────
def test_effective_rank_matches_qelm_utils():
    from native_reservoirs.hardware_specs import REPO_ROOT
    spec = importlib.util.spec_from_file_location("qelm_utils", REPO_ROOT / "experiments/QELM/Fourier/qelm_utils.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    F = np.random.default_rng(0).normal(size=(50, 12)) @ np.diag(np.linspace(0.1, 3, 12))
    assert effective_rank(F) == pytest.approx(mod.effective_rank(F))


# ── Haar-dialled photonic mesh ────────────────────────────────────────────────
def test_haar_alpha_rule_matches_decomposition():
    from scipy.stats import unitary_group
    from native_reservoirs.quandela_photonic import haar_alpha, rectangular_decomposition
    m, rng, acc = 6, np.random.default_rng(0), {}
    for _ in range(3000):
        for c, k, phi in rectangular_decomposition(unitary_group.rvs(m, random_state=rng)):
            acc.setdefault((c, k), []).append(np.sin(phi / 2) ** 2)
    assert len(acc) == m * (m - 1) // 2 and all(c % 2 == k % 2 for c, k in acc)
    for (c, k), xs in acc.items():
        a_mle = -len(xs) / np.log1p(-np.clip(np.array(xs), 0, 1 - 1e-15)).sum()
        assert a_mle == pytest.approx(haar_alpha(m, c, k), rel=0.08)


def test_full_depth_haar_dialled_mesh_is_haar():
    from scipy.stats import beta, kstest
    m = 8
    U = np.array([QuandelaPhotonicReservoir(m, 4, mesh_depth=m, seed=s).build().U_res for s in range(1500)])
    for i, j in [(0, 0), (0, m - 1), (3, 4), (m - 1, 0)]:
        assert kstest(np.abs(U[:, i, j]) ** 2, beta(1, m - 1).cdf).pvalue > 1e-3
    assert np.mean(np.abs(np.trace(U, axis1=1, axis2=2)) ** 2) == pytest.approx(1.0, abs=0.12)
    Uu = np.array([QuandelaPhotonicReservoir(m, 4, mesh_depth=m, mesh_sampling="uniform", seed=s).build().U_res
                   for s in range(1500)])
    assert kstest(np.abs(Uu[:, 0, 0]) ** 2, beta(1, m - 1).cdf).pvalue < 1e-6     # uniform phases: not Haar


# ── windowed (measure-at-the-end) gate QRC ────────────────────────────────────
def test_gate_window_circuit_is_native_and_reproduces_features():
    from native_reservoirs.adapters import GateWindowFeatures
    for res in (IQMLatticeReservoir(6, 2, seed=1).build(), AQTIonTrapReservoir(6, 0.5, seed=1).build()):
        g = GateWindowFeatures(res.circuit, mode="reupload")
        row = np.array([0.25, 0.7])
        qc = g.window_circuit(row, measure=False)
        assert set(qc.count_ops()) <= set(res.NATIVE_OPS)
        sv = Statevector(qc)
        feats = np.array([sv.expectation_value(o).real for o in g._q.observables])
        assert np.allclose(feats, g.window_features(row[None])[0])


def test_reset_window_equals_truncated_exp3_dynamics():
    from native_reservoirs.adapters import GateWindowFeatures
    res = IQMLatticeReservoir(6, 2, seed=2).build()
    q = res.as_qrc()
    y = np.array([[0.1], [0.5], [0.9]])
    ref = q._compute_features_dm(y)[-1]                      # Exp3 dynamics from a fresh start
    got = GateWindowFeatures(res.circuit, mode="reset").window_features(y.T)[0]
    assert np.allclose(ref, got)


def test_classical_window_features():
    from native_reservoirs.adapters import ClassicalWindowFeatures
    X = ClassicalWindowFeatures("quadratic").window_features(np.array([[1.0, 2.0]]))
    assert np.allclose(X, [[2.0, 1.0, 4.0, 2.0, 1.0]])


# ── data ──────────────────────────────────────────────────────────────────────
def test_lih_training_subsets_keep_test_set():
    lih = tasks.load_lih()
    w6 = tasks.lih_configuration_weights(lih["ground_states"])["w6"]
    idx = tasks.lih_train_subset(20, bond_lengths=lih["bond_lengths"])
    tr, te = tasks.lih_split_indices(300)
    assert len(idx) == 20 and set(idx) <= set(tr)
    o = tasks.run_lih_readout(w6, lih["y"], train_idx=idx, bond_lengths=lih["bond_lengths"])
    R = lih["bond_lengths"]; inside = (R[te] >= R[idx].min()) & (R[te] <= R[idx].max())
    assert o["n_test"] == inside.sum() and o["rmse"] == pytest.approx(np.sqrt(o["mse"]))


@pytest.mark.skipif(not tasks.EMS_CSV.exists(), reason="per-EMS EMSIG export not present")
def test_ems_series_reproduces_exp3_split():
    s = tasks.load_ems_series()
    sp = tasks.energy_splits(s, N=5000)
    assert len(s) == 69781 and len(sp["y_train"]) == 3500 and len(sp["y_test"]) == 750


def test_report_formatting():
    from native_reservoirs.report import pm
    assert pm(-2.3691, 1.1652) == "-2.37 ± 1.17"
    assert pm(0.5102, None, "{:.3f}") == "0.510"
    assert pm(float("nan"), 0.1) == "—"



def test_phase_encoding_design():
    r = QuandelaPhotonicReservoir(12, 3, mesh_depth=3, seed=4).build()
    assert r.input_modes == [0, 4, 8] and len(r.patterns) == 364 and r.n_sites == 6
    th = np.linspace(0.1, 3.0, 6)
    assert np.allclose(np.array(r.full_circuit([th, th[::-1]]).compute_unitary()),
                       r.circuit_unitary([th, th[::-1]]))
    E = r.encoding_unitary(th)
    assert np.allclose(np.diag(E)[:6], np.exp(1j * th)) and np.allclose(np.diag(E)[6:], 1)
    assert r.resource_metrics()["phase_shifters_data"] == 6
    P = r.run(np.random.default_rng(0).uniform(0, 0.4, (4, 6)), task="lih", angle_map="arcsin_sqrt")
    assert np.allclose(P.sum(axis=1), 1)


def test_noisy_emulation_basis_rotations_and_counts():
    """Native R-gate basis rotations + counts → the same 1-local Paulis as the exact state."""
    from native_reservoirs import noisy_emulation as ne
    n, rng = 3, np.random.default_rng(3)
    qc = QuantumCircuit(n)
    for q in range(n):
        qc.r(*rng.uniform(0, np.pi, 2), q)
    qc.cz(0, 1); qc.rxx(0.7, 1, 2)
    for q in range(n):
        qc.r(*rng.uniform(0, np.pi, 2), q)
    psi = Statevector(qc).data
    counts = np.zeros((1, 3, 2 ** n))
    for s, b in enumerate(ne.SETTINGS):
        c = qc.copy()
        for name, theta, phi in ne.BASIS_OPS[b]:
            for q in range(n):
                c.r(theta, phi, q)
        counts[0, s] = Statevector(c).probabilities()
    assert np.allclose(ne.pauli1_from_counts(counts, n), ne.pauli1_exact(psi[None], n), atol=1e-12)


def test_noisy_emulation_calibrations():
    from native_reservoirs import noisy_emulation as ne
    rng = np.random.default_rng(0)
    X = rng.uniform(-1, 1, (50, 4))
    a, b = ne.calibrate(X, 0.8 * X + 0.05)
    assert np.allclose(a, 0.8) and np.allclose(b, 0.05)
    Xd = rng.uniform(0, 1, (50, 4))
    assert abs(ne.calibrate_mixture(X, Xd, 0.9 * X + 0.1 * Xd) - 0.9) < 1e-12
