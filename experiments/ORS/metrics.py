"""
metrics.py — Scalable complexity metrics for parametrized quantum circuit families.

Three complementary metrics:
  - Expressibility: Lorentz/majorization curves and Haar order-statistics log-likelihood.
  - Entanglement:   Rényi-2 entropy via classical shadows (U-statistic purity estimator).
  - Input sensitivity: Var_x[<O>_x] via classical shadows with bias correction.

Public API
----------
Circuit factory
    build_circuit(gates_name, nqbits, *, depth, expected_degree, n_layers,
                  scrambler_depth, haar_mode) -> Callable[[], QuantumCircuit]

Expressibility
    lorentz_curves(circuit_fn, nqbits, nshots, ns)
    log_pk(x, k, D)
    order_statistics_score(sorted_probs, D, K=50)

Entanglement
    renyi2_entropy(circuit_fn, nqbits, n_shadows, ns, *, subsystem_qubits,
                   subsystem_size, n_subsystems, use_statevector, chunk_size)

Input sensitivity
    shadow_observable_estimate(circuit, nqbits, obs_qubits, obs_paulis, n_shadows, ...)
    input_sensitivity(circuit_fn, inputs, nqbits, obs_qubits, obs_paulis, n_shadows, ...)

Design principles
-----------------
- Every metric function accepts a Callable[[], QuantumCircuit] (zero-arg factory), not
  a gates_name string. This fully decouples circuit construction from metric computation.
- For input_sensitivity, circuit_fn accepts a scalar float and returns a QuantumCircuit
  with that value already encoded.
- _sample_shadows is the single point where Qiskit simulation runs for shadow-based
  metrics. Both renyi2_entropy and shadow_observable_estimate call it and post-process
  the returned (bases_arr, bits_arr).
- No file I/O lives here; persistence helpers belong in the calling notebooks.
"""

# ── 0. Imports ───────────────────────────────────────────────────────────────

import itertools
import random
import warnings
from typing import Callable, List, Optional, Sequence, Union

import networkx as nx
import numpy as np
from tqdm import tqdm

from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import UnitaryGate, StatePreparation
from qiskit.circuit.random import random_circuit
from qiskit.quantum_info import random_unitary, random_statevector
from qiskit_aer import AerSimulator
from scipy.special import betaln
import mpmath as mp
from scipy.special import betaln, digamma

warnings.filterwarnings("ignore")


# ── 1. Internal shadow utilities ─────────────────────────────────────────────
# Basis rotation matrices for single-qubit Pauli measurements.
# Applied in numpy to batches of statevectors (statevector shadow mode).

_H_GATE = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
_HY_GATE = _H_GATE @ np.array([[1, 0], [0, -1j]], dtype=complex)   # H @ Sdg  (Y basis)


def _apply_gate_batch(sv_batch: np.ndarray, gate: np.ndarray,
                      qubit: int, nqbits: int) -> np.ndarray:
    """Apply a 2×2 gate to `qubit` of every statevector in sv_batch.

    Qiskit uses little-endian ordering: qubit 0 is the LSB of the state index.
    After reshaping sv_batch to (B, 2, 2, ..., 2), row-major order places
    qubit j at axis (nqbits - j):

        axes: (batch=0, q_{n-1}=1, q_{n-2}=2, ..., q_1=n-1, q_0=n)

    Args:
        sv_batch: Complex array of shape (B, 2^nqbits).
        gate:     2×2 complex matrix to apply.
        qubit:    Target qubit index (0-indexed, little-endian).
        nqbits:   Total number of qubits.

    Returns:
        Complex array of shape (B, 2^nqbits) after gate application.
    """
    B = sv_batch.shape[0]
    sv_nd = sv_batch.reshape([B] + [2] * nqbits)
    ax = nqbits - qubit                                    # correct axis for qubit j
    result = np.tensordot(gate, sv_nd, axes=[[1], [ax]])
    result = np.moveaxis(result, 0, ax)
    return result.reshape(B, 2 ** nqbits)


# ── 2. Circuit builders ───────────────────────────────────────────────────────

# ── Private helpers (each returns one fresh random instance per call) ─────────

def _circuit_D2(nqbits: int) -> QuantumCircuit:
    """IQP circuit: H^⊗n · (all-pairs 2-qubit diagonal UnitaryGate) · H^⊗n.

    Each pair (i,j) gets an independent random 4×4 diagonal unitary drawn
    from U[0, 2π] phases. Both phase and pairing order are re-sampled every call.
    """
    qubit_idx = list(range(nqbits))
    qubits_set = list(itertools.combinations(qubit_idx, 2))
    qubits_set = [qubits_set[i] for i in np.random.permutation(len(qubits_set))]
    phis = np.random.uniform(0, 2 * np.pi, size=(len(qubits_set), 4))

    qc = QuantumCircuit(nqbits)
    for i in range(nqbits):
        qc.h(i)
    for i, pair in enumerate(qubits_set):
        diag = np.diag(np.exp(1j * phis[i]))
        qc.append(UnitaryGate(diag), [pair[0], pair[1]])
    for i in range(nqbits):
        qc.h(i)
    return qc


def _circuit_D2_random(nqbits: int, expected_degree: float) -> QuantumCircuit:
    """IQP circuit on an Erdős–Rényi random graph: H^⊗n · (Rz + Rzz on edges) · H^⊗n.

    Graph topology and all phases are re-sampled on every call.
    """
    p = expected_degree / (nqbits - 1)
    G = nx.erdos_renyi_graph(nqbits, p)
    edges = list(G.edges())
    phis_1q = np.random.uniform(0, 2 * np.pi, size=nqbits)
    phis_2q = np.random.uniform(0, 2 * np.pi, size=max(len(edges), 1))

    qc = QuantumCircuit(nqbits)
    for i in range(nqbits):
        qc.h(i)
    for idx, phi in enumerate(phis_1q):
        qc.rz(2 * phi, idx)
    for (u, v), phi in zip(edges, phis_2q):
        qc.rzz(2 * phi, u, v)
    for i in range(nqbits):
        qc.h(i)
    return qc


def _circuit_G(nqbits: int, gate_set_name: str, depth: int) -> QuantumCircuit:
    """Random gate-sequence circuit from a named gate set.

    gate_set_name must be one of 'G1', 'G2', 'G3'.
    Applies nqbits * depth gates sampled uniformly from the gate set.
    """
    gate_sets = {
        "G1": ["CNOT", "H", "X"],
        "G2": ["CNOT", "H", "S"],
        "G3": ["CNOT", "H", "T"],
    }
    gates_set = gate_sets[gate_set_name]
    num_gates = nqbits * depth
    qubit_idx = list(range(nqbits))
    gate_idx = list(range(len(gates_set)))

    qc = QuantumCircuit(nqbits)
    for _ in range(num_gates):
        gate = gates_set[random.choice(gate_idx)]
        if gate == "CNOT":
            q1 = random.choice(qubit_idx)
            others = [q for q in qubit_idx if q != q1]
            q2 = random.choice(others)
            qc.cx(q1, q2)
        else:
            q = random.choice(qubit_idx)
            if gate == "H":
                qc.h(q)
            elif gate == "X":
                qc.x(q)
            elif gate == "S":
                qc.s(q)
            elif gate == "T":
                qc.t(q)
    return qc


def _apply_brickwork_scrambler(qc: QuantumCircuit, nqbits: int,
                                scrambler_depth: int) -> None:
    """Apply scrambler_depth brickwork layers of random 2-qubit Haar gates (in-place).

    1D open chain; even layers cover pairs (0,1),(2,3),...; odd layers (1,2),(3,4),...
    Each pair gets an independent fresh Haar-random 4×4 unitary.
    """
    for k in range(scrambler_depth):
        start = k % 2
        pairs = [(i, i + 1) for i in range(start, nqbits - 1, 2)]
        for q0, q1 in pairs:
            U = random_unitary(4).data
            qc.append(UnitaryGate(U), [q0, q1])


def _circuit_scrambling_ansatz(nqbits: int, n_layers: int,
                                scrambler_depth: int) -> QuantumCircuit:
    """Build one scrambling-ansatz instance (arXiv:2602.17281).

    Each layer: Rx(θ) + Rz(θ) on all qubits → brickwork Haar scrambler → Ry(θ) on all.
    All angles and Haar unitaries are re-sampled on every call.
    """
    qc = QuantumCircuit(nqbits)
    for _ in range(n_layers):
        for i in range(nqbits):
            qc.rx(np.random.uniform(0, 2 * np.pi), i)
            qc.rz(np.random.uniform(0, 2 * np.pi), i)
        _apply_brickwork_scrambler(qc, nqbits, scrambler_depth)
        for i in range(nqbits):
            qc.ry(np.random.uniform(0, 2 * np.pi), i)
    return qc


def _circuit_haar_statevector(nqbits: int) -> QuantumCircuit:
    """Haar-random circuit via random_statevector + StatePreparation.

    Produces an exactly Haar-random output state. Preferred for entanglement
    and input-sensitivity measurements. Re-samples on every call.
    """
    sv = random_statevector(2 ** nqbits).data
    qc = QuantumCircuit(nqbits)
    qc.append(StatePreparation(sv), range(nqbits))
    return qc


def _circuit_haar_random(nqbits: int, depth: int) -> QuantumCircuit:
    """Haar-random circuit via random_circuit (gate-sequence approximation).

    Approximates Haar measure via random gate sequences. Use for expressibility
    (Lorentz curves) to match the Expressivity measures.ipynb convention.
    """
    return random_circuit(nqbits, depth, measure=False)


# ── Public factory ────────────────────────────────────────────────────────────

def build_circuit(
    gates_name: str,
    nqbits: int,
    *,
    depth: int = 100,
    expected_degree: Optional[float] = None,
    n_layers: int = 1,
    scrambler_depth: int = 1,
    haar_mode: str = "statevector",
) -> Callable[[], QuantumCircuit]:
    """Return a zero-argument factory that builds one random circuit instance per call.

    The returned callable re-samples all random parameters on every invocation,
    so it can be passed directly as circuit_fn to any metric function.

    Args:
        gates_name: One of 'D2', 'D2_random', 'G1', 'G2', 'G3', 'Haar',
                    'scrambling_ansatz', 'identity'.
        nqbits: Number of qubits.
        depth: Gate depth for G-family and Haar(haar_mode='random_circuit').
        expected_degree: Mean graph degree for 'D2_random'. Required when
                         gates_name='D2_random'.
        n_layers: Ansatz layer count for 'scrambling_ansatz'.
        scrambler_depth: Brickwork depth K for 'scrambling_ansatz'.
        haar_mode: 'statevector' (default) — exact Haar state via random_statevector
                   + StatePreparation; best for entanglement/sensitivity metrics.
                   'random_circuit' — approximate Haar via random_circuit(nqbits, depth);
                   use for lorentz_curves to match Expressivity measures.ipynb convention.

    Returns:
        A Callable[[], QuantumCircuit]. Each call returns a fresh random instance.

    Haar mode note:
        'statevector' and 'random_circuit' produce different objects.
        random_circuit is not exactly Haar at any finite depth; random_statevector
        + StatePreparation is exact. For order-statistics and Lorentz comparisons,
        either is fine. For quantitative entanglement benchmarks, use 'statevector'.

    Example:
        fn = build_circuit('G2', nqbits=4, depth=100)
        qc = fn()   # one G2 random instance
    """
    gates_name = gates_name.strip()
    if gates_name == "D2":
        return lambda: _circuit_D2(nqbits)
    elif gates_name == "D2_random":
        if expected_degree is None:
            raise ValueError("expected_degree is required for 'D2_random'")
        return lambda: _circuit_D2_random(nqbits, expected_degree)
    elif gates_name in ("G1", "G2", "G3"):
        return lambda: _circuit_G(nqbits, gates_name, depth)
    elif gates_name == "Haar":
        if haar_mode == "statevector":
            return lambda: _circuit_haar_statevector(nqbits)
        elif haar_mode == "random_circuit":
            return lambda: _circuit_haar_random(nqbits, depth)
        else:
            raise ValueError(f"Unknown haar_mode '{haar_mode}'. Use 'statevector' or 'random_circuit'.")
    elif gates_name == "scrambling_ansatz":
        return lambda: _circuit_scrambling_ansatz(nqbits, n_layers, scrambler_depth)
    elif gates_name == "identity":
        return lambda: QuantumCircuit(nqbits)
    else:
        raise ValueError(
            f"Unknown gates_name '{gates_name}'. "
            "Valid options: 'D2', 'D2_random', 'G1', 'G2', 'G3', 'Haar', "
            "'scrambling_ansatz', 'identity'."
        )


# ── 3. Shadow sampling kernel ─────────────────────────────────────────────────

def _sample_shadows(
    circuit_fn: Callable[[], QuantumCircuit],
    nqbits: int,
    n_shadows: int,
    use_statevector: bool = True,
    chunk_size: int = 50,
) -> tuple:
    """Sample n_shadows classical shadows from one circuit instance.

    Builds ONE circuit via circuit_fn(), then draws n_shadows random single-qubit
    Pauli-basis measurements (0=Z, 1=X, 2=Y per qubit) and simulates outcomes.

    This is the shared simulation kernel for renyi2_entropy and
    shadow_observable_estimate. Factoring it out means Qiskit runs exactly once
    per circuit instance regardless of how many downstream metrics are computed.

    Two execution modes:
    - Statevector mode (use_statevector=True, default):
        One AerSimulator statevector run → all n_shadows outcomes sampled in numpy
        via inverse-CDF. ~100–500× faster than hardware mode. Simulation only.
    - Hardware mode (use_statevector=False):
        Transpile base circuit once → append basis-rotation gates + measure_all for
        each shadow → submit n_shadows 1-shot circuits in a single batched job.
        Works on any Qiskit-compatible backend.

    Args:
        circuit_fn: Zero-arg callable returning a QuantumCircuit (no measurements).
        nqbits: Number of qubits.
        n_shadows: Number of shadow measurements to collect.
        use_statevector: True = fast numpy mode; False = hardware-compatible.
        chunk_size: Batch size for statevector numpy sampling (statevector mode only).
                    50 keeps peak memory ≈ 52 MB for nqbits=16.

    Returns:
        bases_arr: int ndarray, shape (n_shadows, nqbits), values in {0,1,2} (Z/X/Y).
        bits_arr:  int8 ndarray, shape (n_shadows, nqbits), values in {0,1}.
    """
    qc = circuit_fn()
    bases_arr = np.random.randint(0, 3, size=(n_shadows, nqbits))

    if use_statevector:
        simulator = AerSimulator(method="statevector")
        sv_qc = qc.copy()
        sv_qc.save_statevector()
        sv_qc = transpile(sv_qc, simulator)
        sv = np.array(simulator.run(sv_qc).result().get_statevector())  # (2^n,) complex

        bits_arr = np.empty((n_shadows, nqbits), dtype=np.int8)
        for start in range(0, n_shadows, chunk_size):
            end = min(start + chunk_size, n_shadows)
            B = end - start
            bases_chunk = bases_arr[start:end]           # (B, nqbits)
            sv_batch = np.tile(sv, (B, 1))               # (B, 2^nqbits) complex

            # Apply basis-rotation gates in numpy (no Qiskit overhead per shadow)
            for j in range(nqbits):
                x_mask = bases_chunk[:, j] == 1           # X basis → apply H
                y_mask = bases_chunk[:, j] == 2           # Y basis → apply H·Sdg
                if x_mask.any():
                    sv_batch[x_mask] = _apply_gate_batch(
                        sv_batch[x_mask], _H_GATE, j, nqbits)
                if y_mask.any():
                    sv_batch[y_mask] = _apply_gate_batch(
                        sv_batch[y_mask], _HY_GATE, j, nqbits)

            # Inverse-CDF sampling: one outcome per shadow
            probs = np.abs(sv_batch) ** 2
            probs /= probs.sum(axis=1, keepdims=True)    # normalise (guard float drift)
            cumprobs = np.cumsum(probs, axis=1)
            rand_v = np.random.random((B, 1))
            sample_idx = (cumprobs < rand_v).sum(axis=1)  # (B,) int in [0, 2^nqbits)

            # Little-endian bit extraction: qubit j = bit j of sample_idx
            shifts = np.arange(nqbits)
            bits_arr[start:end] = (
                (sample_idx[:, None] >> shifts[None, :]) & 1
            ).astype(np.int8)

    else:
        # Hardware-compatible mode: transpile once, append rotations, batch 1-shot jobs
        simulator = AerSimulator()
        transpiled_base = transpile(qc, simulator)
        shadow_circuits = []
        for t in range(n_shadows):
            qc_s = transpiled_base.copy()
            for j in range(nqbits):
                if bases_arr[t, j] == 1:        # X basis: H
                    qc_s.h(j)
                elif bases_arr[t, j] == 2:      # Y basis: Sdg then H
                    qc_s.sdg(j)
                    qc_s.h(j)
            qc_s.measure_all()
            shadow_circuits.append(qc_s)

        job_result = simulator.run(shadow_circuits, shots=1, memory=True).result()
        # Bitstrings from Qiskit are big-endian (qubit 0 = rightmost char); reverse each
        all_bits = [job_result.get_memory(t)[0] for t in range(n_shadows)]
        bits_arr = (
            np.frombuffer(
                "".join(bs[::-1] for bs in all_bits).encode(), dtype=np.uint8
            ).reshape(n_shadows, nqbits) - 48
        ).astype(np.int8)

    return bases_arr, bits_arr


# ── 4. Expressibility metrics ─────────────────────────────────────────────────

def lorentz_curves(
    circuit_fn: Callable[[], QuantumCircuit],
    nqbits: int,
    nshots: int,
    ns: int,
    statevector = True,
    *,
    verbose: bool = True,
) -> tuple:
    """Compute Lorentz curves for ns independent random circuit instances.

    Samples the measurement output distribution of each circuit instance and
    forms the Lorenz (Lorentz) curve — the cumulative sum of the sorted
    probability vector. Lower std-dev across instances indicates behavior
    closer to Haar-random (Porter–Thomas distribution).

    Extracted from get_lorentz_curves in Expressivity measures.ipynb and
    decoupled from circuit construction.

    Args:
        circuit_fn: Zero-arg callable returning a QuantumCircuit. Called ns times.
        nqbits: Number of qubits.
        nshots: Measurement shots per instance (trade-off: more shots → less noise).
        ns: Number of independent circuit instances.

    Returns:
        curves_list: List of ns ndarrays, each shape (2^nqbits,).
                     curves_list[i][k] = sum of the k+1 largest probabilities.
        sorted_probs: ndarray shape (ns, 2^nqbits), row i = descending prob vector.
        last_qc: The last QuantumCircuit built (for inspection).
    """
    simulator = AerSimulator()
    curves_list = []
    sorted_probs_list = []
    last_qc = None

    for _ in tqdm(range(ns), desc="lorentz_curves", disable=not verbose):
        qc = circuit_fn()
        last_qc = qc
        qc_m = qc.copy()
        #qc_m.measure_all()
        transpiled = transpile(qc_m, simulator)
        if statevector:
            transpiled.save_statevector()
            sv = np.array(simulator.run(transpiled).result().get_statevector())  #
            p = np.abs(sv) ** 2
        else:
            transpiled.measure_all()
            counts = simulator.run(transpiled, shots=nshots).result().get_counts()
            # Full probability vector over all 2^nqbits basis states (0-pad unobserved)
            p = np.array(
                  [counts.get(format(k, f"0{nqbits}b"), 0) for k in range(2**nqbits)],
                dtype=float,) / nshots

        p_sorted = np.sort(p)[::-1]        # descending
        curves_list.append(np.cumsum(p_sorted))
        sorted_probs_list.append(p_sorted)

    return curves_list, np.array(sorted_probs_list), last_qc


def log_pk(x: float, k: int, D: int, eps: float = 1e-300) -> float:
    """Log of the Haar order-statistic density for the k-th largest probability.

    Analytical density (Micklitz 2025, arXiv:2510.13026):
        P_k(x) ∝ exp(-k·a) · (1 - exp(-a))^(D-k),   a = (D-2)·x

    Computed in log-space for numerical stability.

    Args:
        x: Observed probability value. Must be in (0, 1).
        k: Rank (1-indexed: k=1 is the largest probability).
        D: Hilbert space dimension (2^nqbits).

    Returns:
        log P_k(x), or -inf if x is outside (0, 1).
    """
    if D <= 2:
        raise ValueError("D must be > 2")
    if not (1 <= k <= D):
        raise ValueError("k must satisfy 1 <= k <= D")
    if not np.isfinite(x):
        return -np.inf

    x = np.clip(x, eps, 1.0 - eps)

    A = D - 2
    y = A * x

    log_norm = np.log(A) - betaln(k, D - k + 1)
    log_tail = np.log1p(-np.exp(-y))

    return log_norm - k * y + (D - k) * log_tail


def order_statistics_score(
    sorted_probs: np.ndarray,
    D: int,
    K: int = 50,
) -> np.ndarray:
    """Haar order-statistics log-likelihood scores for ns circuit instances.

    For each instance s:
        score[s] = Σ_{k=1}^{K}  log_pk(sorted_probs[s, k-1], k, D) / k

    A score closer to the Haar mean indicates behavior closer to Haar-random.
    Lower (more negative) score relative to the Haar mean indicates a more
    structured, non-random family.

    Args:
        sorted_probs: ndarray shape (ns, D) — descending sorted probability vectors.
        D: Hilbert space dimension (2^nqbits). Must equal sorted_probs.shape[1].
        K: Number of top probabilities to include (K << D for scalability).

    Returns:
        scores: ndarray shape (ns,).

    Important — K vs distribution support:
        If K exceeds the number of non-zero probabilities in a sample (e.g. Clifford /
        stabilizer states on n qubits have exactly 2^(n/2) non-zero entries), the terms
        for zero-probability ranks return -inf, making the entire score -inf. This is
        the same behaviour as the original log_score_family in Expressivity measures.ipynb.

        To get finite, comparable scores, set K ≤ min(support_size_of_all_families).
        For Clifford states on n qubits: support_size = 2^(n//2), so K must satisfy
        K ≤ 2^(n//2). Example: n=4 → K ≤ 4; n=8 → K ≤ 16; n=16 → K ≤ 256.

        The gap formula (mean_Haar - mean_family) / K only makes sense when both
        families contribute the same number of finite terms, i.e. when K is within
        the support of the smallest-support family being compared.
    """
    ns = sorted_probs.shape[0]
    scores = np.zeros(ns)
    for s in range(ns):
        xvec = sorted_probs[s, :K]
        xvec = xvec[xvec > 0]   # filter zero probabilities to avoid -inf in log_pk
        K_eff = len(xvec)          # adjust K to actual number of non-zero probabilities
        sval = sum(log_pk(xvec[k - 1], k, D) / k for k in range(1, K_eff + 1))

        while not np.isfinite(sval):
            K_eff -=1
            sval = sum(log_pk(xvec[k - 1], k, D) / k for k in range(1, K_eff + 1))
        scores[s] = sval
    return scores

# ------------------------------------------------------------------
# Large-D analytical Haar baseline for the ORS score
#
# P_k(x) =
#   a / B(k, D-k+1)
#   * exp(-k a x)
#   * (1 - exp(-a x))^(D-k)
#
# with a = D - 2
#
# We compute:
#
# Lambda_Haar =
#   Σ_k w_k ∫ P_k(x) log P_k(x) dx
#
# using the analytical formula derived in the paper.
# ------------------------------------------------------------------

def lambda_k_haar_largeD(k, D):
    """
    Analytical Haar self-log-likelihood contribution for rank k.

    Computes:
        ∫ P_k(x) log P_k(x) dx

    using the large-D approximation.

    Parameters
    ----------
    k : int
        Rank index (1 <= k <= K)

    D : int
        Hilbert-space dimension = 2^n

    Returns
    -------
    float
    """

    a = D - 2

    # log normalization term
    log_norm = np.log(a) - betaln(k, D - k + 1)

    # expectation of log(u)
    term_u = k * (digamma(k) - digamma(D + 1))

    # expectation of log(1-u)
    term_1mu = (D - k) * (
        digamma(D - k + 1) - digamma(D + 1)
    )

    return log_norm + term_u + term_1mu


def lambda_haar_largeD(D, K):
    """
    Full analytical Haar baseline for the ORS score.

    Uses weights:
        w_k = 1/k

    Computes:
        Lambda_Haar = Σ_k (1/k) * Lambda_k

    Parameters
    ----------
    D : int
        Hilbert-space dimension

    K : int
        Number of top ranks used in the ORS score

    Returns
    -------
    float
    """

    total = 0.0

    for k in range(1, K + 1):
        wk = 1.0 / k
        total += wk * lambda_k_haar_largeD(k, D)

    return total

###############################
# Noisy results
###############################

def apply_global_depolarizing(p_ideal: np.ndarray, f: float, D: int) -> np.ndarray:
    """Apply the global depolarizing channel at the probability level:
        p_noisy[k] = f * p_ideal[k] + (1 - f) / D
    """
    if not (0.0 < f <= 1.0):
        raise ValueError("Fidelity must be in (0, 1].")
    return f * p_ideal + (1.0 - f) / D


def lorentz_curves_noisy(
    circuit_fn,
    nqbits: int,
    nshots: int,
    ns: int,
    fidelity: float = 1.0,
    statevector: bool = True,
    *,
    verbose: bool = True,
) -> tuple:
    """Like lorentz_curves, but applies a global depolarizing channel with the
    given fidelity to each output distribution before sampling/sorting.

    Args:
        fidelity: f in (0, 1]. f=1 recovers the noiseless case.
        statevector: if True, applies noise analytically to exact probabilities;
                     if False, applies noise then samples shots from p_noisy.
    """
    simulator = AerSimulator()
    D = 2 ** nqbits
    curves_list = []
    sorted_probs_list = []
    last_qc = None

    for _ in tqdm(range(ns), desc="lorentz_curves_noisy", disable=not verbose):
        qc = circuit_fn()
        last_qc = qc
        qc_sv = qc.copy()
        qc_sv_t = transpile(qc_sv, simulator)
        qc_sv_t.save_statevector()
        sv = np.array(simulator.run(qc_sv_t).result().get_statevector())
        p_ideal = np.abs(sv) ** 2
        p_noisy = apply_global_depolarizing(p_ideal, fidelity, D)

        if statevector:
            p = p_noisy
        else:
            # Sample shots from the noisy distribution
            counts_array = np.random.multinomial(nshots, p_noisy)
            p = counts_array / nshots

        p_sorted = np.sort(p)[::-1]
        curves_list.append(np.cumsum(p_sorted))
        sorted_probs_list.append(p_sorted)

    return curves_list, np.array(sorted_probs_list), last_qc

def log_pk_noisy(y: float, k: int, D: int, f: float, eps: float = 1e-300) -> float:
    """Log Haar order-statistic density under global depolarizing noise.

        log P_k^noisy(y; f) = log P_k^Haar(x) - log f
        where x = (y - (1-f)/D) / f

    Returns -inf if y is outside the noisy support.
    """
    if not (0.0 < f <= 1.0):
        raise ValueError("Fidelity must be in (0, 1].")
    if f == 1.0:
        return log_pk(y, k, D)

    x = (y - (1.0 - f) / D) / f
    if (x <= 0.0) or (x >= 1.0):
        return -np.inf

    return log_pk(x, k, D) - np.log(f)


def order_statistics_score_noisy(
    sorted_probs: np.ndarray,
    D: int,
    K: int,
    f: float,
) -> np.ndarray:
    """Λ_s under depolarizing noise, evaluated with known fidelity f.

    Same shrink-K_eff fallback as the noiseless version: if the score is -inf
    because some y_(k) falls below the noise floor (1-f)/D, drop the offending
    ranks until the score is finite.
    """
    ns = sorted_probs.shape[0]
    scores = np.zeros(ns)
    noise_floor = (1.0 - f) / D

    for s in range(ns):
        yvec = sorted_probs[s, :K]
        # Filter values at or below the noise floor (these give x <= 0).
        yvec = yvec[yvec > noise_floor]
        K_eff = len(yvec)
        sval = sum(log_pk_noisy(yvec[k - 1], k, D, f) / k for k in range(1, K_eff + 1))

        while not np.isfinite(sval) and K_eff > 0:
            K_eff -= 1
            sval = sum(log_pk_noisy(yvec[k - 1], k, D, f) / k for k in range(1, K_eff + 1))

        scores[s] = sval if K_eff > 0 else -np.inf
    return scores


def lambda_haar_largeD_noisy(D: int, K: int, f: float) -> float:
    """Analytical Haar baseline for the ORS score under depolarizing noise.

        Λ^Haar,noisy(f) = Λ^Haar - H_K * log(f)

    where H_K = Σ_{k=1}^K 1/k.
    """
    if not (0.0 < f <= 1.0):
        raise ValueError("Fidelity must be in (0, 1].")
    H_K = sum(1.0 / k for k in range(1, K + 1))
    return lambda_haar_largeD(D, K) - H_K * np.log(f)

mp.mp.dps = 80  # high precision

def Pk_exact_mp(x, k, D):
    x = mp.mpf(x)

    if x <= 0 or x >= 1/k:
        return mp.mpf("0")

    jmax = min(D, int(mp.floor(1/x)))

    total = mp.mpf("0")
    for j in range(k, jmax + 1):
        total += (
            mp.binomial(D-k, j-k)
            * ((-1) ** j)
            * (1 - j*x) ** (D-2)
        )

    norm = ((-1) ** k) * (D - 1) / mp.beta(k, D-k+1)

    return norm * total

def lambda_k_exact_mp(k, D):
    def integrand(x):
        p = Pk_exact_mp(x, k, D)
        if p <= 0:
            return mp.mpf("0")
        return p * mp.log(p)

    total = mp.mpf("0")

    # intervals: [1/(j+1), 1/j], where jmax is constant
    for j in range(k, D):
        a = mp.mpf(1) / (j + 1)
        b = mp.mpf(1) / j
        total += mp.quad(integrand, [a, b])

    # final small interval [0, 1/D]
    total += mp.quad(integrand, [mp.mpf("0"), mp.mpf(1) / D])

    return total


def lambda_haar_exact(D, K):
    total = mp.mpf("0")
    for k in range(1, K + 1):
        total += lambda_k_exact_mp(k, D) / k
    return float(total)

# ── 5. Entanglement metric ────────────────────────────────────────────────────

def renyi2_entropy(
    circuit_fn: Callable[[], QuantumCircuit],
    nqbits: int,
    n_shadows: int,
    ns: int,
    *,
    subsystem_qubits: Optional[Union[list, str]] = None,
    subsystem_size: Optional[int] = None,
    n_subsystems: int = 1,
    use_statevector: bool = True,
    chunk_size: int = 50,
    verbose: bool = True,
) -> tuple:
    """Estimate Rényi-2 entanglement entropy via classical shadows.

    Uses the unbiased U-statistic purity estimator (Huang, Kueng & Preskill 2020):

        P̂_A = (1 / N(N-1)) · Σ_{t≠t'} Π_{j∈A} factor(j, t, t')

    where the per-qubit factor is:
        5.0   if same basis and same bit
       -4.0   if same basis and different bit
        0.5   if different basis

    This estimator avoids the bias of the naive Tr(ρ̂²) from a single snapshot.
    The Rényi-2 entropy is then S2(A) = -log(max(P̂_A, 1e-10)).

    Haar baseline: s2_haar = -log((dA + dB) / (dA·dB + 1))
                   dA = 2^k, dB = 2^(nqbits-k).

    Shadow norm (Huang et al.): Std[P̂_A] ≈ 2^k / N. Use k=2 with N≥500
    for practical accuracy; k≥5 with N≤1000 is unreliable.

    Args:
        circuit_fn: Zero-arg callable. Called ns times independently.
        nqbits: Number of qubits.
        n_shadows: Shadow measurements per instance.
        ns: Number of independent circuit instances.
        subsystem_qubits: Subsystem A:
            None → use first subsystem_size qubits [0, ..., k-1].
            list of ints → those specific qubits.
            'random' → sample k random qubits; averaged over n_subsystems draws.
        subsystem_size: k. Required when subsystem_qubits is None or 'random'.
                        Defaults to nqbits // 2.
        n_subsystems: Number of random subsystem draws to average per instance.
                      Active only when subsystem_qubits='random'. Free: reuses shadows.
        use_statevector: See _sample_shadows.
        chunk_size: See _sample_shadows.

    Returns:
        s2_estimates: ndarray shape (ns,) — Rényi-2 entropy Ŝ2(A) per instance.
        delta_s: ndarray shape (ns,) — ΔS = s2_haar - Ŝ2(A).
                 Positive → circuit has less entanglement than Haar.
                 Near zero → indistinguishable from Haar at state level.
        s2_haar: float — analytical Haar baseline.
    """
    # Resolve subsystem
    if isinstance(subsystem_qubits, (list, tuple)):
        fixed_sub = list(subsystem_qubits)
        k = len(fixed_sub)
        n_subsystems = 1
    elif subsystem_qubits == "random":
        k = subsystem_size if subsystem_size is not None else nqbits // 2
        fixed_sub = None
    else:
        k = subsystem_size if subsystem_size is not None else nqbits // 2
        fixed_sub = list(range(k))
        n_subsystems = 1

    dA, dB = 2 ** k, 2 ** (nqbits - k)
    s2_haar = float(-np.log((dA + dB) / (dA * dB + 1)))

    rng = np.random.default_rng()
    s2_estimates = []

    for _ in tqdm(range(ns), desc="renyi2_entropy", disable=not verbose):
        bases_arr, bits_arr = _sample_shadows(
            circuit_fn, nqbits, n_shadows,
            use_statevector=use_statevector, chunk_size=chunk_size,
        )
        N = n_shadows
        sub_s2_list = []

        for _ in range(n_subsystems):
            sub_A = (
                sorted(rng.choice(nqbits, k, replace=False).tolist())
                if fixed_sub is None
                else fixed_sub
            )
            bases_sub = bases_arr[:, sub_A]   # (N, k)
            bits_sub = bits_arr[:, sub_A]     # (N, k)

            same_basis = bases_sub[:, None, :] == bases_sub[None, :, :]   # (N,N,k)
            same_bit = bits_sub[:, None, :] == bits_sub[None, :, :]       # (N,N,k)
            factors = np.where(same_basis, np.where(same_bit, 5.0, -4.0), 0.5)
            prod_f = np.prod(factors, axis=2)                              # (N,N)
            np.fill_diagonal(prod_f, 0.0)
            purity_est = np.sum(prod_f) / (N * (N - 1))
            sub_s2_list.append(-np.log(max(purity_est, 1e-10)))

        s2_estimates.append(float(np.mean(sub_s2_list)))

    s2_estimates = np.array(s2_estimates)
    delta_s = s2_haar - s2_estimates
    return s2_estimates, delta_s, s2_haar


# ── 6. Input sensitivity metric ───────────────────────────────────────────────

def shadow_observable_estimate(
    circuit: QuantumCircuit,
    nqbits: int,
    obs_qubits: Sequence[int],
    obs_paulis: Sequence[int],
    n_shadows: int,
    use_statevector: bool = True,
    chunk_size: int = 50,
) -> tuple:
    """Estimate <O> and its shadow noise for a fixed circuit and Pauli observable.

    The observable is O = ⊗_{j} P_j for j in obs_qubits, where P_j ∈ {Z=0, X=1, Y=2}.

    Shadow estimator (per shot t):
        contribution_t = Π_j  [3·(1 - 2·bit[t,j])   if basis[t,j] == P_j
                                0                      otherwise]

    The product is nonzero only when ALL qubits in obs_qubits have their
    measurement basis matching the required Pauli (probability (1/3)^k per shot).
    The mean over N shots is an unbiased estimator of <O>.

    Shadow noise magnitude: for a single-qubit Z observable, per-shot values are
    in {-3, 0, 3} (prob 1/3 for ±3, prob 2/3 for 0), giving Var[per-shot] ≈ 3.
    Over N shots: shadow_variance ≈ 3 / N. For k-qubit Pauli products: ~6^k / N.

    Args:
        circuit: QuantumCircuit with x already encoded (not called again here).
        nqbits: Total number of qubits in the circuit.
        obs_qubits: Qubit indices in the observable support, e.g. [0] for Z_0.
        obs_paulis: Pauli type per qubit in obs_qubits (same length):
                    0=Z, 1=X, 2=Y.
        n_shadows: Number of shadow shots.
        use_statevector: See _sample_shadows.
        chunk_size: See _sample_shadows.

    Returns:
        mean_estimate: float — unbiased estimate of <O>.
        shadow_variance: float — estimated Var[mean_estimate] = Var[per_shot] / N.
                         Use this as σ²_shadow(x) in the bias correction of
                         input_sensitivity.
    """
    obs_qubits = list(obs_qubits)
    obs_paulis = list(obs_paulis)

    bases_arr, bits_arr = _sample_shadows(
        lambda: circuit.copy(), nqbits, n_shadows,
        use_statevector=use_statevector, chunk_size=chunk_size,
    )

    # Build per-shot Pauli product estimate
    per_shot = np.ones(n_shadows, dtype=float)
    for qubit, pauli in zip(obs_qubits, obs_paulis):
        basis_match = bases_arr[:, qubit] == pauli           # (n_shadows,) bool
        signs = 1.0 - 2.0 * bits_arr[:, qubit].astype(float)  # +1 (bit=0) or -1 (bit=1)
        factor = np.where(basis_match, 3.0 * signs, 0.0)
        per_shot *= factor

    mean_estimate = float(np.mean(per_shot))
    # Variance of the mean (ddof=1 for unbiased estimate)
    shadow_variance = float(np.var(per_shot, ddof=1) / n_shadows)
    return mean_estimate, shadow_variance


def input_sensitivity(
    circuit_fn: Callable[[float], QuantumCircuit],
    inputs: np.ndarray,
    nqbits: int,
    obs_qubits: Sequence[int],
    obs_paulis: Sequence[int],
    n_shadows: int,
    use_statevector: bool = True,
    verbose: bool = True,
) -> tuple:
    """Estimate Var_x[<O>_x] via classical shadows with shadow-noise bias correction.

    Computes equation (5) from the research plan:

        Var_corrected = sample_var(per_input_means) - mean(per_input_shadow_vars)

    The raw sample variance of the per-input shadow estimates overestimates the
    true input sensitivity because each estimate has finite-sample noise. The
    bias correction subtracts the mean per-input shadow variance, which removes
    this noise contribution in expectation:

        E[sample_var(means)] = Var_x[<O>_x] + mean(Var_shadow[<O>_hat | x])

    Note: var_corrected can be slightly negative (consistent with zero sensitivity).
    Do NOT clamp to zero — that would introduce bias.

    Args:
        circuit_fn: Callable[[float], QuantumCircuit] — takes a scalar input x and
                    returns a QuantumCircuit with x already encoded (e.g. via Rx(x)).
                    Called once per element of inputs.
        inputs: ndarray shape (n_inputs,) — the scalar input values {x_i}.
                For a uniform grid: np.linspace(0, 2*pi, M, endpoint=False).
        nqbits: Number of qubits.
        obs_qubits: Qubit indices for observable O.
        obs_paulis: Pauli type per qubit (0=Z, 1=X, 2=Y), same length as obs_qubits.
        n_shadows: Shadow shots per input value.
        use_statevector: See _sample_shadows.

    Returns:
        var_corrected: float — bias-corrected Var_x[<O>_x].
        var_raw: float — uncorrected sample variance of per_input_means.
        per_input_estimates: ndarray shape (n_inputs,) — <O>_x estimate per input.
                             Useful for plotting <O>_x vs x and for FFT analysis.
    """
    inputs = np.asarray(inputs, dtype=float)
    n_inputs = len(inputs)
    per_input_means = np.empty(n_inputs)
    per_input_shadow_vars = np.empty(n_inputs)

    for i, x in enumerate(tqdm(inputs, desc="input_sensitivity", disable=not verbose)):
        qc = circuit_fn(x)
        mean_est, shadow_var = shadow_observable_estimate(
            qc, nqbits, obs_qubits, obs_paulis,
            n_shadows, use_statevector=use_statevector,
        )
        per_input_means[i] = mean_est
        per_input_shadow_vars[i] = shadow_var

    var_raw = float(np.var(per_input_means, ddof=1))
    mean_shadow_var = float(np.mean(per_input_shadow_vars))
    var_corrected = var_raw - mean_shadow_var

    return var_corrected, var_raw, per_input_means
