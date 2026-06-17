import itertools
import numpy as np
import random
import networkx as nx

from scipy.linalg import expm
from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp, random_unitary, Statevector, Operator
from qiskit.circuit.library import UnitaryGate, DiagonalGate
from qiskit_aer import AerSimulator
from itertools import combinations, product
from scipy.stats import unitary_group


class QuantumCircQiskit:
    def __init__(self, gates_name, num_gates=50, nqbits=8, observables_type=[1],
                 degree=None, n_layers=1, scrambler_depth=1, backend=None,
                 statevector=True, nshots=1024, encoding='quantum',
                 k_exp=2, theta_scale=np.pi, cache_threshold=9,
                 cache_reservoir_threshold=12,
                 # ── Ising parameters ─────────────────────────
                 h_field=10.0, W_disorder=1e-2, J_s=1.0, dt=10.0):
        """
        Parameters
        ----------
        cache_threshold : int
            Cache dense observable matrices for nqbits <= this value.
        cache_reservoir_threshold : int
            Cache the reservoir as a unitary matrix for nqbits <= this value.
            Enables fast matvec evaluation instead of per-input gate simulation.
        """
        self.gates_name = gates_name
        self.num_gates = num_gates
        self.nqbits = nqbits
        self.statevector = statevector
        self.nshots = nshots
        self.encoding = encoding
        self.k_exp = k_exp
        self.theta_scale = theta_scale
        self.observables_type = observables_type
        self.cache_threshold = cache_threshold
        self.cache_reservoir_threshold = cache_reservoir_threshold

        # ── Ising parameters ─────────────────────────
        self.h_field = h_field
        self.W_disorder = W_disorder
        self.J_s = J_s
        self.dt = dt

        # Build reservoir ONCE
        self.reservoir_circuit = self._build_reservoir(degree, n_layers, scrambler_depth)

        # Build observables ONCE and cache their matrices
        self.observables = self._build_observables()

        # Cache dense observable matrices only for small systems
        if self.nqbits <= self.cache_threshold:
            self._obs_mats = np.array([obs.to_matrix() for obs in self.observables])
        else:
            self._obs_mats = None   # use sparse Pauli path

        # Cache the reservoir unitary as a matrix for fast matvec evaluation.
        # This is the key optimization: instead of re-simulating the reservoir
        # gate-by-gate for every input, we apply U_res as a single matvec.
        if self.statevector and self.nqbits <= self.cache_reservoir_threshold:
            self._U_res = Operator(self.reservoir_circuit).data
        else:
            self._U_res = None

        # Backend setup (only needed for shots/hardware path)
        self.backend = backend
        self._estimator = None  # lazily instantiated

    @property
    def estimator(self):
        """Lazy estimator instantiation - only built when shots-mode is used."""
        if self._estimator is None:
            from qiskit_ibm_runtime import EstimatorV2 as Estimator
            backend = self.backend if self.backend is not None else AerSimulator()
            self._estimator = Estimator(backend)
        return self._estimator

    # ─────────────────────────────────────────────────────────────────────
    # Reservoir construction (dispatched once at init)
    # ─────────────────────────────────────────────────────────────────────
    def _build_reservoir(self, degree, n_layers, scrambler_depth):
        qc = QuantumCircuit(self.nqbits)
        name = self.gates_name

        if name in ('G1', 'G2', 'G3'):
            gate_sets = {'G1': ['CNOT', 'H', 'X'],
                         'G2': ['CNOT', 'H', 'S'],
                         'G3': ['CNOT', 'H', 'T']}
            self._build_G_gates(qc, gate_sets[name])

        elif name == 'D2':
            self._build_Dk(qc, k=2)

        elif name == 'D3':
            self._build_Dk(qc, k=3)

        elif name == 'Dn':
            phis = np.random.uniform(0, 2 * np.pi, size=2 ** self.nqbits)
            qc.append(DiagonalGate(np.exp(1j * phis).tolist()),
                      list(range(self.nqbits)))

        elif name == 'D2_random':
            if degree is None:
                raise ValueError("'degree' must be provided for D2_random")
            self._build_D2_random(qc, degree)
            
        elif name == 'D2_random_XZ':
            if degree is None:
                raise ValueError("'degree' must be provided for D2_random_XZ")
            self._build_D2_random_XZ(qc, degree, n_layers)
        
        elif name == 'D2_random_XZY':
            if degree is None:
                raise ValueError("'degree' must be provided for D2_random_XZY")
            self._build_D2_random_XZY(qc, degree, n_layers)

        elif name == 'scrambling_ansatz':
            self._build_scrambling_ansatz(qc, n_layers, scrambler_depth)

        elif name == 'RandomRotation':
            self._build_random_rotation(qc, n_layers=10)

        elif name == 'Haar':
            U = random_unitary(2 ** self.nqbits).data
            qc.append(UnitaryGate(U), list(range(self.nqbits)))

        elif name == 'MG':
            self._build_matchgates(qc)
        elif name == 'symmetric_Ising':
            self._build_symmetric_ising(qc)

        else:
            raise ValueError(f"Unknown gates_name: '{name}'")

        return qc

    # ─── Individual reservoir builders ──────────────────────────────────
    def _build_G_gates(self, qc, gates):
        qubit_idx = list(range(self.nqbits))
        for _ in range(self.num_gates):
            gate = random.choice(gates)
            if gate == 'CNOT':
                q1, q2 = random.sample(qubit_idx, 2)
                qc.cx(q1, q2)
            else:
                q = random.choice(qubit_idx)
                {'X': qc.x, 'S': qc.s, 'H': qc.h, 'T': qc.t}[gate](q)

    def _build_Dk(self, qc, k):
        qubit_idx = list(range(self.nqbits))
        pairs = list(itertools.combinations(qubit_idx, k))
        phis = np.random.uniform(0, 2 * np.pi, size=(len(pairs), 2 ** k))
        for group, phi_vec in zip(pairs, phis):
            diag = np.diag(np.exp(1j * phi_vec))
            qc.append(UnitaryGate(diag), list(group))

    def _build_D2_random_layer(self, qc, basis):
        """One D2_random commuting layer in the chosen basis ('Z' or 'X').

        basis='Z': H · Rz · Rzz · H   (standard IQP-like, diagonal in Z)
        basis='X': H · Rx · Rxx · H   (diagonal in X)
        Each call re-samples a fresh Erdős–Rényi graph and fresh phases.
        """
        p = self.degree / (self.nqbits - 1)
        G = nx.erdos_renyi_graph(self.nqbits, p)
        edges = list(G.edges())
        phis_1q = np.random.uniform(0, 2 * np.pi, size=self.nqbits)
        phis_2q = np.random.uniform(0, 2 * np.pi, size=len(edges))

        rot1 = qc.rz if basis == 'Z' else qc.rx if basis =='X' else qc.ry
        rot2 = qc.rzz if basis == 'Z' else qc.rxx if basis =='X' else qc.ryy

        for q in range(self.nqbits):
            qc.h(q)
        for q, phi in enumerate(phis_1q):
            rot1(2 * phi, q)
        for (u, v), phi in zip(edges, phis_2q):
            rot2(2 * phi, u, v)
        for q in range(self.nqbits):
            qc.h(q)

    def _build_D2_random(self, qc, degree):
        if degree is None:
            raise ValueError("'degree' must be provided for D2_random")
        self.degree = degree   # so the helper sees it
        self._build_D2_random_layer(qc, basis='Z')

    def _build_D2_random_XZ(self, qc, degree, n_layers):
        """Two stacked commuting blocks: Z-flavour layer then X-flavour layer."""
        if degree is None:
            raise ValueError("'degree' must be provided for D2_random_XZ")
        self.degree = degree
        for _ in range(n_layers):
            self._build_D2_random_layer(qc, basis='Z')
            self._build_D2_random_layer(qc, basis='X')

    def _build_D2_random_XZY(self, qc, degree, n_layers):
        """Two stacked commuting blocks: Z-flavour layer then X-flavour layer."""
        if degree is None:
            raise ValueError("'degree' must be provided for D2_random_XZ")
        self.degree = degree
        for _ in range(n_layers):
            self._build_D2_random_layer(qc, basis='Y')
            self._build_D2_random_layer(qc, basis='Z')
            self._build_D2_random_layer(qc, basis='X')

    def _build_scrambling_ansatz(self, qc, n_layers, scrambler_depth):
        for _ in range(n_layers):
            for q in range(self.nqbits):
                qc.rx(np.random.uniform(0, 2 * np.pi), q)
                qc.rz(np.random.uniform(0, 2 * np.pi), q)
            self._brickwork_scrambler(qc, scrambler_depth)
            for q in range(self.nqbits):
                qc.ry(np.random.uniform(0, 2 * np.pi), q)

    def _brickwork_scrambler(self, qc, depth):
        for k in range(depth):
            start = k % 2
            for q0 in range(start, self.nqbits - 1, 2):
                qc.append(UnitaryGate(random_unitary(4).data), [q0, q0 + 1])

    def _build_random_rotation(self, qc, n_layers=10):
        """10 layers of random single-qubit rotations + CNOT ladder (Xiong et al.)."""
        for _ in range(n_layers):
            for q in range(self.nqbits):
                theta = np.random.uniform(0, 2 * np.pi)
                phi = np.random.uniform(0, 2 * np.pi)
                lam = np.random.uniform(0, 2 * np.pi)
                qc.u(theta, phi, lam, q)
            for q in range(self.nqbits - 1):
                qc.cx(q, q + 1)

    def _build_matchgates(self, qc):
        qubit_idx = list(range(self.nqbits))
        for _ in range(self.num_gates):
            A = unitary_group.rvs(2)
            B = unitary_group.rvs(2)
            B = B / np.sqrt(np.linalg.det(B)) * np.sqrt(np.linalg.det(A))
            G = np.array([[A[0, 0], 0, 0, A[0, 1]],
                          [0, B[0, 0], B[0, 1], 0],
                          [0, B[1, 0], B[1, 1], 0],
                          [A[1, 0], 0, 0, A[1, 1]]])
            q1, q2 = random.sample(qubit_idx, 2)
            qc.unitary(G, [q1, q2], label='MG')

    def _build_symmetric_ising(self, qc):
        """
        Build the symmetric Ising reservoir from arXiv:2505.10062.
    
        Hamiltonian:
            H = sum_{i>j} J_ij σ^x_i σ^x_j  +  (1/2) sum_i (h + h_i) σ^z_i
    
        where:
            J_ij ~ Uniform[-J_s/2, J_s/2]   (all-to-all random couplings)
            h_i  ~ Uniform[-W, W]           (on-site disorder along z)
            h                                (uniform transverse field along z)
    
        Reservoir unitary:
            U = exp(-i H dt)
    
        symmetric Ising's deep-thermal-phase parameters: W = 1e-2, h = 1e+1, J_s = 1.
        Vary `dt` to control scrambling depth.
        """
        n = self.nqbits
        J_s = self.J_s
        h = self.h_field
        W = self.W_disorder
        dt = self.dt
    
        # Sample disorder realization (ONE per QELM instance — the randomness
        # of the family lives in (J_ij, h_i), not in the gate sequence)
        J = np.random.uniform(-J_s / 2, J_s / 2, size=(n, n))
        J = np.triu(J, k=1)   # keep only i<j entries; J[j,i] = 0
        h_i = np.random.uniform(-W, W, size=n)
    
        # Build H as a SparsePauliOp for clarity, then materialize as a matrix.
        pauli_terms = []
        coeffs = []
    
        # XX interaction terms
        for i in range(n):
            for j in range(i + 1, n):
                if J[i, j] == 0.0:
                    continue
                label = ['I'] * n
                # Qiskit convention: label[0] is the highest-index qubit.
                # Use the same ordering as _k_local_observables (qubit 0 → label[-1]).
                label[n - 1 - i] = 'X'
                label[n - 1 - j] = 'X'
                pauli_terms.append(''.join(label))
                coeffs.append(J[i, j])
    
        # Z field terms (uniform + disorder)
        for i in range(n):
            label = ['I'] * n
            label[n - 1 - i] = 'Z'
            pauli_terms.append(''.join(label))
            coeffs.append(0.5 * (h + h_i[i]))
    
        H_op = SparsePauliOp(pauli_terms, coeffs=coeffs)
        H_mat = H_op.to_matrix()   # dense, 2^n x 2^n complex
    
        # U = exp(-i H dt) — exact unitary for the full evolution.
        U = expm(-1j * H_mat * dt)
    
        qc.append(UnitaryGate(U), list(range(n)))

    # ─────────────────────────────────────────────────────────────────────
    # Observables (built once at init)
    # ─────────────────────────────────────────────────────────────────────
    def _build_observables(self):
        """Dispatch observables based on observables_type parameter."""
        ot = self.observables_type
        # Explicit list of SparsePauliOp
        if len(ot) > 0 and isinstance(ot[0], SparsePauliOp):
            return list(ot)

        # Locality specification via integers
        obs = []
        if 1 in ot:
            obs += self._k_local_observables(1)
        if 2 in ot:
            obs += self._k_local_observables(2)
        if 3 in ot:
            obs += self._k_local_observables(3)
        return obs

    def _k_local_observables(self, k, paulis=('X', 'Y', 'Z')):
        """Generate all k-local Pauli observables."""
        obs = []
        for sites in combinations(range(self.nqbits), k):
            for pauli_tuple in product(paulis, repeat=k):
                op = ['I'] * self.nqbits
                for site, pauli in zip(sites, pauli_tuple):
                    op[site] = pauli
                obs.append(SparsePauliOp(''.join(op)))
        return obs

    # ─────────────────────────────────────────────────────────────────────
    # Encoding
    # ─────────────────────────────────────────────────────────────────────
    def _encode(self, initial_state, qubit_offset=0):
        """Build encoding circuit (used for the slow/hardware path and as a fallback)."""
        qc = QuantumCircuit(self.nqbits)

        if self.encoding == 'quantum':
            state = np.asarray(initial_state, dtype=complex).round(6)
            state /= np.linalg.norm(state)
            qc.initialize(state, list(range(int(np.log2(len(state))))))
            return qc

        y_vec = np.asarray(initial_state, dtype=float).reshape(-1)
        scalar = len(y_vec) == 1

        if self.encoding == 'pauli':
            qubits = (range(self.nqbits) if scalar
                      else [qubit_offset + j for j in range(len(y_vec))])
            values = ([y_vec[0]] * self.nqbits if scalar else y_vec)
            for q, x in zip(qubits, values):
                qc.h(q)
                qc.rz(-self.theta_scale * x, q)

        elif self.encoding == 'exponential':
            if scalar:
                for l in range(self.k_exp):
                    q = qubit_offset + l
                    qc.h(q)
                    qc.rz(-self.theta_scale * (3 ** l) * y_vec[0], q)
            else:
                for j, x in enumerate(y_vec):
                    for l in range(self.k_exp):
                        q = qubit_offset + j * self.k_exp + l
                        qc.h(q)
                        qc.rz(-self.theta_scale * (3 ** l) * x, q)
        else:
            raise ValueError(f"Unknown encoding: '{self.encoding}'")

        return qc

    # ─────────────────────────────────────────────────────────────────────
    # Fast encoding: direct product-state construction
    # ─────────────────────────────────────────────────────────────────────
    def _encode_state_fast(self, initial_state):
        """
        Build the encoded statevector |ψ_enc⟩ directly as a Kronecker product
        of single-qubit states, bypassing Qiskit circuit simulation.

        The encoding circuit applies H then Rz(-θ) on each active qubit.
        As an operator, that is Rz(-θ) · H acting on |0⟩, giving the
        single-qubit state:

            Rz(-θ) · H |0⟩ = (1/√2) (e^{+iθ/2}|0⟩ + e^{-iθ/2}|1⟩)

        For exponential encoding: θ_q = θ_scale · 3^q · x  for q < k_exp.
        For pauli encoding:       θ_q = θ_scale · x        for all q.
        Inactive qubits stay at |0⟩.

        Qiskit convention is little-endian (qubit 0 = least significant index),
        so the full state is built as a Kronecker product q = n-1, n-2, ..., 0.

        Returns None if the encoding can't be handled analytically (vector
        input, 'quantum' encoding, etc.) so the caller falls back to the
        circuit-based path.
        """
        if self.encoding == 'quantum':
            return None

        y_vec = np.asarray(initial_state, dtype=float).reshape(-1)
        if len(y_vec) != 1:
            return None  # multi-feature input → fall back to circuit path
        x = float(y_vec[0])

        # Per-qubit angle θ_q (the Rz argument is -θ_q, sign already absorbed below).
        thetas = np.zeros(self.nqbits)
        active = np.zeros(self.nqbits, dtype=bool)
        if self.encoding == 'pauli':
            thetas[:] = self.theta_scale * x
            active[:] = True
        elif self.encoding == 'exponential':
            for q in range(self.k_exp):
                thetas[q] = self.theta_scale * (3 ** q) * x
                active[q] = True
        else:
            return None

        # Build per-qubit 2-vectors.
        # Active qubit:  Rz(-θ) · H |0⟩ = (1/√2)[exp(+iθ/2), exp(-iθ/2)]
        # Inactive qubit: |0⟩
        inv_sqrt2 = 1.0 / np.sqrt(2)
        single = np.empty((self.nqbits, 2), dtype=complex)
        for q in range(self.nqbits):
            if active[q]:
                half = 0.5 * thetas[q]
                single[q] = inv_sqrt2 * np.array(
                    [np.exp(+1j * half), np.exp(-1j * half)], dtype=complex)
            else:
                single[q] = np.array([1.0, 0.0], dtype=complex)

        # Kronecker product right-to-left (Qiskit little-endian convention):
        # |ψ⟩ = single[n-1] ⊗ single[n-2] ⊗ ... ⊗ single[0]
        psi = single[self.nqbits - 1]
        for q in range(self.nqbits - 2, -1, -1):
            psi = np.kron(psi, single[q])
        return psi

    # ─────────────────────────────────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────────────────────────────────
    def get_reservoir(self):
        """Return the (fixed) reservoir circuit."""
        return self.reservoir_circuit

    def get_qc(self, initial_state):
        """Build the full circuit: encoding + reservoir."""
        return self._encode(initial_state).compose(self.reservoir_circuit)

    def run_circuit(self, initial_state):
        """
        Compute the feature vector ⟨O_j⟩ for one input.

        Fast path (statevector + cached U_res + scalar pauli/exponential):
            1. Build |ψ_enc⟩ analytically as a Kronecker product.
            2. Apply U_res via a single matvec.
            3. Compute expectation values.

        Falls back to the original circuit-simulation path otherwise.
        """
        if self.statevector and self._U_res is not None:
            # Try the fully-analytic encoding first.
            psi_enc = self._encode_state_fast(initial_state)

            # Fallback: build encoding circuit and use Statevector
            # (still benefits from cached U_res via matvec).
            if psi_enc is None:
                enc_qc = self._encode(initial_state)
                psi_enc = Statevector(enc_qc).data

            psi = self._U_res @ psi_enc

            if self._obs_mats is not None:
                return np.real(np.einsum('i,ji->j', psi.conj(), self._obs_mats @ psi))

            # n > cache_threshold: sparse Pauli path on the cached state
            sv = Statevector(psi)
            return np.array([sv.expectation_value(obs).real
                             for obs in self.observables])

        # Original path: full circuit simulation per call (slow but general)
        if self.statevector:
            qc = self.get_qc(initial_state)
            psi_sv = Statevector(qc)

            if self._obs_mats is not None:
                psi = psi_sv.data
                return np.real(np.einsum('i,ji->j', psi.conj(), self._obs_mats @ psi))

            return np.array([psi_sv.expectation_value(obs).real
                             for obs in self.observables])

        # Hardware / shot-noise path
        qc = self.get_qc(initial_state)
        return self.estimator.run([(qc, self.observables)]).result()[0].data.evs

    # ─────────────────────────────────────────────────────────────────────
    # Batched API: evaluate all inputs in one shot
    # ─────────────────────────────────────────────────────────────────────
    def feature_matrix(self, x_array):
        """
        Compute the (N, n_obs) feature matrix for an array of inputs in one call.

        Parameters
        ----------
        x_array : array-like
            Array of scalar inputs of length N (or shape (N, 1)).

        Returns
        -------
        feat : np.ndarray, shape (N, n_obs)
            Real expectation values ⟨O_j⟩ for each input.

        Notes
        -----
        Requires statevector mode, cached U_res, cached _obs_mats, and scalar
        pauli/exponential encoding. Falls back to a Python loop over
        run_circuit otherwise.
        """
        x_array = np.asarray(x_array).reshape(-1)
        N = len(x_array)

        # Check if the fast vectorized path is available
        fast_ok = (self.statevector
                   and self._U_res is not None
                   and self._obs_mats is not None
                   and self.encoding in ('pauli', 'exponential'))

        if not fast_ok:
            return np.array([self.run_circuit(x) for x in x_array])

        # Build all encoding states in a batch.
        # _encode_state_fast returns None for vector inputs / quantum encoding,
        # but we already gated on encoding ∈ {pauli, exponential} above.
        psi_enc = np.empty((N, 2 ** self.nqbits), dtype=complex)
        for i, x in enumerate(x_array):
            psi_enc[i] = self._encode_state_fast(x)

        # Apply reservoir to all inputs at once: (N, D) = (N, D) @ (D, D)
        psi = psi_enc @ self._U_res.T

        # Step 1: (n_obs, D, D) @ (D, N) = (n_obs, D, N)
        Opsi = self._obs_mats @ psi.T

        # Step 2: contract D-axis with psi.conj()
        # We want feat[i, j] = sum_a psi[i, a].conj() * Opsi[j, a, i]
        feat = np.real(np.einsum('ia,jai->ij', psi.conj(), Opsi))
        return feat