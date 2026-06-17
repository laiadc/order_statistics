import itertools
import numpy as np
import random
import networkx as nx

from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp, random_unitary, Statevector
from qiskit.circuit.library import UnitaryGate, DiagonalGate
from qiskit_aer import AerSimulator
from qiskit_ibm_runtime import EstimatorV2 as Estimator
from itertools import combinations, product
from scipy.stats import unitary_group


class QuantumCircQiskit:
    def __init__(self, gates_name, num_gates=50, nqbits=8, observables_type=[1],
                 degree=None, n_layers=1, scrambler_depth=1, backend=None,
                 statevector=True, nshots=1024, encoding='quantum',
                 k_exp=2, theta_scale=np.pi, cache_threshold=9):
        self.gates_name = gates_name
        self.num_gates = num_gates
        self.nqbits = nqbits
        self.statevector = statevector
        self.nshots = nshots
        self.encoding = encoding
        self.k_exp = k_exp
        self.theta_scale = theta_scale
        self.observables_type = observables_type
        self.cache_thresold = cache_threshold

        # Build reservoir ONCE
        self.reservoir_circuit = self._build_reservoir(degree, n_layers, scrambler_depth)

        # Build observables ONCE and cache their matrices
        self.observables = self._build_observables()

        # Cache dense matrices only for small systems
        if self.nqbits <= self.cache_thresold:
            self._obs_mats = np.array([obs.to_matrix() for obs in self.observables])
        else:
            self._obs_mats = None   # use sparse Pauli path

        # Backend setup
        self.backend = backend if backend is not None else AerSimulator()
        self.estimator = Estimator(self.backend)

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

        elif name == 'scrambling_ansatz':
            self._build_scrambling_ansatz(qc, n_layers, scrambler_depth)

        elif name == 'RandomRotation':
            self._build_random_rotation(qc, n_layers=10)

        elif name == 'Haar':
            U = random_unitary(2 ** self.nqbits).data
            qc.append(UnitaryGate(U), list(range(self.nqbits)))

        elif name == 'MG':
            self._build_matchgates(qc)

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

    def _build_D2_random(self, qc, degree):
        p = degree / (self.nqbits - 1)
        G = nx.erdos_renyi_graph(self.nqbits, p)
        edges = list(G.edges())
        phis_1q = np.random.uniform(0, 2 * np.pi, size=self.nqbits)
        phis_2q = np.random.uniform(0, 2 * np.pi, size=len(edges))

        for q in range(self.nqbits):
            qc.h(q)
        for q, phi in enumerate(phis_1q):
            qc.rz(2 * phi, q)
        for (u, v), phi in zip(edges, phis_2q):
            qc.rzz(2 * phi, u, v)
        for q in range(self.nqbits):
            qc.h(q)

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
    # Encoding (called per run_circuit)
    # ─────────────────────────────────────────────────────────────────────
    def _encode(self, initial_state, qubit_offset=0):
        """Apply encoding for initial_state."""
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
    # Public API
    # ─────────────────────────────────────────────────────────────────────
    def get_reservoir(self):
        """Return the (fixed) reservoir circuit."""
        return self.reservoir_circuit

    def get_qc(self, initial_state):
        """Build the full circuit: encoding + reservoir."""
        return self._encode(initial_state).compose(self.reservoir_circuit)

    def run_circuit(self, initial_state):
        qc = self.get_qc(initial_state)
        
        if self.statevector:
            psi_sv = Statevector(qc)
            
            if self._obs_mats is not None:
                # Fast dense path for n ≤ 9
                psi = psi_sv.data
                return np.real(np.einsum('i,ji->j', psi.conj(), self._obs_mats @ psi))
            
            # Efficient sparse Pauli path for n ≥ 10
            return np.array([psi_sv.expectation_value(obs).real 
                            for obs in self.observables])
        
        # Hardware / shot-noise path
        return self.estimator.run([(qc, self.observables)]).result()[0].data.evs