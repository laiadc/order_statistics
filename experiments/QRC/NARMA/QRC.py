import os
os.environ["OMP_NUM_THREADS"]        = "1"
os.environ["MKL_NUM_THREADS"]        = "1"
os.environ["OPENBLAS_NUM_THREADS"]   = "1"
os.environ["NUMEXPR_NUM_THREADS"]    = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"

import itertools
from random import sample

import networkx as nx
import numpy as np
from qiskit import QuantumCircuit
from qiskit.synthesis.evolution import SuzukiTrotter
from qiskit.quantum_info import SparsePauliOp, random_unitary, Operator
from qiskit.circuit.library import UnitaryGate, Diagonal, PauliEvolutionGate
from scipy.stats import unitary_group
from sklearn.linear_model import Ridge
from sklearn.svm import SVR
from sklearn.multioutput import MultiOutputRegressor

from qiskit_aer import AerSimulator
from qiskit_aer.primitives import EstimatorV2


class QuantumRC_Circuit:
    """Circuit-based Quantum Reservoir Computer with density_matrix and shots exec modes.

    Qubit layout:
        0 .. n_enc-1             : input qubits (reset + re-encoded each step)
        n_enc .. n_measure-1     : measurement-only qubits
        n_measure .. n_total-1   : hidden qubits (carry quantum memory)

    Total qubits = n_enc + nqbits.
    """

    # ================================================================== #
    # Construction
    # ================================================================== #
    def __init__(self, nqbits=6, num_gates=20, gates_set=['CNOT', 'T', 'H'],
                 alpha=1e-1, t=10, Nv=1, observables_type=[1],
                 degree=None, n_layers=1, scrambler_depth=1, t_dismiss=0,
                 encoding='pauli', k_exp=1, theta_scale=np.pi,
                 n_measure=None, exec_mode='density_matrix',
                 backend=None, estimator_options=None,
                 rolling_window=20, predictor='ridge',
                 input_mode='autoregressive', feedback='teacher',
                 dim_s=0, dim_y=1):
        """
        Reservoir input dimension m is derived from (input_mode, dim_s, dim_y):
          autoregressive -> m = dim_y
          exogenous      -> m = dim_s
          hybrid         -> m = dim_s + dim_y

        Defaults (input_mode='autoregressive', dim_s=0, dim_y=1) reproduce the
        old m=1 behavior, so existing calls keep working.
        """

        # Config
        self.nqbits = nqbits
        self.num_gates = num_gates
        self.gates_set = gates_set
        self.alpha = alpha
        self.t = t
        self.Nv = Nv
        self.observables_type = observables_type
        self.degree = degree
        self.n_layers = n_layers
        self.scrambler_depth = scrambler_depth
        self.t_dismiss = t_dismiss
        self.predictor = predictor
        self.lm = None

        # Input mode
        if input_mode not in ('autoregressive', 'exogenous', 'hybrid'):
            raise ValueError(f"input_mode must be 'autoregressive'|'exogenous'|"
                             f"'hybrid', got {input_mode!r}")
        if feedback not in ('teacher', 'closed_loop'):
            raise ValueError(f"feedback must be 'teacher' or 'closed_loop', "
                             f"got {feedback!r}")
        self.input_mode = input_mode
        self.feedback = feedback
        self.dim_s = dim_s
        self.dim_y = dim_y

        # Derive m from mode
        if input_mode == 'autoregressive':
            m = dim_y
        elif input_mode == 'exogenous':
            m = dim_s
        else:  # hybrid
            m = dim_s + dim_y
        if m <= 0:
            raise ValueError(f"Derived m={m} is invalid. Check dim_s={dim_s}, "
                             f"dim_y={dim_y} for input_mode={input_mode!r}.")

        # Encoding
        self.m = m
        self.encoding = encoding
        self.k_exp = k_exp
        self.theta_scale = theta_scale
        self.n_enc = m * k_exp if encoding in ('pauli', 'exponential') \
                     else self._raise(f"Unknown encoding '{encoding}'.")
        self.n_total = self.n_enc + self.nqbits

        # Measurement subsystem
        n_measure = self.n_enc if n_measure is None else n_measure
        if not (self.n_enc <= n_measure <= self.n_total):
            raise ValueError(f"Need n_enc <= n_measure <= n_total "
                             f"(got {self.n_enc}, {n_measure}, {self.n_total}).")
        self.n_measure = n_measure

        # Execution mode
        if exec_mode not in ('shots', 'density_matrix'):
            raise ValueError(f"exec_mode must be 'shots' or 'density_matrix'.")
        self.exec_mode = exec_mode
        self.rolling_window = rolling_window

        # Reservoir circuit (built once)
        self.qc_U = self._build_reservoir()
        self.U = Operator(self.qc_U).data

        # Observables (built once, matrices cached for DM mode)
        self.observables = self._build_observables()
        self.obs_array = np.array([o.to_matrix() for o in self.observables])

        # Estimator (shots mode)
        self.backend = backend if backend is not None else AerSimulator()
        self._estimator_options = estimator_options or {'default_precision': 0.01}
        self._estimator = EstimatorV2(options=self._estimator_options)

    @staticmethod
    def _raise(msg):
        raise ValueError(msg)

    def get_reservoir(self):
        return self.qc_U

    # ================================================================== #
    # Reservoir builders (dispatched once at __init__)
    # ================================================================== #
    def _build_reservoir(self):
        g = self.gates_set
        nq = self.n_total

        if g == 'Ising':              return self._build_ising()
        if g == 'MG':                 return self._build_MG()
        if g == 'D2':                 return self._build_Dk(k=2)
        if g == 'D3':                 return self._build_Dk(k=3)
        if g == 'Dn':                 return self._build_Dn()
        if g == 'D2_random':          return self._build_D2_random()
        if g == 'D2_random_XZ':       return self._build_D2_random_XZ()
        if g == 'D2_random_XZY':       return self._build_D2_random_XZY()
        if g == 'scrambling_ansatz':  return self._build_scrambling_ansatz()
        if g == 'Haar':               return self._build_haar()
        if g in ('G1', 'G2', 'G3'):
            self.gates_set = {'G1': ['CNOT', 'H', 'X'],
                              'G2': ['CNOT', 'H', 'S'],
                              'G3': ['CNOT', 'H', 'T']}[g]
        return self._build_random_gates()

    def _build_ising(self):
        nq = self.n_total
        base = 'I' * nq
        terms = []
        for (i, j) in itertools.combinations(range(nq), 2):
            name = base[:i] + 'Z' + base[i+1:j] + 'Z' + base[j+1:]
            terms.append((name, np.random.normal(0.75, 0.1)))
        for q in range(nq):
            terms.append((base[:q] + 'X' + base[q+1:], np.random.normal(1.0, 0.1)))
        evo = PauliEvolutionGate(SparsePauliOp.from_list(terms), time=self.t,
                                  synthesis=SuzukiTrotter(order=2, reps=1))
        qc = QuantumCircuit(nq)
        qc.append(evo, range(nq))
        return qc.decompose()

    def _matchgate(self):
        A, B = unitary_group.rvs(2), unitary_group.rvs(2)
        B = B / np.sqrt(np.linalg.det(B)) * np.sqrt(np.linalg.det(A))
        return np.array([[A[0,0], 0, 0, A[0,1]],
                         [0, B[0,0], B[0,1], 0],
                         [0, B[1,0], B[1,1], 0],
                         [A[1,0], 0, 0, A[1,1]]])

    def _build_MG(self):
        qc = QuantumCircuit(self.n_total)
        for _ in range(self.num_gates):
            q1, q2 = np.random.choice(self.n_total, 2, replace=False)
            qc.unitary(self._matchgate(), [int(q1), int(q2)], label='MG')
        return qc

    def _build_Dk(self, k):
        """Unified D2/D3 builder."""
        qc = QuantumCircuit(self.n_total)
        groups = list(itertools.combinations(range(self.n_total), k))
        phis = np.random.uniform(0, 2*np.pi, size=(len(groups), 2**k))
        for group, phi in zip(groups, phis):
            qc.append(UnitaryGate(np.diag(np.exp(1j * phi))), list(group))
        return qc

    def _build_Dn(self):
        qc = QuantumCircuit(self.n_total)
        phis = np.random.uniform(0, 2*np.pi, size=2**self.n_total)
        return qc.compose(Diagonal(np.exp(1j * phis)))

    def _build_D2_random_layer(self, qc, basis):
        """One D2_random commuting layer in basis ∈ {'Z','X'}. Re-samples per call."""
        nq = self.n_total
        G = nx.erdos_renyi_graph(nq, self.degree / (nq - 1))
        phis_1q = np.random.uniform(0, 2*np.pi, size=nq)
        phis_2q = np.random.uniform(0, 2*np.pi, size=len(G.edges()))

        rot1 = qc.rz if basis == 'Z' else qc.rx if basis =='X' else qc.ry
        rot2 = qc.rzz if basis == 'Z' else qc.rxx if basis =='X' else qc.ryy

        for q in range(nq): qc.h(q)
        for q, phi in enumerate(phis_1q): rot1(2*phi, q)
        for (u, v), phi in zip(G.edges(), phis_2q): rot2(2*phi, u, v)
        for q in range(nq): qc.h(q)

    def _build_D2_random(self):
        qc = QuantumCircuit(self.n_total)
        self._build_D2_random_layer(qc, basis='Z')
        return qc

    def _build_D2_random_XZ(self):
        qc = QuantumCircuit(self.n_total)
        for layer in range(self.n_layers):
            self._build_D2_random_layer(qc, basis='Z')
            self._build_D2_random_layer(qc, basis='X')
        return qc
    
    def _build_D2_random_XZY(self):
        qc = QuantumCircuit(self.n_total)
        for layer in range(self.n_layers):
            self._build_D2_random_layer(qc, basis='Y')
            self._build_D2_random_layer(qc, basis='Z')
            self._build_D2_random_layer(qc, basis='X')
        return qc

    def _build_scrambling_ansatz(self):
        nq = self.n_total
        qc = QuantumCircuit(nq)
        for _ in range(self.n_layers):
            for q in range(nq):
                qc.rx(np.random.uniform(0, 2*np.pi), q)
                qc.rz(np.random.uniform(0, 2*np.pi), q)
            for k in range(self.scrambler_depth):
                for q0 in range(k % 2, nq - 1, 2):
                    qc.append(UnitaryGate(random_unitary(4).data), [q0, q0 + 1])
            for q in range(nq):
                qc.ry(np.random.uniform(0, 2*np.pi), q)
        return qc

    def _build_haar(self):
        qc = QuantumCircuit(self.n_total)
        qc.append(UnitaryGate(random_unitary(2**self.n_total).data), range(self.n_total))
        return qc

    def _build_random_gates(self):
        qc = QuantumCircuit(self.n_total)
        nq = self.n_total
        gate_fns = {'X': qc.x, 'S': qc.s, 'H': qc.h, 'T': qc.t}
        for _ in range(self.num_gates):
            gate = sample(self.gates_set, 1)[0]
            if gate == 'CNOT':
                q1, q2 = np.random.choice(nq, 2, replace=False)
                qc.cx(int(q1), int(q2))
            else:
                gate_fns[gate](sample(range(nq), 1)[0])
        return qc

    # ================================================================== #
    # Observables
    # ================================================================== #
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
        """All k-local Pauli observables on the first n_measure qubits."""
        obs = []
        for sites in itertools.combinations(range(self.n_measure), k):
            for pauli_tuple in itertools.product(paulis, repeat=k):
                op = ['I'] * self.n_total
                for site, pauli in zip(sites, pauli_tuple):
                    op[self.n_total - 1 - site] = pauli
                obs.append(SparsePauliOp(''.join(op)))
        return obs

    # ================================================================== #
    # Encoding (unified: both circuit and pure-state forms share logic)
    # ================================================================== #
    def _encoding_angles(self, y_vec):
        """Return [(qubit_index, angle), ...] for the encoding of y_vec."""
        y_vec = np.asarray(y_vec, dtype=float).reshape(-1)
        angles = []
        if self.encoding == 'pauli':
            for j, x in enumerate(y_vec):
                for l in range(self.k_exp):
                    angles.append((j * self.k_exp + l, self.theta_scale * x))
        elif self.encoding == 'exponential':
            for j, x in enumerate(y_vec):
                for l in range(self.k_exp):
                    angles.append((j * self.k_exp + l, self.theta_scale * (3**l) * x))
        return angles

    def _apply_encoding(self, qc, y_vec, qubit_offset=0):
        """Apply encoding to a circuit (for shots mode)."""
        for q, theta in self._encoding_angles(y_vec):
            qc.h(qubit_offset + q)
            qc.rz(-theta, qubit_offset + q)

    def _reset_and_reencode(self, qc, y_vec):
        for j in range(self.n_enc):
            qc.reset(j)
        self._apply_encoding(qc, y_vec)

    def _make_window_init_circuit(self, y_vec):
        qc = QuantumCircuit(self.n_total)
        self._apply_encoding(qc, y_vec)
        for q in range(self.n_enc, self.n_total):
            qc.h(q)
        return qc

    def _encode_pure_state(self, y_vec):
        """Return the 2**n_enc pure state from encoding y_vec on |0...0>."""
        angles = self._encoding_angles(y_vec)
        qstates = [None] * self.n_enc
        for q, theta in angles:
            qstates[q] = np.array([np.exp(1j * theta / 2), np.exp(-1j * theta / 2)]) / np.sqrt(2)
        # Kron with highest qubit leftmost (Qiskit convention)
        state = qstates[-1]
        for q in range(self.n_enc - 2, -1, -1):
            state = np.kron(state, qstates[q])
        return state

    # ================================================================== #
    # Density-matrix utilities
    # ================================================================== #
    def _partial_trace_enc(self, dm):
        """Trace out the n_enc input qubits, return reservoir density matrix."""
        d_enc, d_res = 2**self.n_enc, 2**self.nqbits
        return np.einsum('aibi->ab', dm.reshape(d_res, d_enc, d_res, d_enc))

    def _inject_encoding(self, rho_res, y_vec):
        """Tensor a fresh encoded pure state with the traced reservoir."""
        psi = self._encode_pure_state(y_vec)
        rho_enc = np.outer(psi, psi.conj())
        return np.kron(rho_res, rho_enc)

    def _fresh_initial_dm(self, y_vec):
        """|psi(y)>_enc ⊗ |+>^nqbits density matrix."""
        d_res = 2**self.nqbits
        psi_res = np.ones(d_res) / np.sqrt(d_res)
        return self._inject_encoding(np.outer(psi_res, psi_res.conj()), y_vec)

    # ================================================================== #
    # Feature extraction: shots mode
    # ================================================================== #
    def _compute_features_shots(self, y_scaled):
        T = y_scaled.shape[0]
        n_obs = len(self.observables)
        X = np.zeros((T, n_obs * self.Nv))

        step = 0
        while step < T:
            end = min(step + self.rolling_window, T)
            qc = self._make_window_init_circuit(y_scaled[step])
            snapshots = []
            for t in range(step, end):
                for nv in range(self.Nv):
                    qc = qc.compose(self.qc_U)
                    snapshots.append((t, nv, qc.copy()))
                if t + 1 < end:
                    self._reset_and_reencode(qc, y_scaled[t + 1])

            pubs = [(c, self.observables) for (_, _, c) in snapshots]
            result = self._estimator.run(pubs).result()
            for (t, nv, _), r in zip(snapshots, result):
                X[t, nv * n_obs:(nv + 1) * n_obs] = np.asarray(r.data.evs)
            step = end
        return X

    # ================================================================== #
    # Feature extraction: density-matrix mode (fast path)
    # ================================================================== #
    def _compute_features_dm(self, y_scaled, dm_init=None,
                             obs_array=None, cache_dm_traj=False):
        """Run the density-matrix reservoir over a pre-built input sequence.

        Parameters
        ----------
        y_scaled : ndarray, shape (T, m)
            Reservoir input sequence (each row is encoded into the input qubits).
        dm_init : ndarray or None
            Initial density matrix. If None, built from y_scaled[0].
        obs_array : ndarray or None
            Override the observable stack (shape (n_obs, d, d)).
        cache_dm_traj : bool
            If True, store self.dm_traj of shape (T*Nv, d, d) for post-hoc
            projection onto any observable bank via `project_cached`.
        """
        T = y_scaled.shape[0]
        obs = self.obs_array if obs_array is None else obs_array
        n_obs = obs.shape[0]
        X = np.zeros((T, n_obs * self.Nv))

        dm = dm_init if dm_init is not None else self._fresh_initial_dm(y_scaled[0])
        U_dag = self.U.conj().T

        if cache_dm_traj:
            d = 2**self.n_total
            dm_traj = np.empty((T * self.Nv, d, d), dtype=complex)

        for t in range(T):
            for nv in range(self.Nv):
                dm = self.U @ dm @ U_dag
                if cache_dm_traj:
                    dm_traj[t * self.Nv + nv] = dm
                X[t, nv*n_obs:(nv+1)*n_obs] = np.tensordot(
                    obs, dm, axes=[[2, 1], [0, 1]]).real

            if t + 1 < T:
                dm = self._inject_encoding(self._partial_trace_enc(dm), y_scaled[t + 1])

        self.dm = dm
        if cache_dm_traj:
            self.dm_traj = dm_traj
        return X

    def project_cached(self, obs_array):
        """Project a cached DM trajectory onto `obs_array`.

        Call after a run with cache_dm_traj=True. Returns feature matrix of
        shape (T, n_obs * Nv) identical to what would have been produced if
        the reservoir were constructed with these observables.
        """
        if not hasattr(self, 'dm_traj'):
            raise RuntimeError("No cached DM trajectory. Run with cache_dm_traj=True.")
        n_obs = obs_array.shape[0]
        Nv = self.Nv
        TNv = self.dm_traj.shape[0]
        T = TNv // Nv
        feats = np.tensordot(obs_array, self.dm_traj,
                             axes=[[2, 1], [1, 2]]).real  # (n_obs, T*Nv)
        return feats.T.reshape(T, Nv, n_obs).reshape(T, Nv * n_obs)

    def _compute_features(self, y_scaled, dm_init=None, cache_dm_traj=False):
        if self.exec_mode == 'shots':
            return self._compute_features_shots(y_scaled)
        return self._compute_features_dm(
            y_scaled, dm_init=dm_init, cache_dm_traj=cache_dm_traj)

    # ================================================================== #
    # Input-sequence construction
    # ================================================================== #
    def _build_train_sequence(self, s, y, y_warmup=None):
        """Assemble the reservoir input sequence x_k for training.

        At training, target at step k is y_k and the reservoir reads x_k that
        depends on input_mode. The natural lag of "input read -> target
        predicted" is handled downstream by prepending a zero row in `train`
        before calling `_compute_features`, so here x_seq is NOT pre-lagged.

          - autoregressive : x_k = y_k      (the prepended zero effectively
                                             makes the reservoir see y_{k-1}
                                             when predicting y_k)
          - exogenous      : x_k = s_k      (prepended zero means step 0 sees
                                             no real input, matching the old
                                             behavior)
          - hybrid         : x_k = [s_k, y_k] with same convention

        Returns (x_seq, y_target_seq) both of length T.
        """
        if self.input_mode == 'exogenous':
            if s is None:
                raise ValueError("exogenous mode requires s at training")
            s = np.asarray(s).reshape(len(s), -1)
            y = np.asarray(y).reshape(len(y), -1)
            assert s.shape[0] == y.shape[0]
            return s, y

        if self.input_mode == 'autoregressive':
            y = np.asarray(y).reshape(len(y), -1)
            return y, y   # no pre-lag; train() handles it via zero-prepend

        # hybrid
        if s is None:
            raise ValueError("hybrid mode requires s at training")
        s = np.asarray(s).reshape(len(s), -1)
        y = np.asarray(y).reshape(len(y), -1)
        assert s.shape[0] == y.shape[0]
        x = np.hstack([s, y])
        return x, y

    # ================================================================== #
    # Predictor
    # ================================================================== #
    def _make_predictor(self):
        if self.predictor == 'ridge':
            return Ridge(alpha=self.alpha)
        if self.predictor == 'svr':
            return MultiOutputRegressor(
                SVR(kernel='rbf', C=self.alpha, epsilon=0.1, gamma='scale'))
        raise ValueError(f"Unknown predictor '{self.predictor}'.")

    # ================================================================== #
    # Public API: train / test
    # ================================================================== #
    def train(self, s=None, y=None, scaler=None, X_train_scaled=None,
              y_warmup=None, cache_dm_traj=False, **legacy_kwargs):
        """Train the QRC.

        Signature (new, preferred):
            qrc.train(s=s_train, y=y_train)

        Legacy signature (only for input_mode='autoregressive'):
            qrc.train(y_train_scaled, scaler, X_train_scaled=None, y_target=None)
            -> treated as qrc.train(y=y_train_scaled or y_target, scaler=scaler, ...)
        """
        # Legacy positional compat: if only one array passed and mode is autoregressive
        if 'y_train_scaled' in legacy_kwargs and y is None:
            y = legacy_kwargs.pop('y_train_scaled')
        if 'y_target' in legacy_kwargs:
            y_tgt_override = legacy_kwargs.pop('y_target')
            if y is None:
                y = y_tgt_override
        else:
            y_tgt_override = None
        if legacy_kwargs:
            raise TypeError(f"Unexpected kwargs: {list(legacy_kwargs)}")

        # If user passed `s` as the first positional (old-style NARMA code),
        # warn unless mode is set correctly.
        if self.input_mode == 'autoregressive' and s is not None and y is None:
            # Old: qrc.train(y_train)
            y = s
            s = None
        if self.input_mode != 'autoregressive' and s is None:
            raise ValueError(
                f"input_mode={self.input_mode!r} requires `s` (exogenous signal). "
                f"Pass qrc.train(s=..., y=...).")

        if y is None:
            raise ValueError("y is required")

        # Build input sequence and target
        x_seq, y_tgt = self._build_train_sequence(s, y, y_warmup=y_warmup)
        if y_tgt_override is not None:
            y_tgt = np.asarray(y_tgt_override)
            if y_tgt.ndim == 1: y_tgt = y_tgt.reshape(-1, 1)

        assert x_seq.shape[1] == self.m, \
            f"Input dim {x_seq.shape[1]} != reservoir m {self.m}"

        # Prepend blank input (matches old behavior), then drop last feature row
        y_ext = np.vstack([np.zeros((1, self.m)), x_seq])
        X_all = self._compute_features(y_ext, cache_dm_traj=cache_dm_traj)
        X_feat = X_all[:-1]

        if X_train_scaled is not None:
            assert X_train_scaled.shape[0] == X_feat.shape[0]
            X_feat = np.hstack([X_feat, X_train_scaled])

        if self.t_dismiss > 0:
            X_feat, y_tgt = X_feat[self.t_dismiss:], y_tgt[self.t_dismiss:]

        self.lm = self._make_predictor()
        self.lm.fit(X_feat, y_tgt)

        y_pred = self.lm.predict(X_feat)
        if scaler is not None:
            y_pred = scaler.inverse_transform(y_pred.reshape(-1, y_tgt.shape[1]))

        # Cache training state for test-time continuation
        self.X_train = X_feat
        self.y_train_scaled = y_tgt
        self.X_feat = X_feat
        self._last_x_input = x_seq[-1]    # last reservoir input at train end
        self._last_y_true = y_tgt[-1]     # last true y at train end (for warmup)
        return y_pred

    def test(self, s=None, y=None, scaler=None, X_test_scaled=None,
             retrain=False, cache_dm_traj=False, **legacy_kwargs):
        """Test the QRC.

        New signature:
            qrc.test(s=s_test, y=y_test)   # y optional if feedback='closed_loop'
                                           # and mode='exogenous'

        Test-time behavior depends on self.input_mode and self.feedback:
          - exogenous                  : reservoir reads s_k each step
          - autoregressive + teacher   : reservoir reads true y_{k-1}
          - autoregressive + closed_loop : reservoir reads y_hat_{k-1}
          - hybrid + teacher           : reservoir reads [s_k, y_{k-1}]
          - hybrid + closed_loop       : reservoir reads [s_k, y_hat_{k-1}]

        Closed-loop modes interleave evolve/predict/construct step-by-step.
        """
        # Legacy kwarg shim
        if 'y_test_scaled' in legacy_kwargs and y is None:
            y = legacy_kwargs.pop('y_test_scaled')
        y_tgt_override = legacy_kwargs.pop('y_target', None)
        if legacy_kwargs:
            raise TypeError(f"Unexpected kwargs: {list(legacy_kwargs)}")

        # Old positional compat
        if self.input_mode == 'autoregressive' and s is not None and y is None:
            y = s
            s = None

        # Decide whether we can run a single-pass (teacher-forced path)
        can_single_pass = (
            self.input_mode == 'exogenous'
            or self.feedback == 'teacher'
        )

        if can_single_pass:
            return self._test_teacher_forced(
                s, y, scaler, X_test_scaled, retrain,
                cache_dm_traj, y_tgt_override)
        else:
            return self._test_closed_loop(
                s, y, scaler, X_test_scaled, retrain, y_tgt_override)

    # ------------------------------------------------------------------ #
    # Teacher-forced / exogenous test: single-pass feature extraction
    # ------------------------------------------------------------------ #
    def _test_teacher_forced(self, s, y, scaler, X_test_scaled, retrain,
                             cache_dm_traj, y_tgt_override):
        # Build x_seq to match the convention in _build_train_sequence:
        #   x_seq has the same layout as training; lag is handled by the
        #   prepended last-training-input row below (via _compute_features).
        if self.input_mode == 'exogenous':
            if s is None: raise ValueError("exogenous test requires s")
            s_arr = np.asarray(s).reshape(len(s), -1)
            x_seq = s_arr
        elif self.input_mode == 'autoregressive':
            if y is None: raise ValueError("autoregressive teacher test requires y")
            y_arr = np.asarray(y).reshape(len(y), -1)
            x_seq = y_arr
        else:  # hybrid
            if s is None or y is None:
                raise ValueError("hybrid teacher test requires s and y")
            s_arr = np.asarray(s).reshape(len(s), -1)
            y_arr = np.asarray(y).reshape(len(y), -1)
            x_seq = np.hstack([s_arr, y_arr])

        # Continue reservoir from training: re-inject the last training input
        dm_cont = self._inject_encoding(
            self._partial_trace_enc(self.dm), self._last_x_input)

        # Feature extraction: the prepended row re-injects _last_x_input so
        # the first real step sees (state evolved) -> predict y_0 with
        # reservoir having just seen x_{T-1} (last training). Matches old code.
        y_ext = np.vstack([self._last_x_input.reshape(1, -1), x_seq])
        X_all = self._compute_features(y_ext, dm_init=dm_cont,
                                       cache_dm_traj=cache_dm_traj)
        X_feat = X_all[:-1]

        if X_test_scaled is not None:
            assert X_test_scaled.shape[0] == X_feat.shape[0]
            X_feat = np.hstack([X_feat, X_test_scaled])

        # Target for optional retrain / scaler shape
        y_tgt = y_tgt_override if y_tgt_override is not None else y
        if y_tgt is not None:
            y_tgt = np.asarray(y_tgt)
            if y_tgt.ndim == 1: y_tgt = y_tgt.reshape(-1, 1)
        out_dim = y_tgt.shape[1] if y_tgt is not None else self.dim_y

        y_pred = []
        for i in range(X_feat.shape[0]):
            row = X_feat[i:i+1]
            y_hat = self.lm.predict(row).reshape(1, -1)
            if scaler is not None:
                y_hat = scaler.inverse_transform(
                    np.clip(y_hat, 0.001, 0.999).reshape(1, out_dim))
            y_pred.append(y_hat.flatten())

            if retrain and y_tgt is not None:
                self.X_train = np.vstack([self.X_train, row])
                self.y_train_scaled = np.vstack(
                    [self.y_train_scaled, y_tgt[i].reshape(1, -1)])
                self.lm.fit(self.X_train, self.y_train_scaled)

        self.X_feat_test = X_feat
        return np.array(y_pred)

    # ------------------------------------------------------------------ #
    # Closed-loop test: step-by-step, y_hat fed back into reservoir
    # ------------------------------------------------------------------ #
    def _test_closed_loop(self, s, y, scaler, X_test_scaled, retrain,
                          y_tgt_override):
        y_warmup = np.asarray(self._last_y_true).reshape(-1)

        if self.input_mode == 'autoregressive':
            if y is None:
                raise ValueError("need y for length + (optional) retrain target")
            T_test = len(y)
            s_arr = None
        elif self.input_mode == 'hybrid':
            if s is None:
                raise ValueError("hybrid closed-loop test requires s")
            s_arr = np.asarray(s).reshape(len(s), -1)
            T_test = s_arr.shape[0]
        else:
            raise RuntimeError("closed-loop path shouldn't be entered for 'exogenous'")

        # Continue from training
        dm = self._inject_encoding(
            self._partial_trace_enc(self.dm), self._last_x_input)
        U_dag = self.U.conj().T

        y_fb = y_warmup.copy()  # y_{k-1} fed back at step 0
        y_pred_list = []

        y_tgt = y_tgt_override if y_tgt_override is not None else y
        if y_tgt is not None:
            y_tgt = np.asarray(y_tgt)
            if y_tgt.ndim == 1: y_tgt = y_tgt.reshape(-1, 1)
        out_dim = y_tgt.shape[1] if y_tgt is not None else self.dim_y

        for k in range(T_test):
            # Build reservoir input for this step
            if self.input_mode == 'autoregressive':
                x_k = y_fb.reshape(-1)
            else:  # hybrid
                x_k = np.concatenate([s_arr[k], y_fb.reshape(-1)])

            # If first step: state already continued + encoded with _last_x_input,
            # we need to RE-inject using our x_k instead (the first-step convention
            # in the teacher-forced path uses y_ext[0]=_last_x_input as "stale" and
            # the loop advances with y_ext[1]; here we bypass that).
            if k == 0:
                dm = self._inject_encoding(self._partial_trace_enc(dm), x_k)
            else:
                dm = self._inject_encoding(self._partial_trace_enc(dm), x_k)

            # Evolve Nv times, collect features from the last evolution
            feats = np.zeros(self.obs_array.shape[0] * self.Nv)
            for nv in range(self.Nv):
                dm = self.U @ dm @ U_dag
                feats[nv * self.obs_array.shape[0]:
                      (nv + 1) * self.obs_array.shape[0]] = np.tensordot(
                    self.obs_array, dm, axes=[[2, 1], [0, 1]]).real

            row = feats.reshape(1, -1)
            if X_test_scaled is not None:
                row = np.hstack([row, X_test_scaled[k:k+1]])

            y_hat = self.lm.predict(row).reshape(1, -1)
            if scaler is not None:
                y_hat_out = scaler.inverse_transform(
                    np.clip(y_hat, 0.001, 0.999).reshape(1, out_dim))
            else:
                y_hat_out = y_hat
            y_pred_list.append(y_hat_out.flatten())

            # Feedback for next step uses the (unscaled) prediction, which lives
            # in the same space as training y (scaled). This matches what the
            # reservoir saw at training time.
            y_fb = y_hat.flatten()

            if retrain and y_tgt is not None:
                self.X_train = np.vstack([self.X_train, row])
                self.y_train_scaled = np.vstack(
                    [self.y_train_scaled, y_tgt[k].reshape(1, -1)])
                self.lm.fit(self.X_train, self.y_train_scaled)

        self.dm = dm
        return np.array(y_pred_list)