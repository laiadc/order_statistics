"""Adapters plugging native reservoirs into the EXISTING QELM / QRC classes.

Nothing in ``QELM.py`` / ``QRC.py`` is modified. The adapters subclass the existing classes
and override only the reservoir-construction hook (and, where needed, the encoding / feature
hooks), so that encoding conventions, observables, the density-matrix QRC dynamics,
teacher forcing, washout, Ridge readout and online retraining are exactly those used by
Exp2/Exp3.

* :class:`NativeQELM`  — ``QuantumCircQiskit`` with an externally supplied native circuit and
  an additional ``'weights_ry'`` encoding (θ_j = 2·arcsin√w_j on qubit j via PRX(θ_j, π/2)).
* :class:`NativeQRC`   — ``QuantumRC_Circuit`` with an externally supplied native circuit
  (Exp3's continuous density-matrix dynamics; an idealised reference).
* :class:`WindowedQRC` — ``QuantumRC_Circuit`` whose feature hooks are replaced by a windowed,
  measure-at-the-end protocol; ``train`` / ``test`` are inherited unchanged. Feature providers:
  :class:`GateWindowFeatures` (IQM/AQT), the photonic reservoir (:class:`PhotonicQRC`) and
  :class:`ClassicalWindowFeatures` (AR baselines).
"""

from __future__ import annotations

import numpy as np
from qiskit import QuantumCircuit

from QELM import QuantumCircQiskit
from QRC import QuantumRC_Circuit


# ── Angle maps for the LiH configuration-weight encoding ─────────────────────
def weights_to_angles(w, angle_map: str = "arcsin_sqrt", w_range=None) -> np.ndarray:
    """Map configuration weights w_j ∈ [0, 1] to encoding angles θ_j.

    ``'arcsin_sqrt'`` : θ = 2·arcsin(√w)   → the encoding site is excited with probability w
                        (RY(θ)|0⟩ = √(1-w)|0⟩ + √w|1⟩; photonic: second-rail probability w).
    ``'minmax_pi'``   : θ = π·(w - lo)/(hi - lo) with (lo, hi) = ``w_range`` per feature
                        (ablation; ``w_range`` must be fitted on training data).
    """
    w = np.clip(np.asarray(w, dtype=float), 0.0, 1.0)
    if angle_map == "arcsin_sqrt":
        return 2.0 * np.arcsin(np.sqrt(w))
    if angle_map == "minmax_pi":
        if w_range is None:
            raise ValueError("angle_map='minmax_pi' needs w_range=(lo, hi)")
        lo, hi = (np.asarray(a, dtype=float) for a in w_range)
        return np.pi * np.clip((w - lo) / np.where(hi > lo, hi - lo, 1.0), 0.0, 1.0)
    raise ValueError(f"unknown angle_map {angle_map!r}")


# ── QELM ──────────────────────────────────────────────────────────────────────
class NativeQELM(QuantumCircQiskit):
    """``QuantumCircQiskit`` driven by a hardware-native reservoir circuit.

    Parameters
    ----------
    reservoir_circuit : QuantumCircuit
        Native circuit (e.g. ``IQMLatticeReservoir(...).build().circuit``).
    encoding : str
        ``'weights_ry'`` (new, native: PRX(θ_j, π/2) = RY(θ_j) on qubit j) or any encoding of
        the parent class (``'pauli'``, ``'exponential'``, ``'quantum'``).
    angle_map, w_range
        See :func:`weights_to_angles` (only for ``'weights_ry'``).
    Other keyword arguments go to ``QuantumCircQiskit`` (``observables_type`` etc.).
    """

    def __init__(self, reservoir_circuit: QuantumCircuit, encoding: str = "weights_ry",
                 observables_type=(1,), angle_map: str = "arcsin_sqrt", w_range=None, **kw):
        self._native_circuit = reservoir_circuit
        self.angle_map = angle_map
        self.w_range = w_range
        kw.setdefault("statevector", True)
        super().__init__("native", nqbits=reservoir_circuit.num_qubits,
                         observables_type=list(observables_type), encoding=encoding, **kw)

    def _build_reservoir(self, degree, n_layers, scrambler_depth):
        return self._native_circuit.copy()

    def _angles(self, x):
        return weights_to_angles(np.asarray(x, dtype=float).reshape(-1), self.angle_map, self.w_range)

    def _encode(self, initial_state, qubit_offset=0, threshold=1e-6):
        if self.encoding != "weights_ry":
            return super()._encode(initial_state, qubit_offset, threshold)
        qc = QuantumCircuit(self.nqbits)
        for j, theta in enumerate(self._angles(initial_state)):
            qc.r(float(theta), np.pi / 2, qubit_offset + j)     # native PRX(θ, π/2) == RY(θ)
        return qc

    def _encode_state_fast(self, initial_state):
        if self.encoding != "weights_ry":
            return super()._encode_state_fast(initial_state)
        thetas = self._angles(initial_state)
        if len(thetas) > self.nqbits:
            return None
        single = np.tile(np.array([1.0, 0.0], dtype=complex), (self.nqbits, 1))
        single[: len(thetas), 0] = np.cos(thetas / 2)
        single[: len(thetas), 1] = np.sin(thetas / 2)
        psi = single[self.nqbits - 1]
        for q in range(self.nqbits - 2, -1, -1):     # qiskit little-endian
            psi = np.kron(psi, single[q])
        return psi


# ── QRC (gate-based) ──────────────────────────────────────────────────────────
class NativeQRC(QuantumRC_Circuit):
    """``QuantumRC_Circuit`` driven by a hardware-native reservoir circuit.

    The circuit acts on ``n_total = n_enc + nqbits`` qubits; logical qubit 0 is the input
    qubit (reset + re-encoded every step, Exp3 density-matrix dynamics), the others carry
    the memory. Defaults reproduce the Exp3 EMSIG configuration (Pauli encoding with
    θ = π·y, observables [1, 2] on all qubits).
    """

    def __init__(self, reservoir_circuit: QuantumCircuit, **kw):
        self._native_circuit = reservoir_circuit
        n_total = reservoir_circuit.num_qubits
        kw.setdefault("encoding", "pauli")
        kw.setdefault("k_exp", 1)
        kw.setdefault("theta_scale", np.pi)
        kw.setdefault("observables_type", [1, 2])
        kw.setdefault("n_measure", n_total)
        kw.setdefault("input_mode", "autoregressive")
        kw.setdefault("dim_y", 1)
        n_enc = kw["k_exp"] * kw["dim_y"]
        super().__init__(nqbits=n_total - n_enc, gates_set="native", **kw)

    def _build_reservoir(self):
        return self._native_circuit.copy()


# ── Windowed QRC: one circuit per time step, measured only at the end ─────────
class WindowedQRC(QuantumRC_Circuit):
    """Windowed QRC that reuses ``QuantumRC_Circuit.train/test`` unchanged.

    At time step t a *fresh* circuit processes the last ``W`` inputs u_{t-W+1}, …, u_t and all
    qubits / modes are measured at the end, so no mid-circuit measurement and no quantum memory
    between circuits are needed: the reservoir "state" is the input window itself. The actual
    circuits are supplied by a *feature provider* with ``window_features(windows, **kw)``:

    * :class:`GateWindowFeatures`  — IQM / AQT native circuits (modes 'reupload' or 'reset');
    * ``QuandelaPhotonicReservoir`` — photonic circuits (see :class:`PhotonicQRC`);
    * :class:`ClassicalWindowFeatures` — classical AR baselines with the same readout.

    The window plays the role of the density matrix in the parent's hooks:
    ``_partial_trace_enc`` is the identity on the window and ``_inject_encoding`` shifts it and
    appends the new input, so the teacher-forced ``test`` continuation (including the
    re-injection of the last training input) has exactly the parent's semantics.
    """

    def __init__(self, provider, window: int = 2, alpha: float = 1e-12, t_dismiss: int = 100,
                 predictor: str = "ridge", **feature_kw):
        # Deliberately NOT calling super().__init__: it would build a qubit reservoir.
        # Set exactly the attributes that the inherited train/test path uses.
        self.provider = provider
        self.window = int(window)
        self.feature_kw = feature_kw
        self.alpha = alpha
        self.t_dismiss = t_dismiss
        self.predictor = predictor
        self.lm = None
        self.input_mode = "autoregressive"
        self.feedback = "teacher"
        self.dim_s, self.dim_y, self.m = 0, 1, 1
        self.Nv = 1
        self.exec_mode = "window"
        self.n_enc = 1

    # -- parent hooks ----------------------------------------------------------
    def _fresh_window(self, y_vec):
        buf = np.zeros((self.window, self.m))
        buf[-1] = np.asarray(y_vec, dtype=float).reshape(-1)
        return buf

    def _partial_trace_enc(self, state):
        return state

    def _inject_encoding(self, state, y_vec):
        return np.vstack([state[1:], np.asarray(y_vec, dtype=float).reshape(1, -1)])

    def _compute_features(self, y_scaled, dm_init=None, cache_dm_traj=False):
        T = y_scaled.shape[0]
        state = dm_init if dm_init is not None else self._fresh_window(y_scaled[0])
        windows = np.empty((T, self.window))
        for t in range(T):
            windows[t] = state[:, 0]
            if t + 1 < T:
                state = self._inject_encoding(state, y_scaled[t + 1])
        self.dm = state
        return self.provider.window_features(windows, **self.feature_kw)

    def get_reservoir(self):
        return getattr(self.provider, "reservoir_circuit", None)


class PhotonicQRC(WindowedQRC):
    """Photonic windowed QRC: V_t = U_res E(u_t) U_res E(u_{t-1}) … U_res E(u_{t-W+1}).

    The same fixed interferometer ``U_res`` follows every injection (the photonic analogue of
    Exp3 applying the same U each step); features are photon-number correlators of
    V_t|1,0,1,0,…⟩. Photons do not persist and Lucy has no optical memory, so the window is
    the only memory.
    """

    def __init__(self, reservoir, window: int = 2, features: str = "correlators",
                 theta_scale: float = np.pi, **kw):
        super().__init__(reservoir, window=window, features=features, theta_scale=theta_scale, **kw)
        self.photonic = reservoir


def _prx(theta, phi):
    c, s = np.cos(theta / 2), np.sin(theta / 2)
    return np.array([[c, -1j * np.exp(-1j * phi) * s], [-1j * np.exp(1j * phi) * s, c]])


class GateWindowFeatures:
    """Window features for a gate-based native reservoir (all qubits measured at the end).

    Conventions are Exp3's (via :class:`NativeQRC`): logical qubit 0 is the input qubit, the
    input u is encoded by the native gate PRX(π/2, π/2 − θ·u) (≅ H·Rz(−θu)|0⟩), hidden qubits
    start in |+⟩ = PRX(π/2, π/2)|0⟩, and the features are all 1- and 2-local Paulis on all
    qubits (18 product-basis measurement settings on 6 qubits).

    mode='reupload' : the encoding gate is applied on top of the current input-qubit state
                      (no reset): one unitary circuit, only final measurements — executable
                      on IQM and on the AQT API as is.
    mode='reset'    : the input qubit is reset and re-encoded between steps, i.e. Exp3's
                      trace-and-reinject dynamics truncated to the window. Needs mid-circuit
                      reset (IQM dynamic circuits) or one fresh ion per step (AQT).
    """

    def __init__(self, reservoir_circuit: QuantumCircuit, mode: str = "reupload",
                 theta_scale: float = np.pi, observables_type=(1, 2)):
        if mode not in ("reupload", "reset"):
            raise ValueError("mode must be 'reupload' or 'reset'")
        self.mode, self.theta_scale = mode, theta_scale
        self.reservoir_circuit = reservoir_circuit
        self._q = NativeQRC(reservoir_circuit, observables_type=list(observables_type), theta_scale=theta_scale)
        self.U = self._q.U
        self.obs = self._q.obs_array
        self.n = reservoir_circuit.num_qubits

    def window_states(self, rows) -> np.ndarray:
        """Output states (T, 2^n) of the no-reset window circuits (qiskit little-endian)."""
        rows = np.asarray(rows, dtype=float)
        psi = np.zeros((len(rows), 2 ** self.n), dtype=complex)
        hidden = np.ones(2 ** (self.n - 1)) / np.sqrt(2 ** (self.n - 1))
        psi[:] = np.kron(hidden, np.array([1.0, 0.0]))
        for w in range(rows.shape[1]):
            G = np.stack([_prx(np.pi / 2, np.pi / 2 - self.theta_scale * u) for u in rows[:, w]])
            psi = np.einsum("tab,tcb->tca", G, psi.reshape(len(rows), -1, 2)).reshape(len(rows), -1)
            psi = psi @ self.U.T
        return psi

    def window_features(self, windows, chunk: int = 256, **kw) -> np.ndarray:
        windows = np.asarray(windows, dtype=float)
        if self.mode == "reset":
            return self._features_reset(windows)
        out = []
        for i in range(0, len(windows), chunk):
            psi = self.window_states(windows[i:i + chunk])
            Opsi = np.einsum("oab,tb->toa", self.obs, psi)
            out.append(np.real(np.einsum("ta,toa->to", psi.conj(), Opsi)))
        return np.vstack(out)

    def _features_reset(self, windows):
        q, U, Ud = self._q, self.U, self.U.conj().T
        out = np.empty((len(windows), self.obs.shape[0]))
        for i, row in enumerate(windows):
            dm = U @ q._fresh_initial_dm(np.array([row[0]])) @ Ud
            for u in row[1:]:
                dm = U @ q._inject_encoding(q._partial_trace_enc(dm), np.array([u])) @ Ud
            out[i] = np.tensordot(self.obs, dm, axes=[[2, 1], [0, 1]]).real
        return out

    def window_circuit(self, row, measure: bool = True) -> QuantumCircuit:
        """Native circuit of one time step (for resource counting and vendor validation)."""
        n = self.n
        qc = QuantumCircuit(n, n if measure else 0)
        for h in range(1, n):
            qc.r(np.pi / 2, np.pi / 2, h)                       # |+⟩ on hidden qubits
        for w, u in enumerate(row):
            if self.mode == "reset" and w > 0:
                qc.reset(0)
            qc.r(np.pi / 2, (np.pi / 2 - self.theta_scale * float(u)) % (2 * np.pi), 0)
            qc.compose(self.reservoir_circuit, inplace=True)
        if measure:
            qc.measure(range(n), range(n))
        return qc


class ClassicalWindowFeatures:
    """Classical baseline with the SAME readout protocol: the raw window (linear AR(W)),
    optionally with all pairwise products (quadratic AR(W))."""

    def __init__(self, kind: str = "linear"):
        if kind not in ("linear", "quadratic"):
            raise ValueError("kind must be 'linear' or 'quadratic'")
        self.kind = kind
        self.reservoir_circuit = None

    def window_features(self, windows, **kw) -> np.ndarray:
        X = np.asarray(windows, dtype=float)[:, ::-1]           # most recent input first
        if self.kind == "quadratic":
            i, j = np.triu_indices(X.shape[1])
            X = np.hstack([X, X[:, i] * X[:, j]])
        return X
