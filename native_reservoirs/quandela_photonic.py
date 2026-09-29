"""Quandela (Lucy) photonic-native reservoir, implemented in Perceval.

Conceptual basis: photonic quantum reservoir processing on a programmable chip excited with
single photons (arXiv:2605.10471): data are written into a subset of the chip's phases, a
fixed random section of the MZI mesh acts as the reservoir, and photon-number-resolved
coincidences (post-selected on the photon number) feed a linear ridge readout.

This is NOT a gate-by-gate translation of the qubit reservoirs. Its ingredients are optical.

Default design (``encoding='phases'``, as in the paper):

* **Input**: n single photons spread over m modes (3 photons in 12 modes: |1,0,0,0,1,0,0,0,1,0,0,0⟩).
  Few photons keep the n-photon coincidence rate high (it drops by ~η per extra photon).
* **Circuit**: V(x) = U_res · E(x) · U_pre. U_pre and U_res are fixed random MZI meshes of depth D
  (the complexity parameter), and E(x) is a column of phase shifters on modes 0 … n_data-1 that
  carries the data. For LiH, the 6 weights go to 6 phase shifters, θ_j ∈ [0, π].
  For the windowed QRC: V_t = U_res E(u_t) U_res E(u_{t-1}) U_pre, with every data phase = π·u.
* **Meshes**: rectangular (brick) MZI meshes, each MZI = PS(φ_ext) · BS · PS(φ_int) · BS, followed by
  an output phase screen. With ``mesh_sampling='haar'`` (default) the splitting ratios are
  "Haar-dialled" (cf. Russell et al., NJP 19, 033007 (2017)): the reflectivity x = sin²(φ_int/2)
  of the MZI in column c on modes (k, k+1) is drawn from Beta(1, α) with

      α(c, k) = min(2c+1, 2(m-1-c)+1, 2(k+1), 2(m-1-k)),

  and all other phases uniformly, so a full-depth mesh (D = m) is Haar-distributed on U(m).
  The rule was derived for this mesh and MZI convention by decomposing Haar unitaries
  (:func:`rectangular_decomposition`) and is re-checked in the tests. ``haar=True`` uses
  Haar-random matrices instead of meshes.
* **Measurement**: photon-number-resolving detection of all m modes, post-selected on n photons;
  exact output distribution over C(m+n-1, n) patterns from Perceval's SLOS.
* **Features**: 1- and 2-body photon-number correlators ⟨n_i⟩, ⟨n_i n_j⟩ (default) or the full
  pattern distribution. First moments alone are blind to multi-photon interference
  (identical for distinguishable photons), see :meth:`first_moments_distinguishable`.

``encoding='dualrail'`` is the earlier design (one photon per dual-rail pair, one data MZI per
photon, no U_pre), kept for comparison.
"""

from __future__ import annotations

from itertools import combinations

import numpy as np
import perceval as pcvl
from perceval.utils import allstate_iterator
from scipy.stats import unitary_group

from .adapters import weights_to_angles
from .base import NativeReservoir

_BS = np.array(pcvl.BS().compute_unitary())          # Perceval default BS (Rx convention, 50:50)


def mzi_circuit(phi_int: float, phi_ext: float = 0.0) -> pcvl.Circuit:
    """Programmable MZI: PS(φ_ext) · BS · PS(φ_int) · BS on two modes (Perceval components)."""
    c = pcvl.Circuit(2, name="MZI")
    c.add(0, pcvl.PS(float(phi_ext))).add((0, 1), pcvl.BS()).add(0, pcvl.PS(float(phi_int))).add((0, 1), pcvl.BS())
    return c


def mzi_matrix(phi_int: float, phi_ext: float = 0.0) -> np.ndarray:
    """Numeric 2x2 unitary of :func:`mzi_circuit` (checked against Perceval in the tests)."""
    return _BS @ np.diag([np.exp(1j * phi_int), 1.0]) @ _BS @ np.diag([np.exp(1j * phi_ext), 1.0])


def haar_alpha(m: int, col: int, k: int) -> int:
    """Beta(1, α) exponent of the Haar-dialled reflectivity of the MZI at (column, upper mode k)."""
    return int(min(2 * col + 1, 2 * (m - 1 - col) + 1, 2 * (k + 1), 2 * (m - 1 - k)))


def rectangular_decomposition(U: np.ndarray):
    """Decompose U into the brick MZI mesh (Clements-type nulling, this module's MZI convention).

    Returns ``[(column, k, phi_int), ...]`` in input→output order. The internal phases
    (hence splitting ratios) and positions are exact; external/output phases are not
    returned (they do not affect the splitting-ratio statistics). Used to derive and test
    :func:`haar_alpha`.
    """
    U = np.array(U, dtype=complex)
    m = U.shape[0]
    right, left = [], []
    for i in range(m - 1):
        if i % 2 == 0:
            for j in range(i + 1):
                r, c = m - 1 - j, i - j
                a, b = U[r, c], U[r, c + 1]
                phi_i, phi_e = (np.pi, 0.0) if abs(a) < 1e-14 else (2 * np.arctan(abs(b / a)), -np.angle(-b / a))
                T = np.eye(m, dtype=complex)
                T[c:c + 2, c:c + 2] = mzi_matrix(phi_i, phi_e).conj().T
                U = U @ T
                right.append((c, phi_i))
        else:
            for j in range(1, i + 2):
                r, c = m + j - i - 2, j - 1
                k = r - 1
                a, b = U[k, c], U[r, c]
                phi_i, phi_e = (np.pi, 0.0) if abs(b) < 1e-14 else (2 * np.arctan(abs(a / b)), np.angle(b / a))
                T = np.eye(m, dtype=complex)
                T[k:k + 2, k:k + 2] = mzi_matrix(phi_i, phi_e)
                U = T @ U
                left.append((k, phi_i))
    if np.abs(U - np.diag(np.diag(U))).max() > 1e-8:
        raise RuntimeError("decomposition did not diagonalise U")
    last, placed = [-1] * m, []
    for k, phi in right + left[::-1]:
        col = max(last[k], last[k + 1]) + 1
        last[k] = last[k + 1] = col
        placed.append((col, k, phi))
    return placed


def encoding_phase(theta):
    """Internal MZI phase giving second-rail probability sin²(θ/2) for a photon in the first rail."""
    return np.pi - np.asarray(theta, dtype=float)


class QuandelaPhotonicReservoir(NativeReservoir):
    """Photonic-native reservoir for Quandela MOSAIQ-12 (Lucy).

    Parameters
    ----------
    n_modes, n_photons : int
        Optical modes m and single photons n (Exp4: 12 modes, 3 photons).
    mesh_depth : int
        MZI columns of each random mesh (platform-native complexity).
    window : int
        Number of data injections per circuit (1 for QELM, W for the windowed QRC).
    haar : bool
        Haar-random matrices instead of depth-limited meshes (reference ensemble).
    mesh_sampling : {'haar', 'uniform'}
        MZI parameter distribution ('haar': Haar-dialled splitting ratios, see module doc).
    encoding : {'phases', 'dualrail'}
        'phases' (default): data in ``n_data`` phase shifters between U_pre and U_res.
        'dualrail': one data MZI per photon pair, no U_pre.
    n_data : int
        Number of data phase shifters ('phases' encoding).
    detection : {'pnr', 'threshold'}
    seed : int
    """

    platform_key = "quandela"
    modality = "photonic"
    NATIVE_OPS = ("BS", "PS")
    complexity_name = "mesh_depth"

    def __init__(self, n_modes: int = 12, n_photons: int = 3, mesh_depth: int = 4, window: int = 1,
                 haar: bool = False, mesh_sampling: str = "haar", encoding: str = "phases",
                 n_data: int = 6, detection: str = "pnr", seed: int = 0):
        super().__init__(seed)
        if encoding not in ("phases", "dualrail"):
            raise ValueError("encoding must be 'phases' or 'dualrail'")
        if encoding == "dualrail" and n_modes < 2 * n_photons:
            raise ValueError("dual-rail layout needs n_modes >= 2 * n_photons")
        if mesh_sampling not in ("haar", "uniform"):
            raise ValueError("mesh_sampling must be 'haar' or 'uniform'")
        if mesh_sampling == "haar" and not haar and mesh_depth > n_modes:
            raise ValueError("Haar-dialled mesh defined for mesh_depth <= n_modes")
        self.n_modes, self.n_photons = int(n_modes), int(n_photons)
        self.mesh_depth, self.window = int(mesh_depth), int(window)
        self.haar, self.mesh_sampling, self.detection = bool(haar), mesh_sampling, detection
        self.encoding, self.n_data = encoding, int(n_data if encoding == "phases" else n_photons)
        if encoding == "phases":
            self.input_modes = [int(round(k * n_modes / n_photons)) for k in range(n_photons)]
        else:
            self.input_modes = [2 * k for k in range(n_photons)]
        occ = [0] * self.n_modes
        for k in self.input_modes:
            occ[k] = 1
        self.input_occupation = occ

    # -- description -----------------------------------------------------------
    @property
    def complexity_value(self):
        return "haar" if self.haar else self.mesh_depth

    @property
    def n_sites(self) -> int:
        return self.n_data

    @property
    def input_state_str(self) -> str:
        return "|" + ",".join(map(str, self.input_occupation)) + ">"

    @property
    def n_meshes(self) -> int:
        return 2 if self.encoding == "phases" else 1

    @property
    def mesh_mzi_count(self) -> int:
        """MZIs in ONE random mesh."""
        m = self.n_modes
        if self.haar:
            return m * (m - 1) // 2
        return sum(len(range(d % 2, m - 1, 2)) for d in range(self.mesh_depth))

    # -- construction ----------------------------------------------------------
    def _mesh(self, rng, name):
        m = self.n_modes
        if self.haar:
            U = unitary_group.rvs(m, random_state=rng)
            return pcvl.Circuit(m, name=name).add(0, pcvl.Unitary(pcvl.Matrix(U), name="Haar"))
        c = pcvl.Circuit(m, name=name)
        for d in range(self.mesh_depth):
            for k in range(d % 2, m - 1, 2):
                if self.mesh_sampling == "haar":
                    x = 1.0 - rng.random() ** (1.0 / haar_alpha(m, d, k))       # Beta(1, α) reflectivity
                    phi_int = 2.0 * np.arcsin(np.sqrt(x))
                else:
                    phi_int = rng.uniform(0, 2 * np.pi)
                c.add(k, mzi_circuit(phi_int, rng.uniform(0, 2 * np.pi)), merge=True)   # flat BS/PS
        for k in range(m):                                                       # output phase screen
            c.add(k, pcvl.PS(float(rng.uniform(0, 2 * np.pi))))
        return c

    def build(self) -> "QuandelaPhotonicReservoir":
        rng = np.random.default_rng(self.seed)
        m = self.n_modes
        self.reservoir_circuit = self._mesh(rng, "U_res")
        self.U_res = np.array(self.reservoir_circuit.compute_unitary(), dtype=complex)
        if self.encoding == "phases":
            self.pre_circuit = self._mesh(rng, "U_pre")
            self.U_pre = np.array(self.pre_circuit.compute_unitary(), dtype=complex)
        else:
            self.pre_circuit, self.U_pre = None, np.eye(m, dtype=complex)

        self.input_state = pcvl.BasicState(self.input_occupation)
        self.patterns = np.array([list(s) for s in allstate_iterator(self.input_state)], dtype=np.int16)
        pairs = list(combinations(range(m), 2))
        self.correlator_labels = [f"n{i}" for i in range(m)] + [f"n{i}n{j}" for i, j in pairs]
        P = self.patterns.astype(float)
        self._corr_design = np.hstack([P, np.stack([P[:, i] * P[:, j] for i, j in pairs], axis=1)])
        clicks = (self.patterns > 0).astype(np.int8)
        _, self._click_index = np.unique(clicks, axis=0, return_inverse=True)
        self._backend = pcvl.SLOSBackend()
        self._built = True
        return self

    # -- circuits --------------------------------------------------------------
    def encoding_unitary(self, thetas) -> np.ndarray:
        """Numeric m×m unitary of one data column."""
        thetas = np.broadcast_to(np.asarray(thetas, dtype=float), (self.n_data,))
        E = np.eye(self.n_modes, dtype=complex)
        if self.encoding == "phases":
            E[np.arange(self.n_data), np.arange(self.n_data)] = np.exp(1j * thetas)
        else:
            for k, th in enumerate(thetas):
                E[2 * k:2 * k + 2, 2 * k:2 * k + 2] = mzi_matrix(encoding_phase(th), 0.0)
        return E

    def encoding_circuit(self, thetas) -> pcvl.Circuit:
        thetas = np.broadcast_to(np.asarray(thetas, dtype=float), (self.n_data,))
        c = pcvl.Circuit(self.n_modes, name="E")
        for k, th in enumerate(thetas):
            if self.encoding == "phases":
                c.add(k, pcvl.PS(float(th)))
            else:
                c.add(2 * k, mzi_circuit(float(encoding_phase(th)), 0.0), merge=True)
        return c

    def circuit_unitary(self, thetas_per_slot) -> np.ndarray:
        """V = U_res E(θ_W) … U_res E(θ_1) U_pre."""
        V = self.U_pre.copy()
        for th in thetas_per_slot:
            V = self.U_res @ self.encoding_unitary(th) @ V
        return V

    def full_circuit(self, thetas_per_slot, barriers: bool = True) -> pcvl.Circuit:
        """Flat Perceval circuit U_pre | E(θ_1) | U_res | … | E(θ_W) | U_res, made of BS and PS only."""
        self._require_built()
        m = self.n_modes
        c = pcvl.Circuit(m, name="V")
        add_barrier = (lambda: c.add(0, pcvl.Barrier(m))) if barriers else (lambda: None)
        if self.pre_circuit is not None:
            c.add(0, self.pre_circuit, merge=True); add_barrier()
        for th in thetas_per_slot:
            c.add(0, self.encoding_circuit(th), merge=True); add_barrier()
            c.add(0, self.reservoir_circuit, merge=True); add_barrier()
        return c

    # -- simulation ------------------------------------------------------------
    def probabilities(self, V: np.ndarray) -> np.ndarray:
        """Exact n-photon output distribution of V|input⟩ (Perceval SLOS), in ``self.patterns`` order."""
        self._backend.set_circuit(pcvl.Unitary(pcvl.Matrix(V)))
        self._backend.set_input_state(self.input_state)
        return np.asarray(self._backend.all_prob(), dtype=float)

    def run(self, inputs=None, task: str = "lih", **kw) -> np.ndarray:
        """Output distributions.

        ``inputs=None`` → the bare reservoir U_res·U_pre on the input Fock state (used for ORS).
        ``task='lih'``  → rows of configuration weights (N, n_data).
        ``task='qrc'``  → rows of input windows (N, W) of scalar inputs in [0, 1].
        """
        self._require_built()
        if inputs is None:
            return self.probabilities(self.U_res @ self.U_pre)
        if task == "lih":
            thetas = weights_to_angles(inputs, kw.get("angle_map", "arcsin_sqrt"), kw.get("w_range"))
            return np.array([self.probabilities(self.circuit_unitary([t])) for t in thetas])
        if task == "qrc":
            return np.array(list(self._window_probs(np.asarray(inputs, float), kw.get("theta_scale", np.pi))))
        raise ValueError(f"unknown task {task!r}")

    def _window_probs(self, windows, theta_scale):
        cache = {}
        for row in windows:
            V = self.U_pre.copy()
            for u in row:                                   # oldest → newest
                key = float(u)
                M = cache.get(key)
                if M is None:
                    M = self.U_res @ self.encoding_unitary(theta_scale * key)
                    if len(cache) < 20000:
                        cache[key] = M
                V = M @ V
            yield self.probabilities(V)

    def features_from_probs(self, P: np.ndarray, kind: str = "correlators") -> np.ndarray:
        P = np.atleast_2d(P)
        if kind == "patterns":
            return P
        if kind == "correlators":
            return P @ self._corr_design
        if kind == "first_moments":
            return P @ self.patterns.astype(float)
        if kind == "threshold":
            out = np.zeros((P.shape[0], self._click_index.max() + 1))
            for i in range(P.shape[0]):
                out[i] = np.bincount(self._click_index, weights=P[i], minlength=out.shape[1])
            return out
        raise ValueError(f"unknown feature kind {kind!r}")

    def extract_features(self, inputs, task: str = "lih", features: str = "correlators", **kw) -> np.ndarray:
        if task == "qrc":
            return self.window_features(inputs, features=features, **kw)
        return self.features_from_probs(self.run(inputs, task=task, **kw), features)

    def window_features(self, windows, features: str = "correlators", theta_scale: float = np.pi,
                        chunk: int = 512) -> np.ndarray:
        """Features for a sequence of input windows (T, W) — used by :class:`PhotonicQRC`."""
        self._require_built()
        windows = np.asarray(windows, dtype=float)
        out, buf = [], []
        for p in self._window_probs(windows, theta_scale):
            buf.append(p)
            if len(buf) == chunk:
                out.append(self.features_from_probs(np.array(buf), features))
                buf = []
        if buf:
            out.append(self.features_from_probs(np.array(buf), features))
        return np.vstack(out)

    def first_moments_distinguishable(self, V: np.ndarray) -> np.ndarray:
        """⟨n_i⟩ for fully distinguishable photons: Σ_k |V_{i, p_k}|² (no interference)."""
        return (np.abs(V[:, self.input_modes]) ** 2).sum(axis=1)

    def noisy_probabilities(self, V: np.ndarray, noise: "pcvl.NoiseModel") -> np.ndarray:
        """Noise-aware distribution via a Perceval Processor, post-selected on n photons."""
        proc = pcvl.Processor("SLOS", pcvl.Unitary(pcvl.Matrix(V)), noise=noise)
        proc.with_input(self.input_state)
        proc.min_detected_photons_filter(self.n_photons)
        res = pcvl.algorithm.Sampler(proc).probs()["results"]
        index = {tuple(p): i for i, p in enumerate(self.patterns.tolist())}
        out = np.zeros(len(self.patterns))
        for state, prob in res.items():
            key = tuple(state)
            if key in index:
                out[index[key]] += prob
        s = out.sum()
        return out / s if s > 0 else out

    # -- description -----------------------------------------------------------
    def resource_metrics(self) -> dict:
        from .resources import photonic_resources
        self._require_built()
        return photonic_resources(self)

    def metadata(self) -> dict:
        md = super().metadata()
        md.update({
            "n_modes": self.n_modes, "n_photons": self.n_photons, "window": self.window,
            "mesh_depth": self.mesh_depth, "haar": self.haar, "mesh_sampling": self.mesh_sampling,
            "encoding": self.encoding, "n_data": self.n_data, "detection": self.detection,
            "input_state": self.input_state_str, "n_patterns": int(len(self.patterns)) if self._built else None,
        })
        return md
