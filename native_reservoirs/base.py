"""Common interface for hardware-native quantum reservoirs (Exp4).

Every platform implementation (IQM lattice, AQT ion trap, Quandela photonic) exposes the
same lightweight API::

    res = SomeNativeReservoir(..., seed=s).build()
    res.run(...)                 # exact output states / distributions
    res.extract_features(X)      # (N, F) feature matrix for the linear readout
    res.resource_metrics()       # platform-native resource counts
    res.metadata()               # topology, native ops, layout, complexity, seed

Gate-based platforms share :class:`GateNativeReservoir`, which holds a qiskit circuit that
uses only the platform's native operations, validates it against the native basis and the
hardware connectivity, and plugs it into the *existing* QELM / QRC classes through the
adapters in :mod:`native_reservoirs.adapters` (no re-implementation of the readout).
"""

from __future__ import annotations

import hashlib
from abc import ABC, abstractmethod

import numpy as np

from .hardware_specs import PLATFORMS


def derive_seed(*keys) -> int:
    """Deterministic 32-bit seed from a tuple of ints / strings (via ``np.random.SeedSequence``).

    Used as ``derive_seed(base_seed, platform, level, instance)`` so that every reservoir
    instance has its own reproducible random stream, independent of execution order.
    """
    ints = []
    for k in keys:
        if isinstance(k, (int, np.integer)):
            ints.append(int(k) & 0xFFFFFFFF)
        else:
            ints.append(int.from_bytes(hashlib.sha256(str(k).encode()).digest()[:4], "little"))
    return int(np.random.SeedSequence(ints).generate_state(1)[0])


class NativeReservoir(ABC):
    """Abstract hardware-native reservoir."""

    platform_key: str = ""          # 'iqm' | 'aqt' | 'quandela'
    modality: str = ""              # 'gate' | 'photonic'
    NATIVE_OPS: tuple = ()
    complexity_name: str = ""       # name of the platform-native complexity parameter

    def __init__(self, seed: int = 0):
        self.seed = int(seed)
        self._built = False

    # -- lifecycle -------------------------------------------------------------
    @abstractmethod
    def build(self) -> "NativeReservoir":
        """Construct the native circuit from ``self.seed``; returns ``self``."""

    def _require_built(self):
        if not self._built:
            raise RuntimeError(f"{type(self).__name__}: call .build() first")

    # -- evaluation ------------------------------------------------------------
    @abstractmethod
    def run(self, inputs=None):
        """Exact output of the reservoir (state vector or output distribution)."""

    @abstractmethod
    def extract_features(self, inputs, task: str = "lih", **kw) -> np.ndarray:
        """Feature matrix (N, F) for the linear readout."""

    # -- description -----------------------------------------------------------
    @property
    @abstractmethod
    def complexity_value(self):
        """Value of the platform-native complexity parameter."""

    @property
    @abstractmethod
    def n_sites(self) -> int:
        """Number of encoding sites (qubits, or dual-rail photons)."""

    @abstractmethod
    def resource_metrics(self) -> dict:
        """Platform-native resource counts."""

    def metadata(self) -> dict:
        info = PLATFORMS.get(self.platform_key, {})
        return {
            "platform": self.platform_key,
            "machine": info.get("machine"),
            "modality": self.modality,
            "class": type(self).__name__,
            "native_ops": list(self.NATIVE_OPS),
            "complexity_name": self.complexity_name,
            "complexity_value": self.complexity_value,
            "n_sites": self.n_sites,
            "seed": self.seed,
        }


class GateNativeReservoir(NativeReservoir):
    """Shared logic for gate-based native reservoirs (qiskit circuits in the native basis).

    Subclasses implement :meth:`_build_circuit` and set ``NATIVE_OPS``, ``ONE_QUBIT_OPS``,
    ``TWO_QUBIT_OPS`` (and optionally ``VIRTUAL_OPS``) plus ``self.edges`` — the allowed
    logical 2-qubit pairs (``None`` means all-to-all).
    """

    modality = "gate"
    ONE_QUBIT_OPS: tuple = ()
    TWO_QUBIT_OPS: tuple = ()
    VIRTUAL_OPS: tuple = ()

    def __init__(self, n_qubits: int, seed: int = 0):
        super().__init__(seed)
        self.n_qubits = int(n_qubits)
        self.circuit = None
        self.edges = None           # list[tuple[int,int]] of allowed logical pairs, or None
        self.layout = list(range(self.n_qubits))   # logical -> physical qubit index
        self._U = None

    @property
    def n_sites(self) -> int:
        return self.n_qubits

    # -- construction ----------------------------------------------------------
    @abstractmethod
    def _build_circuit(self, rng: np.random.Generator):
        """Return a qiskit ``QuantumCircuit`` built only from ``NATIVE_OPS``."""

    def build(self) -> "GateNativeReservoir":
        rng = np.random.default_rng(self.seed)
        self.circuit = self._build_circuit(rng)
        self._U = None
        self.validate()
        self._built = True
        return self

    def validate(self):
        """Assert native-op compliance, connectivity and parameter ranges."""
        allowed = set(self.NATIVE_OPS) | {"barrier"}
        edge_set = None if self.edges is None else {frozenset(e) for e in self.edges}
        for inst in self.circuit.data:
            name = inst.operation.name
            if name not in allowed:
                raise ValueError(f"non-native operation '{name}' for {self.platform_key}")
            qubits = [self.circuit.find_bit(q).index for q in inst.qubits]
            if len(qubits) == 2 and edge_set is not None and frozenset(qubits) not in edge_set:
                raise ValueError(f"2-qubit gate on uncoupled pair {qubits}")
            self._check_params(name, [float(p) for p in inst.operation.params if np.ndim(p) == 0])

    def _check_params(self, name, params):
        """Hook for platform-specific angle-range checks (default: none)."""

    # -- evaluation ------------------------------------------------------------
    def unitary(self) -> np.ndarray:
        self._require_built()
        if self._U is None:
            from qiskit.quantum_info import Operator
            self._U = Operator(self.circuit).data
        return self._U

    def run(self, inputs=None):
        """Apply the reservoir unitary to state vector(s) (default: |0…0⟩)."""
        U = self.unitary()
        if inputs is None:
            psi = np.zeros(2 ** self.n_qubits, dtype=complex)
            psi[0] = 1.0
            return U @ psi
        psi = np.asarray(inputs, dtype=complex)
        return psi @ U.T if psi.ndim == 2 else U @ psi

    def circuit_factory(self):
        """Zero-argument factory returning the bare reservoir circuit (for metrics.py ORS)."""
        self._require_built()
        return lambda: self.circuit.copy()

    def as_qelm(self, **kw):
        """Wrap as :class:`native_reservoirs.adapters.NativeQELM` (existing QELM machinery)."""
        from .adapters import NativeQELM
        self._require_built()
        return NativeQELM(self.circuit, **kw)

    def as_qrc(self, **kw):
        """Wrap as :class:`native_reservoirs.adapters.NativeQRC` (existing QRC machinery)."""
        from .adapters import NativeQRC
        self._require_built()
        return NativeQRC(self.circuit, **kw)

    def as_windowed_qrc(self, window: int = 2, mode: str = "reupload", **kw):
        """Hardware-executable windowed QRC (:class:`~native_reservoirs.adapters.WindowedQRC`)."""
        from .adapters import GateWindowFeatures, WindowedQRC
        self._require_built()
        return WindowedQRC(GateWindowFeatures(self.circuit, mode=mode), window=window, **kw)

    def extract_features(self, inputs, task: str = "lih", **kw) -> np.ndarray:
        """LiH: 1-local Pauli features after native weight encoding (via NativeQELM).

        Temporal (QRC) features are sequential and produced by ``as_qrc().train/test``.
        """
        if task != "lih":
            raise ValueError("gate-based temporal features are produced by as_qrc().train/test")
        kw.setdefault("encoding", "weights_ry")
        kw.setdefault("observables_type", [1])
        return self.as_qelm(**kw).feature_matrix(inputs)

    # -- description -----------------------------------------------------------
    def export_ops(self) -> list:
        """JSON-serialisable op list [{name, qubits, params}] (logical qubit indices)."""
        self._require_built()
        ops = []
        for inst in self.circuit.data:
            ops.append({
                "name": inst.operation.name,
                "qubits": [self.circuit.find_bit(q).index for q in inst.qubits],
                "params": [float(p) for p in inst.operation.params if np.ndim(p) == 0],
            })
        return ops

    def resource_metrics(self) -> dict:
        from .resources import gate_resources
        self._require_built()
        return gate_resources(self)

    def metadata(self) -> dict:
        md = super().metadata()
        md.update({
            "n_qubits": self.n_qubits,
            "edges": None if self.edges is None else [list(e) for e in self.edges],
            "layout_physical": list(self.layout),
        })
        return md
