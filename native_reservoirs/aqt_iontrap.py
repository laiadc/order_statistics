"""AQT (Piast-Q) native trapped-ion reservoir.

Native operations (verified from the AQT SDK / API models, ``data/hardware/aqt_offline20.json``):
``r`` = R(θ, φ) with θ ∈ [0, π], φ ∈ [0, 2π]; ``rz`` (unconstrained); ``rxx`` = RXX(θ)
(Mølmer–Sørensen type) with θ ∈ [0, π/2]; ``measure`` only at the end of the circuit.
Connectivity: all-to-all (20 ions), so the entangling graph is free.

Reservoir (complexity parameter: entangling-graph ``density`` ρ)::

    choose the entangling graph E once per instance:
        'random'    : exactly M = max(1, round(ρ·C(n,2))) pairs, uniformly (G(n, M))
        'power_law' : all pairs, strength w_ij = |i-j|^(-α) / max  (long-range ion-chain pattern)
    for ℓ in 1..n_layers:
        R(θ_q, φ_q) and RZ(λ_q) on every ion             θ ~ U[0,π], φ, λ ~ U[0,2π)
        RXX(θ_ij) on every (i, j) ∈ E                     θ_ij ∈ (0, π/2]  (×w_ij for power law)

Several random instances per density level differ in both graph and angles.
"""

from __future__ import annotations

from itertools import combinations
from math import comb

import numpy as np
from qiskit import QuantumCircuit

from .base import GateNativeReservoir
from .hardware_specs import load_aqt_target

_TOL = 1e-12


def _api_bounds():
    """Angle bounds (radians) from the exported AQT API models, with documented defaults."""
    t = load_aqt_target()
    if t is None:
        return {"r.theta": (0.0, np.pi), "r.phi": (0.0, 2 * np.pi), "rxx.theta": (0.0, np.pi / 2)}
    c = t["api_angle_constraints"]
    return {k: (c[k]["min_rad"], c[k]["max_rad"]) for k in ("r.theta", "r.phi", "rxx.theta")}


class AQTIonTrapReservoir(GateNativeReservoir):
    """Hardware-native reservoir for AQT trapped-ion processors (Piast-Q).

    Parameters
    ----------
    n_qubits : int
        Number of ions (6 in Exp4; Piast-Q has 20).
    density : float
        Entangling-graph density ρ ∈ (0, 1] (``pattern='random'``).
    n_layers : int
        Number of (single-qubit layer + entangling layer) repetitions (default 2).
    pattern : {'random', 'power_law'}
    alpha : float
        Power-law exponent for ``pattern='power_law'`` (α = 0: uniform all-to-all).
    seed : int
    """

    platform_key = "aqt"
    NATIVE_OPS = ("r", "rz", "rxx")
    ONE_QUBIT_OPS = ("r", "rz")
    TWO_QUBIT_OPS = ("rxx",)
    VIRTUAL_OPS = ("rz",)
    complexity_name = "density"

    def __init__(self, n_qubits: int = 6, density: float = 0.5, n_layers: int = 2,
                 pattern: str = "random", alpha: float | None = None, seed: int = 0):
        super().__init__(n_qubits, seed)
        if pattern not in ("random", "power_law"):
            raise ValueError("pattern must be 'random' or 'power_law'")
        self.density = float(density)
        self.n_layers = int(n_layers)
        self.pattern = pattern
        self.alpha = alpha
        self.edges = None                 # all-to-all: any pair allowed
        self.graph = []                   # entangling graph of this instance
        self.weights = {}
        self.bounds = _api_bounds()

    @property
    def complexity_value(self):
        return self.density if self.pattern == "random" else self.alpha

    @property
    def graph_density(self) -> float:
        """Fraction of the C(n,2) pairs that carry an RXX gate (unweighted)."""
        return len(self.graph) / comb(self.n_qubits, 2)

    @property
    def weighted_density(self) -> float:
        return sum(self.weights.values()) / comb(self.n_qubits, 2)

    def _choose_graph(self, rng):
        pairs = list(combinations(range(self.n_qubits), 2))
        if self.pattern == "random":
            M = max(1, int(round(self.density * len(pairs))))
            idx = np.sort(rng.choice(len(pairs), size=M, replace=False))
            graph = [pairs[i] for i in idx]
            weights = {e: 1.0 for e in graph}
        else:
            a = 0.0 if self.alpha is None else float(self.alpha)
            w = {e: float(abs(e[0] - e[1])) ** (-a) for e in pairs}
            wmax = max(w.values())
            graph, weights = pairs, {e: v / wmax for e, v in w.items()}
        return graph, weights

    def _build_circuit(self, rng):
        self.graph, self.weights = self._choose_graph(rng)
        th_lo, th_hi = self.bounds["r.theta"]
        xx_hi = self.bounds["rxx.theta"][1]
        qc = QuantumCircuit(self.n_qubits, name=f"AQT_rho{self.density:g}")
        for _ in range(self.n_layers):
            for q in range(self.n_qubits):
                qc.r(rng.uniform(th_lo, th_hi), rng.uniform(0.0, 2 * np.pi), q)
                qc.rz(rng.uniform(0.0, 2 * np.pi), q)
            for (i, j) in self.graph:
                theta = xx_hi * (1.0 - rng.random())          # (0, π/2]
                qc.rxx(theta * self.weights[(i, j)], i, j)
        return qc

    def _check_params(self, name, params):
        if name == "r":
            (lo, hi), (plo, phi) = self.bounds["r.theta"], self.bounds["r.phi"]
            if not (lo - _TOL <= params[0] <= hi + _TOL and plo - _TOL <= params[1] <= phi + _TOL):
                raise ValueError(f"R angles {params} outside AQT API range")
        elif name == "rxx":
            lo, hi = self.bounds["rxx.theta"]
            if not (lo - _TOL <= params[0] <= hi + _TOL):
                raise ValueError(f"RXX angle {params[0]} outside AQT API range")

    def metadata(self) -> dict:
        md = super().metadata()
        md.update({
            "pattern": self.pattern, "alpha": self.alpha, "n_layers": self.n_layers,
            "entangling_graph": [list(e) for e in self.graph],
            "graph_density": self.graph_density, "weighted_density": self.weighted_density,
            "api_angle_bounds_rad": {k: list(v) for k, v in self.bounds.items()},
        })
        return md
