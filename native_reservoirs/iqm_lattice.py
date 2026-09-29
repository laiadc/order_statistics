"""IQM (Euro-Q-Exa) native lattice reservoir.

Native operations (verified, IQM Qiskit target): ``r`` = PRX(θ, φ) and ``cz`` (+ ``measure``).
Connectivity: the Crystal-54 reference coupling map exported from the IQM SDK
(``data/hardware/iqm_crystal54.json``). The reservoir is placed on a connected *patch* of that
map, and every CZ acts on a physical coupler, so no routing / SWAPs are ever required.

Reservoir (complexity parameter: number of cycles ``n_layers`` = L)::

    for ℓ in 1..L:
        PRX(θ_q, φ_q) on every qubit           θ ~ U[0, π], φ ~ U[0, 2π)
        CZ on colour class (ℓ mod C) of a proper edge colouring of the patch

Edges in one colour class share no qubit, so each CZ layer is a single parallel layer.
Logical qubit 0 is a maximum-degree node of the patch (it is the QRC input qubit).

Native state preparation used by the tasks (all unit-tested):
    H·Rz(-θ)|0⟩ ≅ PRX(π/2, π/2 - θ)|0⟩   (Exp3 'pauli' encoding, up to global phase)
    |+⟩ = PRX(π/2, π/2)|0⟩                 (QRC hidden-qubit initialisation)
    RY(θ) = PRX(θ, π/2)                    (LiH configuration-weight encoding)
"""

from __future__ import annotations

import warnings
from itertools import combinations

import networkx as nx
import numpy as np
from qiskit import QuantumCircuit

from .base import GateNativeReservoir
from .hardware_specs import load_iqm_target


# ── Hardware graph + patch selection ──────────────────────────────────────────
def hardware_graph(target: dict | None = None) -> tuple[nx.Graph, dict]:
    """Coupling graph of the exported IQM target (fallback: 6x9 square grid, flagged)."""
    target = target if target is not None else load_iqm_target()
    if target is None:
        warnings.warn("data/hardware/iqm_crystal54.json not found: falling back to a 6x9 "
                      "square-grid model (run experiments/native_hardware/vendor_export.py).")
        G = nx.convert_node_labels_to_integers(nx.grid_2d_graph(6, 9), ordering="sorted")
        info = {"source": "square-grid fallback (not vendor data)", "qubit_names": [f"Q{i}" for i in G]}
        return G, info
    G = nx.Graph()
    G.add_nodes_from(range(target["num_qubits"]))
    G.add_edges_from(tuple(e) for e in target["coupling_edges"])
    return G, target


def _node_error(target: dict, q: int) -> float:
    ep = target.get("error_profile") if target else None
    if not ep:
        return 0.0
    name = target["qubit_names"][q]
    ro = ep["readout_errors"].get(name, {"0": 0.0, "1": 0.0})
    return ep["prx_depolarizing"].get(name, 0.0) + 0.5 * (ro["0"] + ro["1"])


def _edge_error(target: dict, a: int, b: int) -> float:
    ep = target.get("error_profile") if target else None
    if not ep:
        return 0.0
    na, nb = target["qubit_names"][a], target["qubit_names"][b]
    cz = ep["cz_depolarizing"]
    return cz.get(f"{na}-{nb}", cz.get(f"{nb}-{na}", 0.0))


def patch_error(G: nx.Graph, nodes, target: dict | None) -> float:
    """Sum of PRX + readout errors on the nodes and CZ errors on the induced couplers."""
    sub = G.subgraph(nodes)
    return (sum(_node_error(target, q) for q in nodes)
            + sum(_edge_error(target, a, b) for a, b in sub.edges()))


def _connected_subsets(G: nx.Graph, k: int):
    """ESU enumeration of all connected induced k-node subgraphs."""
    def extend(sub, ext, v):
        if len(sub) == k:
            yield frozenset(sub)
            return
        ext = set(ext)
        while ext:
            w = ext.pop()
            new = {u for u in G[w] if u > v and u not in sub and all(u not in G[s] for s in sub)}
            yield from extend(sub | {w}, ext | new, v)
    for v in G.nodes():
        yield from extend({v}, {u for u in G[v] if u > v}, v)


def select_patch(n: int, G: nx.Graph | None = None, target: dict | None = None,
                 exhaustive_max: int = 8) -> list[int]:
    """Deterministic n-qubit patch: max #couplers, then min error, then lowest indices.

    Exhaustive (ESU) for n ≤ ``exhaustive_max``; greedy growth for larger n.
    Returns physical indices ordered logically (index 0 = max-degree node, then BFS order).
    """
    if G is None:
        G, target = hardware_graph(target)
    if n > G.number_of_nodes():
        raise ValueError(f"patch of {n} qubits > {G.number_of_nodes()} available")
    if n == G.number_of_nodes():
        nodes = set(G.nodes())
    elif n <= exhaustive_max:
        best, best_key = None, None
        for s in _connected_subsets(G, n):
            key = (-G.subgraph(s).number_of_edges(), round(patch_error(G, s, target), 12), tuple(sorted(s)))
            if best_key is None or key < best_key:
                best, best_key = s, key
        nodes = set(best)
    else:
        start = min(G.nodes(), key=lambda q: (-G.degree(q), _node_error(target, q), q))
        nodes = {start}
        while len(nodes) < n:
            frontier = {u for s in nodes for u in G[s]} - nodes
            nodes.add(min(frontier, key=lambda u: (-sum(1 for s in G[u] if s in nodes),
                                                    _node_error(target, u), u)))
    sub = G.subgraph(nodes)
    root = min(nodes, key=lambda q: (-sub.degree(q), _node_error(target, q), q))
    dist = nx.single_source_shortest_path_length(sub, root)
    return sorted(nodes, key=lambda q: (dist[q], q))


def edge_colouring(edges) -> list[list[tuple[int, int]]]:
    """Proper edge colouring (greedy on the line graph); each class is a parallel CZ layer."""
    edges = sorted(tuple(sorted(e)) for e in edges)
    if not edges:
        return []
    L = nx.Graph()
    L.add_nodes_from(edges)
    for e, f in combinations(edges, 2):
        if set(e) & set(f):
            L.add_edge(e, f)
    col = nx.greedy_color(L, strategy="largest_first")
    classes = {}
    for e, c in col.items():
        classes.setdefault(c, []).append(e)
    return [sorted(classes[c]) for c in sorted(classes)]


# ── Reservoir ─────────────────────────────────────────────────────────────────
class IQMLatticeReservoir(GateNativeReservoir):
    """Hardware-native reservoir for IQM Crystal lattices (Euro-Q-Exa).

    Parameters
    ----------
    n_qubits : int
        Patch size (6 in Exp4).
    n_layers : int
        Number of PRX + CZ-colour-class cycles (platform-native complexity).
    patch : list[int] or None
        Physical qubits (logical order). ``None`` → :func:`select_patch` on the exported map.
    target : dict or None
        Exported IQM target (default: ``data/hardware/iqm_crystal54.json``).
    seed : int
    """

    platform_key = "iqm"
    NATIVE_OPS = ("r", "cz")
    ONE_QUBIT_OPS = ("r",)
    TWO_QUBIT_OPS = ("cz",)
    complexity_name = "n_layers"

    def __init__(self, n_qubits: int = 6, n_layers: int = 4, patch=None, target=None, seed: int = 0):
        super().__init__(n_qubits, seed)
        self.n_layers = int(n_layers)
        self.G_hw, self.target = hardware_graph(target)
        self.layout = list(patch) if patch is not None else select_patch(n_qubits, self.G_hw, self.target)
        if len(self.layout) != n_qubits:
            raise ValueError("patch size != n_qubits")
        phys2log = {p: i for i, p in enumerate(self.layout)}
        self.edges = sorted(tuple(sorted((phys2log[a], phys2log[b])))
                            for a, b in self.G_hw.subgraph(self.layout).edges())
        self.colour_classes = edge_colouring(self.edges)

    @property
    def complexity_value(self):
        return self.n_layers

    @property
    def hardware_edges(self):
        return sorted(tuple(sorted(e)) for e in self.G_hw.edges())

    def _build_circuit(self, rng):
        qc = QuantumCircuit(self.n_qubits, name=f"IQM_L{self.n_layers}")
        n_cls = max(1, len(self.colour_classes))
        for layer in range(self.n_layers):
            thetas = rng.uniform(0.0, np.pi, self.n_qubits)
            phis = rng.uniform(0.0, 2 * np.pi, self.n_qubits)
            for q in range(self.n_qubits):
                qc.r(thetas[q], phis[q], q)
            for a, b in (self.colour_classes[layer % n_cls] if self.colour_classes else []):
                qc.cz(a, b)
        return qc

    def metadata(self) -> dict:
        md = super().metadata()
        names = self.target.get("qubit_names") if self.target else None
        md.update({
            "hardware_source": (self.target or {}).get("source", "square-grid fallback"),
            "patch_qubit_names": [names[p] for p in self.layout] if names else None,
            "patch_couplers": len(self.edges),
            "cz_colour_classes": [[list(e) for e in c] for c in self.colour_classes],
            "patch_error_score": patch_error(self.G_hw, self.layout, self.target),
        })
        return md
