"""Hardware facts (verified) versus modelling assumptions for the three EuroHPC QPUs.

Everything that the Exp4 feasibility study states about the physical machines lives here,
split into two registries:

* ``FACTS``        — statements checked against a vendor SDK, vendor documentation, an
                     official press release, or the repository data (with source + kind).
* ``ASSUMPTIONS``  — modelling choices that could NOT be verified before QPU access, with
                     their expected impact and how QPU access resolves them.

Vendor targets exported by ``experiments/native_hardware/vendor_export.py`` (run in the
vendor environment, qiskit 1.x) are loaded from ``data/hardware/*.json``.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import pandas as pd

ACCESS_DATE = "2026-09-29"
REPO_ROOT = Path(__file__).resolve().parents[1]
HARDWARE_DIR = REPO_ROOT / "data" / "hardware"
IQM_TARGET_FILE = HARDWARE_DIR / "iqm_crystal54.json"
AQT_TARGET_FILE = HARDWARE_DIR / "aqt_offline20.json"


# ── Platforms ─────────────────────────────────────────────────────────────────
PLATFORMS = {
    "iqm": dict(
        machine="Euro-Q-Exa", host="LRZ, Garching (DE)", vendor="IQM",
        system="IQM Radiance (54 qubits; 150-qubit system planned end 2026)",
        modality="superconducting, gate-based",
        capacity="54 qubits", connectivity="square lattice (IQM Crystal)",
        native_ops="PRX(θ,φ) [qiskit 'r'], CZ, measure",
        sdk="iqm-client[qiskit] 35.0.3 (vendor env)",
    ),
    "aqt": dict(
        machine="Piast-Q", host="PCSS, Poznań (PL)", vendor="AQT",
        system="AQT trapped-ion quantum computer (⁴⁰Ca⁺)",
        modality="trapped ions, gate-based",
        capacity="20 qubits", connectivity="all-to-all",
        native_ops="R(θ,φ), RZ(φ), RXX(θ) [Mølmer–Sørensen], measure (end of circuit)",
        sdk="qiskit-aqt-provider 1.15.0 (vendor env)",
    ),
    "quandela": dict(
        machine="Lucy", host="CEA/TGCC (FR)", vendor="Quandela",
        system="Quandela MOSAIQ-12",
        modality="photonic, linear optics with single photons",
        capacity="24 modes, 12 data qubits (dual-rail)",
        connectivity="all-to-all (universal programmable interferometer)",
        native_ops="beam splitters, phase shifters (MZI mesh), single-photon input, photon counting",
        sdk="perceval-quandela 1.3.0",
    ),
}

# ── Sources ───────────────────────────────────────────────────────────────────
SRC = {
    "lrz": "https://www.lrz.de/en/news/detail/first-european-quantum-computer-for-germany-euro-q-exa-starts-operation-at-lrz",
    "tqi_euroqexa": "https://thequantuminsider.com/2026/02/12/europe-launches-euro-q-exa-quantum-computer-germany/",
    "iqm_docs": "https://docs.iqm.tech/iqm-client/user_guide_qiskit.html",
    "iqm_json": "data/hardware/iqm_crystal54.json (IQMFakeAphrodite, iqm-client 35.0.3)",
    "iqm_target_src": "iqm/qiskit_iqm/iqm_target.py (iqm-client 35.0.3): reset only if QPU reports cc_prx",
    "braket_garnet": "https://aws.amazon.com/blogs/quantum-computing/experiment-with-dynamic-circuits-on-iqm-garnet-with-amazon-braket",
    "piastq_eu": "https://qt.eu/news/2025/2025-06-23_inauguration-of-piast-q",
    "piastq_aqt": "https://www.aqt.eu/inauguration-of-piast-q/",
    "aqt_json": "data/hardware/aqt_offline20.json (qiskit-aqt-provider 1.15.0, aqt-connector 0.3.0)",
    "aqt_docs": "https://qiskit-community.github.io/qiskit-aqt-provider/v1/guide.html",
    "lucy": "https://www.quandela.com/qpus/lucy/",
    "cea_lucy": "https://www.cea.fr/english/Pages/News/Quandela-delivers-Lucy-photonic-quantum-computer-at-CEA-TGCC.aspx",
    "belenos": "https://www.quandela.com/products-and-services/belenos/",
    "perceval": "https://perceval.quandela.net/docs/v1.3/",
    "paper": "arXiv:2605.10471 (López Carreño et al., 2026)",
    "lih_data": "data/LiH/ground_states_LiH.npy (checked in native_reservoirs.tasks)",
}

FACTS = [
    # IQM / Euro-Q-Exa
    ("iqm", "Euro-Q-Exa is an IQM Radiance system with 54 superconducting qubits, in operation at LRZ since Feb 2026; a second 150-qubit IQM system is planned for end 2026.", "press", SRC["lrz"] + " ; " + SRC["tqi_euroqexa"]),
    ("iqm", "IQM Qiskit backends expose the native operations r (PRX(θ,φ) = exp(-iθ/2 (cosφ X + sinφ Y)), identical to qiskit RGate), cz and measure; id is not executed.", "vendor docs", SRC["iqm_docs"]),
    ("iqm", "The Crystal-54 reference target has 54 qubits, 90 couplers (square lattice) and native ops {r, cz, measure, id, delay}; reset is not in the target.", "vendor SDK", SRC["iqm_json"]),
    ("iqm", "The IQM SDK adds 'reset'/'if_else' to a target only when the QPU reports cc_prx (dynamic circuits); dynamic circuits are demonstrated on IQM Garnet.", "vendor SDK + vendor blog", SRC["iqm_target_src"] + " ; " + SRC["braket_garnet"]),
    # AQT / Piast-Q
    ("aqt", "Piast-Q is an AQT laser-based trapped-ion quantum computer hosted by PCSS (Poznań) with 20 physical qubits and all-to-all connectivity (inaugurated June 2025).", "press", SRC["piastq_eu"] + " ; " + SRC["piastq_aqt"]),
    ("aqt", "AQT native ops are r, rz, rxx, measure. API angle ranges: R θ∈[0,π], φ∈[0,2π]; RXX θ∈[0,π/2]; RZ unconstrained.", "vendor SDK", SRC["aqt_json"]),
    ("aqt", "The AQT API accepts measurements only at the end of the circuit and rejects reset (no mid-circuit measurement/reset through qiskit-aqt-provider).", "vendor SDK", SRC["aqt_json"]),
    ("aqt", "The AQT offline simulators (ideal / noisy) expose 20 qubits with no coupling map (all-to-all).", "vendor SDK", SRC["aqt_json"]),
    # Quandela / Lucy
    ("quandela", "Lucy is a Quandela MOSAIQ-12 photonic quantum computer at CEA/TGCC: 24 modes, 12 data qubits, universal photonic chip, all-to-all connectivity; programmable via Perceval and MerLin.", "vendor docs + press", SRC["lucy"] + " ; " + SRC["cea_lucy"]),
    ("quandela", "The Belenos system of the same MOSAIQ-12 line runs 12 photons in 24 modes (dual-rail).", "vendor docs", SRC["belenos"]),
    ("quandela", "Perceval models linear-optical circuits (BS, PS, arbitrary unitaries) acting on Fock states; SLOS computes exact output distributions.", "vendor docs", SRC["perceval"]),
    ("quandela", "Photonic quantum reservoir processing on a programmable chip: data in a subset of chip phases, a fixed random interferometer, photon-number-resolved coincidences post-selected on photon number, linear ridge readout.", "literature", SRC["paper"]),
    # Data
    ("data", "LiH ground states populate only 10 of the 28 Hamming-weight-2 bitstrings; singlet pairs a18=-a33, a66=-a129, a96=-a144 hold exactly; global sign is arbitrary per state.", "repository data", SRC["lih_data"]),
]

ASSUMPTIONS = [
    ("iqm", "Euro-Q-Exa's chip follows IQM's Crystal-54 reference map (IQMFakeAphrodite); the chosen 6-qubit patch is operational.",
     "Patch/edges might differ on the real device (dead qubits/couplers).", "Select the patch from live calibration data; the patch finder takes any coupling map."),
    ("iqm", "The SDK's sample error profile stands in for Euro-Q-Exa calibration (used only for f_est and noisy emulation).",
     "Absolute fidelity estimates are indicative only.", "Replace with measured calibration; compare physical vs noise-aware runs."),
    ("iqm", "Mid-circuit reset (measure + cc_prx) is available on Euro-Q-Exa — needed ONLY for the secondary 'reset window' QRC protocol; the main QRC protocol (no-reset window) uses final measurements only.",
     "Without reset, only the no-reset window protocol runs.", "Check target for 'reset' at access time."),
    ("iqm", "Only PRX and CZ are used (no RZ/virtual-Z assumed).", "Conservative: 1Q counts may be slightly higher than necessary.", "Confirm available virtual-Z support."),
    ("aqt", "Gates execute serially on the ion chain; execution length = number of physical (non-virtual) operations.",
     "Depth comparisons with IQM use the parallel depth only as a secondary figure.", "Measure wall-clock per circuit."),
    ("aqt", "RZ is a virtual (frame-update) operation with no error or duration.", "1Q error budget counts only R gates.", "Confirm with AQT calibration data."),
    ("aqt", "Main QRC protocol (no-reset window of 2 steps, all ions measured at the end) satisfies the verified API constraint; the secondary reset-window protocol would need one fresh ion per time step (W + n_hidden ≤ 20), because the API has no mid-circuit reset.",
     "Reset-window results are a simulation of fresh-ion multiplexing (exactly equivalent to trace-and-reinject).", "Validate fresh-ion multiplexing on hardware."),
    ("aqt", "Piast-Q error rates are approximated by vendor-published AQT figures for f_est (not a Piast-Q calibration).", "Indicative only.", "Replace with measured calibration."),
    ("quandela", "Lucy runs the reservoir with 3 single photons in 12 modes (the paper uses 2 photons); the 3-photon event rate follows the Ascella-calibrated model R_n ≈ (80 MHz / n)·η^n with η ≈ 0.08.",
     "All photonic time estimates scale with this rate.", "Measure the 3-photon coincidence rate on Lucy."),
    ("all", "Throughput (shots per second) figures are indicative, taken from other systems of the same vendors: IQM ≈ 500 shots/s (derived from IQM Garnet CLOPS ≈ 2600), AQT ≈ 5.6 circuit executions/s (IBEX Q1 '>20,000 circuits per hour', treated as single shots), Lucy 3-photon events from the rate model above.",
     "Every minute/shot estimate in the hardware plan scales with these numbers.", "Ask each provider for the actual repetition rate."),
    ("quandela", "Photon-number-resolving detection; a threshold-detection variant is provided.", "Pattern features depend on detector type.", "Confirm detector configuration."),
    ("quandela", "Ideal simulation: lossless, perfectly indistinguishable photons, post-selected on n detected photons.",
     "Real losses/distinguishability reduce contrast and rate.", "Noise-aware Perceval simulation + hardware runs."),
    ("quandela", "Input |1,0,0,0,1,0,0,0,1,0,0,0⟩ (photons in modes 0, 4, 8) from a demultiplexed quantum-dot source; input states are user-selectable (Perceval with_input).", "Other input patterns change the reservoir.", "Configure input modes on Lucy."),
    ("quandela", "Any m×m unitary (m ≤ 24) of the random mesh is programmable on the universal chip.", "Reservoir mesh depth is a model parameter, not chip depth.", "Compile with the vendor toolchain."),
    ("quandela", "Haar-dialled mesh: MZI reflectivities sampled from Beta(1, α(c,k)), α = min(2c+1, 2(m-1-c)+1, 2(k+1), 2(m-1-k)); rule derived here by decomposing Haar unitaries for this mesh/MZI convention and verified statistically (not taken from a vendor).", "Full-depth mesh is Haar on U(m) (verified: KS tests, E|TrU|²=1, ORS_LO≈0).", "—"),
    ("quandela", "No optical quantum memory: photonic QRC memory is an input window of W=2 steps (same as the main gate-based protocol).", "Memory horizon fixed by W.", "—"),
    ("all", "Emulator scale: 6 encoding sites per platform; results are ideal (noise-free) unless stated.", "Hardware scale and noise not yet exercised.", "EuroHPC QPU access."),
]


def facts_table() -> pd.DataFrame:
    return pd.DataFrame(FACTS, columns=["platform", "fact", "kind", "source"])


def assumptions_table() -> pd.DataFrame:
    return pd.DataFrame(ASSUMPTIONS, columns=["platform", "assumption", "impact", "resolved_by"])


def platforms_table() -> pd.DataFrame:
    return pd.DataFrame.from_dict(PLATFORMS, orient="index")


@lru_cache(maxsize=None)
def _load(path: str):
    p = Path(path)
    if not p.exists():
        return None
    return json.loads(p.read_text())


def load_iqm_target() -> dict | None:
    """Exported IQM Crystal-54 target (coupling map, native ops, sample error profile)."""
    return _load(str(IQM_TARGET_FILE))


def load_aqt_target() -> dict | None:
    """Exported AQT offline-simulator target (qubits, native ops, API angle constraints)."""
    return _load(str(AQT_TARGET_FILE))


# Vendor-published AQT figures (IBEX Q1 product page, used ONLY as a stand-in, see ASSUMPTIONS)
AQT_PUBLISHED_FIDELITY = {"1q": 0.99966, "2q": 0.987, "readout": None,   # readout not published there
                          "source": "https://www.aqt.eu/products/ibex-q1/ (IBEX Q1 figures, not Piast-Q)"}


# Indicative throughput used for the hardware plan (see ASSUMPTIONS); parallel = independent
# 6-site reservoirs that fit on the device at once (IQM 54 qubits → 9, AQT 20 ions → 3, Lucy → 1).
INDICATIVE_RATES = {
    "iqm": {"per_s": 500.0, "parallel": 9, "unit": "shots",
            "source": "derived from IQM Garnet CLOPS ≈ 2600 (arXiv:2408.12433)"},
    "aqt": {"per_s": 20000 / 3600, "parallel": 3, "unit": "shots",
            "source": "AQT IBEX Q1 '>20,000 circuits per hour' (aqt.eu), treated as single shots"},
    "quandela": {"per_s": 80e6 / 3 * 0.08 ** 3, "parallel": 1, "unit": "3-photon events",
                 "source": "R_n = (80 MHz/n)·η^n, η = 0.08 — reproduces Ascella's 4 Hz (6 ph.) and 50 Hz (5 ph.), arXiv:2306.00874"},
}
