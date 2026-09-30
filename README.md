# Diagnosing Quantum Reservoirs at Scale: Expressivity and Coverage

Code and data for the paper **"Diagnosing quantum reservoirs at scale based on expressivity and coverage"**: https://arxiv.org/abs/2607.09445.

We introduce scalable, task-independent diagnostics, the **Order-Statistics Score (ORS)** and the **effective rank** ($R_\text{eff}$), and validate them against task performance across synthetic, and real-world benchmarks of quantum reservoir computing and quantum extreme learning machines.

---

## Repository Structure

```
Expressivity/
├── metrics.py              # Core metrics: ORS, Lorentz curves, input sensitivity
├── QELM.py                 # QELM wrapper (static)
├── QRC.py                  # QRC wrapper (time-series)
├── lanczos.py              # Krylov complexity helpers
├── functions_d.py          # D-dimensional auxiliary functions
│
├── Exp1_ORS_analysis.ipynb         # Exp 1 — ORS validation: KL, Krylov, scale
├── Exp2_ORS_Reff.ipynb             # Exp 2 — ORS gap + R_eff vs task performance
├── Exp3_real_data.ipynb            # Exp 3 — LiH chemistry + EMSIG energy demand
├── Exp4_native_hardware_feasibility.ipynb  # Exp 4 — hardware-native reservoirs (EuroHPC feasibility)
├── Exp - Quantum Hardware.ipynb    # Hardware ORS measurements
│
├── native_reservoirs/              # Exp 4 package (imports QELM.py / QRC.py / metrics.py unchanged)
│   ├── base.py                     # NativeReservoir interface, GateNativeReservoir, derive_seed
│   ├── hardware_specs.py           # verified hardware FACTS vs modelling ASSUMPTIONS (+ sources)
│   ├── iqm_lattice.py              # Euro-Q-Exa: PRX+CZ reservoir on a Crystal-54 patch
│   ├── aqt_iontrap.py              # Piast-Q: R/RZ/RXX reservoir, tunable entangling density
│   ├── quandela_photonic.py        # Lucy: Perceval MZI-mesh reservoir, photon counting
│   ├── adapters.py                 # NativeQELM / NativeQRC / PhotonicQRC (subclass existing classes)
│   ├── resources.py                # native resource metrics, routing checks, sampling cost
│   ├── diagnostics.py              # ORS (original), ORS_LO (photonic analogue), R_eff
│   ├── tasks.py                    # LiH weights + energy data, Exp3 readout protocols
│   ├── experiment.py               # seeded sweep runner, caching, shot budget, noise checks
│   ├── noisy_emulation.py          # task performance on vendor noisy emulators + Perceval noise
│   └── plotting.py                 # Exp4 figures and tables
├── tests/test_native_reservoirs.py # pytest suite for the Exp4 package
├── requirements-vendor.txt         # pins for the separate vendor-SDK environment (qiskit 1.x)
│
├── experiments/
│   ├── ORS/
│   │   └── run_ORS_noisy_multibasis.py   # Cluster script: noisy multi-basis ORS sweep
│   ├── QELM/
│   │   ├── Fourier/
│   │   │   ├── sine_QELM.py              # Cluster script: QELM Fourier regression sweep
│   │   │   └── qelm_utils.py             # Shared: run_qelm_experiment, generate_target, R_eff
│   │   └── chemistry/
│   │       └── run_quantum_chemistry.py  # Cluster script: QELM on LiH / H₂O
│   ├── QRC/
│   │   ├── NARMA/narma_QRC.py            # Cluster script: QRC on NARMA-10
│   │   └── emsig/emsig_QRC.py           # Cluster script: QRC on EMSIG energy data
│   └── native_hardware/
│       ├── vendor_export.py              # (vendor env) export IQM / AQT targets → data/hardware/
│       ├── vendor_validate.py            # (vendor env) transpile/run Exp4 circuits on vendor targets
│       └── vendor_tasks.py               # (vendor env) run the LiH / energy task circuits on noisy emulators
│
├── data/
│   ├── LiH/                  # Ground-state energies and bond lengths
│   ├── H2O/                  # Ground-state energies and bond lengths
│   ├── time_series/          # NARMA inputs, EMSIG energy demand, weather, FX
│   └── hardware/             # exported vendor targets (IQM Crystal-54, AQT offline 20 ions)
│
└── results/
    ├── ors/
    │   ├── results_experiment1_krylov.npz   # Cached KL/ORS/Krylov cross-validation
    │   └── noisy_basis/                     # Multi-basis noisy ORS gap files (.npz)
    ├── qelm/
    │   ├── sine/            # QELM Fourier results (keys: ors, mse, reff, n_obs_list)
    │   └── chemistry/
    │       ├── LiH/         # QELM on LiH (8 qubits, 100 runs)
    │       └── H2O/         # QELM on H₂O (10 qubits, 100 runs)
    ├── time_series/
    │   └── NARMA/           # QRC on NARMA-10 (3 qubits, 100 runs)
    └── native_hardware/     # Exp 4: records.csv, ors.csv, metadata.json, vendor/, tables/
```

---

## Circuit Families

| Family | Description | Sweep parameter |
|--------|-------------|-----------------|
| `G1` | Clifford gates {CNOT, H, S} | number of gates |
| `G2` | Universal gates {CNOT, H, T, Rz} | number of gates |
| `G3` | Near-Haar gates {CNOT, H, T} | number of gates |
| `D2_random` | IQP-like diagonal gates on random 2-local graph | expected degree |
| `D2_random_XZ` | D2 with X/Z alternation | expected degree |
| `D2_random_XZY` | D2 with X/Z/Y alternation | expected degree |
| `Haar` | Haar-random unitary | — |

---

## Experiments

### Exp 1 — ORS Validation (`Exp1_ORS_analysis.ipynb`)
Cross-validates ORS against KL divergence, Krylov complexity, and the level-spacing chaos indicator at `n=6`. Demonstrates ORS remains tractable at `n=20` where KL breaks down. Establishes the noisy, multi-basis ORS gap as the canonical expressivity measure.

### Exp 2 — ORS Gap + R_eff vs Task Performance (`Exp2_ORS_Reff.ipynb`)
Two benchmarks with `ns=100` reservoir instances each:

- **QELM Fourier regression** (`n=6`): Fixed reservoir + linear readout on a random Fourier series. ORS gap and $R_\text{eff}$ are swept over gate counts (G families) and degrees (D2 families).
- **NARMA-10 time series** (`n=4`): QRC with exponential encoding on a nonlinear autoregressive benchmark.

ORS gaps are loaded from `results/ors/noisy_basis/` (multi-basis, `f=1`, `K=4`). $R_\text{eff}$ is computed from the feature matrix SVD.

### Exp 3 — Real Data (`Exp3_real_data.ipynb`)
Two experiments on physically meaningful tasks:

- **LiH molecular chemistry** (`n=8`, QELM): Ground-state energy prediction along the dissociation curve. Metrics: MSE and ORS gap.
- **EMSIG energy demand** (`n=4`, QRC): Autoregressive forecasting of real-world electricity consumption. Metric: test $R^2$.

### Exp 4 — Hardware-native reservoirs for EuroHPC (`Exp4_native_hardware_feasibility.ipynb`)
A simulation-based feasibility study for a EuroHPC Quantum Access proposal. It builds the workflow **native reservoir → native resource metrics → ORS / R_eff → QELM/QRC performance** for three machines:

| Machine | Native reservoir | Complexity sweep |
|---|---|---|
| **Euro-Q-Exa** (IQM, superconducting) | PRX + CZ cycles on a 6-qubit patch of the IQM Crystal-54 coupling map. Every CZ is on a physical coupler, so there are 0 SWAPs. | cycles L |
| **Piast-Q** (AQT, trapped ions) | 4 layers of R + RZ (virtual) + RXX (Mølmer–Sørensen) on an all-to-all graph of exact density ρ; a power-law variant is also available. | density ρ |
| **Lucy** (Quandela, photonic) | Perceval: 3 single photons in 12 modes (inputs 0, 4, 8). The data are written into 6 phase shifters between two *Haar-dialled* MZI meshes of depth D (each a Haar U(12) at full depth), followed by photon counting. This follows arXiv:2605.10471, which uses 2 photons. | mesh depth D |

- **One reservoir instance per seed.** Each instance (6 encoding sites on every platform) gives the ORS on the bare reservoir, the LiH QELM result and the energy QRC result.
- **LiH input.** The circuit receives *symmetry-reduced configuration weights* rather than the full state. Particle-number and singlet symmetry leave 6 independent weights, encoded natively as θ ∈ [0, π]. There are also learning curves (interpolation only) for the number of training geometries needed for chemical accuracy.
- **Energy data.** The energy QRC uses Exp3's series (`data/time_series/emsig_energy_data_by_ems.csv`, ems_3, 15 min). This file is large and gitignored; it must be present locally.
- **QRC protocols.** The main QRC protocol runs on hardware without mid-circuit operations: one circuit per time step over a window of 2 inputs, with all qubits/modes measured at the end. A window of 8 with resets and Exp3's continuous dynamics (idealised) are reported for comparison, together with classical AR baselines under the same readout protocol.
- **Hardware plan: inference only.** With 10–25 min of QPU time per provider, only inference is run on the hardware. The readout is trained on a noise-aware digital twin: the same circuits, the vendor error model and the same shot count. The notebook reports accuracy vs shots and what fits in 15 min, using indicative (unverified) shot rates.
- **ORS on photonic outputs.** The gate-based ORS is unchanged. For photonic outputs, where the Porter–Thomas reference does not apply, a separate calibrated analogue **ORS_LO** is used: the same functional, with the dimension replaced by an effective dimension fitted to Haar-random interferometers.
- **Facts vs assumptions.** Verified hardware facts and modelling assumptions are listed separately in `native_reservoirs/hardware_specs.py` and in the notebook.

### Quantum Hardware (`Exp - Quantum Hardware.ipynb`)
ORS gap measured on physical quantum hardware, using the noisy ORS extension with calibrated fidelity `f`.

---

## Key Metrics

**Order-Statistics Score (ORS, Λ)** — measures how close the output probability distribution is to the Haar (maximally expressive) distribution using only the top-$K$ sorted probabilities. The *ORS gap* $\Lambda - \Lambda_\text{Haar}$ is zero for Haar-random circuits and negative for less expressive ensembles. Scales to `n=25`+ qubits.

**Effective Rank ($R_\text{eff}$)** — participation ratio of singular values of the centred feature matrix:
$$R_\text{eff} = \frac{(\sum_i \sigma_i)^2}{\sum_i \sigma_i^2}$$
Measures how many independent directions the reservoir exposes to the linear readout. Task-conditioned counterpart to ORS.

---

## Installation

```bash
# Requires Python ≥ 3.13 and uv
uv sync
```

Dependencies: `numpy`, `qiskit ≥ 2.3`, `qiskit-aer`, `matplotlib`, `scikit-learn`, `scipy`, `mpmath`, `tqdm`, `networkx`, `pandas`, `joblib`, `perceval-quandela ≥ 1.3` (Exp 4), and `pytest` (dev).

The notebooks run on the shared research environment one level up (`../.venv`, kernel `Python 3.13 (QRC research)`, see `../ENVIRONMENT.md`). To add the Exp 4 dependencies there:

```bash
uv pip install --python ../.venv/bin/python "perceval-quandela>=1.3,<2" pytest
```

## Reproducing Exp 4 (hardware-native feasibility study)

```bash
# 1. main environment: tests (≈3 s)
../.venv/bin/python -m pytest tests/test_native_reservoirs.py -q

# 2. optional vendor-SDK environment (iqm-client needs qiskit<2.2, qiskit-aqt-provider needs qiskit<2)
uv venv --python 3.13 ../.venv-vendor
uv pip install --python ../.venv-vendor/bin/python -r requirements-vendor.txt
../.venv-vendor/bin/python experiments/native_hardware/vendor_export.py      # → data/hardware/*.json

# 3. run the notebook (loads cached results/native_hardware/ unless FORCE=True in the first cell)
../.venv/bin/jupyter nbconvert --to notebook --execute --inplace \
    --ExecutePreprocessor.kernel_name=qrc --ExecutePreprocessor.timeout=-1 Exp4_native_hardware_feasibility.ipynb
```

- **Runtime.** A full recomputation (`FORCE = True`) takes about an hour on 16 cores. With the cache present, the notebook runs in under a minute.
- **Data.** The energy task needs `data/time_series/emsig_energy_data_by_ems.csv` (gitignored EMSIG export used by Exp 3).
- **Seeds.** Every reservoir instance is seeded with `derive_seed(base_seed=2026, platform, level, instance)`. Seeds, the configuration hash, package versions and the git SHA are written to `results/native_hardware/metadata.json`.
- **Vendor steps inside the notebook.** The notebook calls `vendor_validate.py` and `vendor_tasks.py` through `subprocess` when `../.venv-vendor` exists, and otherwise uses the cached `results/native_hardware/vendor/` files. `vendor_tasks.py` runs about 32k task circuits on IQM's fake Crystal-54 backend (~3 min) and AQT's noisy offline simulator (~7 min with `--workers 12`).

---

## Running Cluster Experiments

Each script under `experiments/` is a self-contained CLI intended for HPC submission. Example:

```bash
# ORS multi-basis sweep (all n in {6,8,10,12,16})
python experiments/ORS/run_ORS_noisy_multibasis.py \
    --gates-name G1 --num-gates 50 --ns 100

# QELM Fourier regression
python experiments/QELM/Fourier/sine_QELM.py \
    --family-gates G1 --num-gates 50 --ns 100

# QELM chemistry (LiH)
python experiments/QELM/chemistry/run_quantum_chemistry.py \
    --family-gates G1 --depth 50
```
