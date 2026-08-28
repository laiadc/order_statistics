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
├── Exp - Quantum Hardware.ipynb    # Hardware ORS measurements
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
│   └── QRC/
│       ├── NARMA/narma_QRC.py            # Cluster script: QRC on NARMA-10
│       └── emsig/emsig_QRC.py           # Cluster script: QRC on EMSIG energy data
│
├── data/
│   ├── LiH/                  # Ground-state energies and bond lengths
│   ├── H2O/                  # Ground-state energies and bond lengths
│   └── time_series/          # NARMA inputs, EMSIG energy demand, weather, FX
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
    └── time_series/
        └── NARMA/           # QRC on NARMA-10 (3 qubits, 100 runs)
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

Dependencies: `numpy`, `qiskit ≥ 2.3`, `qiskit-aer`, `matplotlib`, `scikit-learn`, `tqdm`, `networkx`, `pandas`.

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
