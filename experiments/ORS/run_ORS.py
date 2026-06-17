import os
import argparse
import numpy as np
from tqdm import tqdm

from QELM import QuantumCircQiskit
from metrics import (
    lorentz_curves,
    order_statistics_score,
    lambda_haar_exact,
    lambda_haar_largeD,
)


# ── CLI ─────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument('--ns',         type=int, default=10)
parser.add_argument('--gates-name', type=str, default='G1')
parser.add_argument('--degree',     type=int, default=-1)
parser.add_argument('--num-gates',  type=int, default=-1)
parser.add_argument('--n-layers',   type=int, default=-1)
args = parser.parse_args()


# ── Config ──────────────────────────────────────────────────────────────────
RESULTS_FOLDER = "/data/cvcqml/common/laia/ORS/large_n"
os.makedirs(RESULTS_FOLDER, exist_ok=True)

ns         = args.ns
gates_name = args.gates_name
degree     = args.degree    if args.degree    != -1 else None
num_gates  = args.num_gates if args.num_gates != -1 else None
n_layers   = args.n_layers  if args.n_layers  != -1 else None

nshot_ors  = 10000
n_list     = [20, 25]
K_ors_list = [4, 8, 10]


def out_path():
    tag = gates_name
    if num_gates is not None: tag += f"_ng{num_gates}"
    if degree    is not None: tag += f"_deg{degree}"
    if n_layers  is not None: tag += f"_L{n_layers}"
    return os.path.join(RESULTS_FOLDER, f"{tag}_{ns}runs.npz")

def generate_reservoir():
    circ = QuantumCircQiskit(gates_name, nqbits=nqbits,
                             degree=degree, num_gates=num_gates, n_layers=n_layers,
                             statevector=True)
    return circ.get_reservoir()


# ── Run ─────────────────────────────────────────────────────────────────────
results     = {}                  # nqbits → (ns, len(K_ors_list)) array of Λ samples
lambda_haar = {}                  # nqbits → (len(K_ors_list),) analytical Λ_Haar (only if Haar)

for nqbits in n_list:
    D = 2 ** nqbits
    print(f"→ n={nqbits}  {gates_name}  degree={degree} ng={num_gates} L={n_layers}")
    
    _, sorted_probs, _ = lorentz_curves(generate_reservoir, nqbits,
                                        nshot_ors, ns, verbose=True)

    results[nqbits] = np.column_stack([
        order_statistics_score(sorted_probs, D, K=K) for K in K_ors_list
    ])

    if gates_name == 'Haar':
        # analytical Haar reference Λ per K: exact at small D, large-D asymptotic above
        lambda_haar[nqbits] = np.array([
            lambda_haar_exact(D, K) if nqbits < 8 else lambda_haar_largeD(D, K) #lambda_haar_exact(D, K) if nqbits < 8 else lambda_haar_largeD(D, K)
            for K in K_ors_list
        ])


# ── Save ────────────────────────────────────────────────────────────────────
to_save = dict(
    K_ors  = np.array(K_ors_list),
    nqbits = np.array(n_list),
    **{f"ors_n{n}": results[n] for n in n_list},
)
if gates_name == 'Haar':
    to_save.update({f"lambda_haar_n{n}": lambda_haar[n] for n in n_list})

path = out_path()
np.savez_compressed(path, **to_save)
print(f"Saved → {path}")