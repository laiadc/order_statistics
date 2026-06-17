import os, argparse
import numpy as np
from tqdm import tqdm
from QELM import QuantumCircQiskit
from metrics import (
    lorentz_curves_noisy,
    order_statistics_score_noisy,
    lambda_haar_largeD_noisy,
)

parser = argparse.ArgumentParser()
 
args = parser.parse_args()

RESULTS_FOLDER = "/data/cvcqml/common/laia/ORS/noise/"
os.makedirs(RESULTS_FOLDER, exist_ok=True)

ns         = args.ns
gates_name = args.gates_name
degree     = args.degree    if args.degree    != -1 else None
num_gates  = args.num_gates if args.num_gates != -1 else None
n_layers   = args.n_layers  if args.n_layers  != -1 else None

n_list     = [6, 8, 10, 12, 16, 20]            # dropped 25 — too expensive
K_ors_list = [4, 8, 10, 50]
K_max      = max(K_ors_list)
fs         = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]

np.random.seed(args.seed)

def out_path():
    tag = gates_name
    if num_gates is not None: tag += f"_ng{num_gates}"
    if degree    is not None: tag += f"_deg{degree}"
    if n_layers  is not None: tag += f"_L{n_layers}"
    return os.path.join(RESULTS_FOLDER, f"{tag}_{ns}runs_seed{args.seed}.npz")

def make_reservoir_fn(nqbits):
    def gen():
        circ = QuantumCircQiskit(gates_name, nqbits=nqbits,
                                 degree=degree, num_gates=num_gates,
                                 n_layers=n_layers, statevector=True)
        return circ.get_reservoir()
    return gen

results     = {}   # (n, f) -> (ns, len(K_ors_list))
haar_ref    = {}   # (n, f) -> (len(K_ors_list),), analytical, family-independent

for nqbits in tqdm(n_list, desc="n"):
    D = 2 ** nqbits
    for f in tqdm(fs, desc=f"f (n={nqbits})", leave=False):
        _, sorted_probs, _ = lorentz_curves_noisy(
            make_reservoir_fn(nqbits), nqbits, nshots=0,
            ns=ns, fidelity=f, statevector=True, verbose=False,
        )
        # Truncate to top-K_max to save memory.
        top = sorted_probs[:, :K_max].copy()
        results[(nqbits, f)] = np.column_stack([
            order_statistics_score_noisy(top, D, K=K, f=f)
            for K in K_ors_list
        ])
        # Always compute the Haar reference — it's analytical and free.
        haar_ref[(nqbits, f)] = np.array([
            lambda_haar_largeD_noisy(D, K, f) for K in K_ors_list
        ])

to_save = dict(
    K_ors  = np.array(K_ors_list),
    nqbits = np.array(n_list),
    fs     = np.array(fs),
)
for n in n_list:
    for f in fs:
        # Encode f in the key (npz keys are strings).
        ftag = f"{f:.2f}".replace('.', 'p')
        to_save[f"ors_n{n}_f{ftag}"]   = results[(n, f)]
        to_save[f"haar_n{n}_f{ftag}"]  = haar_ref[(n, f)]

path = out_path()
np.savez_compressed(path, **to_save)
print(f"Saved → {path}")