import os
import argparse
import numpy as np
from tqdm import tqdm

from QELM import QuantumCircQiskit
from metrics import (
    lorentz_curves_noisy,
    order_statistics_score_noisy,
    lambda_haar_largeD_noisy,
)

# ── CLI ─────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--ns", type=int, default=10)
parser.add_argument("--gates-name", type=str, default="G1")
parser.add_argument("--degree", type=int, default=-1)
parser.add_argument("--num-gates", type=int, default=-1)
parser.add_argument("--n-layers", type=int, default=-1)
parser.add_argument("--seed", type=int, default=1234)
parser.add_argument("--basis-seed", type=int, default=4321)
args = parser.parse_args()

# ── Config ──────────────────────────────────────────────────────────────────
RESULTS_FOLDER = "/data/cvcqml/common/laia/ORS/noise_multibasis/"
os.makedirs(RESULTS_FOLDER, exist_ok=True)

ns = args.ns
gates_name = args.gates_name
degree = args.degree if args.degree != -1 else None
num_gates = args.num_gates if args.num_gates != -1 else None
n_layers = args.n_layers if args.n_layers != -1 else None

n_list = [6, 8, 10, 12, 16]
K_ors_list = [4, 8]
K_max = max(K_ors_list)

fs = [1.0, 0.5, 0.1]

fixed_basis = ["Z", "X", "Y"]
B_rand = 50

basis_rng = np.random.default_rng(args.basis_seed)

# ── Helpers ─────────────────────────────────────────────────────────────────
def ftag_from_float(f):
    return f"{f:.2f}".replace(".", "p")


def random_local_pauli_basis(nqbits, rng):
    return rng.choice(["X", "Y", "Z"], size=nqbits).tolist()


def out_path():
    tag = gates_name

    if num_gates is not None:
        tag += f"_ng{num_gates}"
    if degree is not None:
        tag += f"_deg{degree}"
    if n_layers is not None:
        tag += f"_L{n_layers}"

    return os.path.join(
        RESULTS_FOLDER,
        f"{tag}_{ns}runs_seed{args.seed}_basis{args.basis_seed}_noise_multibasis.npz",
    )


def reservoir_seed(nqbits):
    return args.seed + 10_000 * int(nqbits)


def make_reservoir_fn(nqbits):
    def gen():
        circ = QuantumCircQiskit(
            gates_name,
            nqbits=nqbits,
            degree=degree,
            num_gates=num_gates,
            n_layers=n_layers,
            statevector=True,
        )
        return circ.get_reservoir()

    return gen


def run_noisy_ors_for_basis(nqbits, basis, f):
    """
    Returns ORS scores with shape (ns, len(K_ors_list)).

    The seed is reset so that the same reservoir instances are used
    across fidelities and measurement bases for a fixed n.
    """
    np.random.seed(reservoir_seed(nqbits))

    D = 2 ** nqbits

    _, sorted_probs, _ = lorentz_curves_noisy(
        make_reservoir_fn(nqbits),
        nqbits,
        nshots=0,
        ns=ns,
        fidelity=f,
        statevector=True,
        measurement_basis=basis,
        verbose=False,
    )

    top = sorted_probs[:, :K_max].copy()

    return np.column_stack(
        [
            order_statistics_score_noisy(
                top,
                D,
                K=K,
                f=f,
            )
            for K in K_ors_list
        ]
    )


def lambda_haar_reference(D, f):
    return np.array(
        [
            lambda_haar_largeD_noisy(D, K, f)
            for K in K_ors_list
        ]
    )


# ── Run ─────────────────────────────────────────────────────────────────────
to_save = dict(
    fs=np.array(fs),
    K_ors=np.array(K_ors_list),
)

for nqbits in tqdm(n_list, desc="n"):
    D = 2 ** nqbits

    print(
        f"→ n={nqbits}  {gates_name}  "
        f"degree={degree} ng={num_gates} L={n_layers}"
    )

    random_bases = [
        random_local_pauli_basis(nqbits, basis_rng)
        for _ in range(B_rand)
    ]

    bases = fixed_basis + random_bases

    # Basis labels
    to_save[f"basis_labels_n{nqbits}"] = np.array(
        [
            b if isinstance(b, str) else "".join(b)
            for b in bases
        ]
    )

    for f in tqdm(fs, desc=f"f (n={nqbits})", leave=False):
        ftag = ftag_from_float(f)

        print(f"   Fidelity f={f:.2f}")

        lambda_haar = lambda_haar_reference(D, f)

        # Shape after loop: (3 + B_rand, len(K_ors_list))
        gap_by_basis = []

        for bidx, basis in enumerate(bases):
            basis_label = basis if isinstance(basis, str) else "".join(basis)

            print(
                f"      Basis {bidx + 1}/{len(bases)}: {basis_label}"
            )

            scores = run_noisy_ors_for_basis(
                nqbits=nqbits,
                basis=basis,
                f=f,
            )

            # Signed gap, averaged over reservoir instances.
            gap = scores.mean(axis=0) - lambda_haar
            gap_by_basis.append(gap)

        gap_by_basis = np.array(gap_by_basis)

        for kidx, K in enumerate(K_ors_list):
            key = f"gap_n{nqbits}_K{K}_f{ftag}"
            to_save[key] = gap_by_basis[:, kidx]

            print(
                f"      Saved {key}: shape={to_save[key].shape}, "
                f"mean={to_save[key].mean():.4g}"
            )

        path = out_path()
        np.savez_compressed(path, **to_save)
        print(f"      Intermediate save → {path}")

# ── Final save ──────────────────────────────────────────────────────────────
path = out_path()
np.savez_compressed(path, **to_save)
print(f"Saved → {path}")