"""Diagnostics for native reservoirs: ORS (gate-based), ORS_LO (photonic analogue), R_eff.

Gate-based ORS
    The original, unchanged ``metrics.py`` implementation (Exp1–Exp3): the bare reservoir acts
    on |0…0⟩, output probabilities are measured in B=5 bases (Z, X, Y + 2 random local
    bases from ``default_rng(4321)``, exactly as the multibasis ORS files used in Exp3), and

        Λ_s = Σ_{k=1}^{K} (1/k) · log P_k^Haar(p_(k))        (Porter–Thomas order statistics)
        gap = mean_s Λ_s − Λ_Haar(D, K, f)

Why ORS does not transfer directly to photonic output distributions
    Λ_Haar is the order-statistics density of a Haar-random *pure state in D dimensions*
    measured in an orthonormal basis (Dirichlet(1,…,1) ≈ i.i.d. Exp(D)). A linear-optical
    network only reaches an m²-parameter subgroup of U(D), D = C(m+n-1, n); its output
    probabilities are |Perm(U_S,T)|²/(Π s_i! t_j!) with bunching, so even the *maximally
    expressive photonic ensemble* (Haar-random interferometers) is far from Porter–Thomas and
    the naive gap is not zero (see :func:`naive_gate_ors_gap`).

Photonic analogue ORS_LO (a separate metric; the gate-based one is untouched)
    Same functional, different reference ensemble:

        Λ_s^LO = Σ_k (1/k) · log P_k^LO(p_(k)),    gap_LO = mean_s Λ_s^LO − Λ_ref^LO

    where P_k^LO is the Porter–Thomas order-statistics density of ``metrics.log_pk`` with the
    dimension D replaced by an *effective dimension* D_eff, fitted by maximum likelihood to
    the top-K probabilities of Haar-random U(m) interferometers with the *same* (m, n, input
    state, detection), and Λ_ref^LO is the empirical mean score of independent held-out
    Haar-interferometer draws (so the reference ensemble has zero gap by construction).
    The analytic family keeps the (linear-in-p) tails of the original metric; a non-parametric
    KDE reference was tried and rejected because its Gaussian tails in log p are an artefact
    of the estimator rather than of the ensemble.

    Consistency with the original metric (checked in the tests):
      * for n = 1 photon the reference ensemble *is* the qubit Haar ensemble in dimension m
        (single-photon outputs of Haar interferometers are exactly Dirichlet(1,…,1)), so
        ORS_LO coincides with the same estimator applied to Haar states of dimension m;
      * on Haar qubit states the fitted D_eff recovers D at large D (≈ 66 for D = 64), and
        the empirical Λ_ref exposes the finite-D bias of the analytic Λ_Haar (+0.38 at n = 6),
        which is why true-Haar qubit ensembles show a small positive gate-ORS gap.

    ORS scores scale with the (effective) dimension through the -k·(D-2)·p term, so ORS_LO
    at D_eff ~ 10³–10⁴ spans a much larger range than gate ORS at D = 64. ORS_LO values are
    therefore NOT numerically comparable with gate-based ORS gaps.

R_eff
    Participation ratio of the singular values of the centred feature matrix — verbatim copy
    of ``experiments/QELM/Fourier/qelm_utils.effective_rank`` — computed on the feature matrix
    the readout actually uses, so it is directly comparable across architectures.
"""

from __future__ import annotations

import numpy as np
from scipy.special import betaln
from scipy.stats import unitary_group

from metrics import (lambda_haar_largeD, lambda_haar_largeD_noisy, lorentz_curves_noisy,
                     order_statistics_score, order_statistics_score_noisy)


# ── R_eff (verbatim from experiments/QELM/Fourier/qelm_utils.py) ──────────────
def effective_rank(F, eps=1e-12):
    """Participation ratio of singular values of the centred feature matrix."""
    F = F - F.mean(axis=0, keepdims=True)
    try:
        sv = np.linalg.svd(F, compute_uv=False)
    except np.linalg.LinAlgError:
        from scipy.linalg import svd as sp_svd
        sv = sp_svd(F, compute_uv=False, lapack_driver='gesvd')
    sv = sv[sv > eps]
    if len(sv) == 0:
        return 0.0
    return float(sv.sum()**2 / (sv**2).sum())


# ── Gate-based ORS (unchanged metrics.py functions) ───────────────────────────
def exp3_bases(n: int, B: int = 5, basis_seed: int = 4321) -> list[str]:
    """Z, X, Y + (B-3) random local Pauli bases, drawn as in run_ORS_noisy_multibasis."""
    rng = np.random.default_rng(basis_seed)
    rand = ["".join(rng.choice(["X", "Y", "Z"], size=n).tolist()) for _ in range(max(0, B - 3))]
    return (["Z", "X", "Y"] + rand)[:B]


def gate_ors(factories, n_qubits: int, K: int = 4, B: int = 5, fidelity: float = 1.0,
             basis_seed: int = 4321) -> dict:
    """ORS of an ensemble of gate-based reservoirs with the original metrics.py code.

    Parameters
    ----------
    factories : list of callables
        One zero-argument callable per reservoir instance returning its bare circuit.
    Returns dict with per-basis gaps, per-instance gaps (averaged over bases), the cell gap
    (mean over bases) and its standard error std/√B.
    """
    D = 2 ** n_qubits
    bases = exp3_bases(n_qubits, B, basis_seed)
    lam_ref = lambda_haar_largeD_noisy(D, K, fidelity)
    scores = np.zeros((len(factories), len(bases)))
    for b, basis in enumerate(bases):
        it = iter(factories)
        _, sp, _ = lorentz_curves_noisy(lambda: next(it)(), n_qubits, nshots=0, ns=len(factories),
                                        fidelity=fidelity, statevector=True,
                                        measurement_basis=basis, verbose=False)
        scores[:, b] = order_statistics_score_noisy(sp, D, K, fidelity)
    gaps = scores - lam_ref
    gap_b = gaps.mean(axis=0)
    return {"bases": bases, "K": K, "fidelity": fidelity, "lambda_ref": lam_ref,
            "scores": scores, "per_instance_gap": gaps.mean(axis=1),
            "gap_per_basis": gap_b, "gap": float(gap_b.mean()),
            "gap_sem": float(gap_b.std() / np.sqrt(len(bases)))}


def gate_ors_from_probs(prob_vectors, D: int, K: int = 4, fidelity: float = 1.0) -> dict:
    """ORS gap from (ns, D) probability vectors (e.g. vendor-emulator counts / shots)."""
    sp = -np.sort(-np.asarray(prob_vectors, dtype=float), axis=1)
    s = order_statistics_score_noisy(sp, D, K, fidelity)
    lam = lambda_haar_largeD_noisy(D, K, fidelity)
    g = s - lam
    ok = np.isfinite(g)
    return {"per_instance_gap": g, "gap": float(np.mean(g[ok])) if ok.any() else np.nan,
            "gap_sem": float(np.std(g[ok]) / np.sqrt(ok.sum())) if ok.any() else np.nan,
            "lambda_ref": lam}


def naive_gate_ors_gap(prob_vectors, K: int = 4) -> dict:
    """Plug photonic distributions into the *qubit* ORS with D = number of outcomes.

    Used only to demonstrate the incompatibility (non-zero gap for Haar interferometers).
    """
    P = np.asarray(prob_vectors, dtype=float)
    D = P.shape[1]
    sp = -np.sort(-P, axis=1)
    s = order_statistics_score(sp, D, K)
    return {"per_instance_gap": s - lambda_haar_largeD(D, K),
            "gap": float(np.mean(s) - lambda_haar_largeD(D, K)), "D": D}


# ── Photonic ORS_LO: Porter–Thomas family with an effective dimension ────────
def _log_pk_vec(x, k, D, eps=1e-300):
    """Vectorised ``metrics.log_pk`` (identical formula) for real-valued D."""
    x = np.clip(np.asarray(x, dtype=float), eps, 1.0 - eps)
    A = D - 2.0
    return (np.log(A) - betaln(k, D - k + 1)) - k * A * x + (D - k) * np.log1p(-np.exp(-A * x))


class EffectiveDimensionReference:
    """ORS reference fitted to a *reference ensemble* of output distributions.

    The qubit ORS scores the top-K probabilities with the order-statistics density of D i.i.d.
    Porter–Thomas variables (``metrics.log_pk``). For linear optics the maximally expressive
    ensemble (Haar-random interferometers) is not Porter–Thomas in D = C(m+n-1, n)
    dimensions, so we keep the *same analytic family* and fit its single parameter — an
    effective dimension D_eff — by maximum likelihood to the top-K probabilities of
    ``n_ref`` reference draws. The reference value Λ_ref is the empirical mean score of
    ``n_holdout`` independent reference draws, so the gap of the reference ensemble is zero
    by construction. For a Porter–Thomas ensemble (qubit Haar states, or one photon in a Haar
    interferometer) D_eff recovers D and ORS_LO reduces to the original ORS.

    ``sample_fn(rng)`` returns one probability vector of the reference ensemble.
    """

    def __init__(self, sample_fn, K: int = 4, n_ref: int = 2000, n_holdout: int = 1000,
                 seed: int = 0, label: str = ""):
        from scipy.optimize import minimize_scalar
        rng = np.random.default_rng(seed)
        draws = [np.asarray(sample_fn(rng), dtype=float) for _ in range(n_ref + n_holdout)]
        self.D_nominal = len(draws[0])
        top = np.array([np.sort(p)[::-1][:K] for p in draws])
        self.K, self.n_ref, self.n_holdout, self.label, self.seed = K, n_ref, n_holdout, label, seed
        ref = top[:n_ref]

        def nll(logD):
            D = np.exp(logD)
            return -sum(_log_pk_vec(ref[:, k - 1], k, D).sum() for k in range(1, K + 1))

        lo, hi = np.log(K + 3.0), np.log(50.0 * self.D_nominal + 10.0)
        self.D_eff = float(np.exp(minimize_scalar(nll, bounds=(lo, hi), method="bounded",
                                                  options={"xatol": 1e-6}).x))
        self.holdout_scores = self.score(top[n_ref:])
        self.lambda_ref = float(self.holdout_scores.mean())
        self.lambda_ref_analytic = float(lambda_haar_largeD(self.D_eff, K))

    def score(self, sorted_top) -> np.ndarray:
        """Λ_s = Σ_k (1/k) log P_k(p_(k); D_eff) — ``metrics.order_statistics_score`` at D_eff."""
        S = np.atleast_2d(np.asarray(sorted_top, dtype=float))[:, : self.K]
        return order_statistics_score(S, self.D_eff, self.K)

    def gap(self, prob_vectors) -> dict:
        P = np.atleast_2d(np.asarray(prob_vectors, dtype=float))
        s = self.score(-np.sort(-P, axis=1))
        g = s - self.lambda_ref
        return {"per_instance_gap": g, "gap": float(g.mean()),
                "gap_sem": float(g.std() / np.sqrt(len(g))), "lambda_ref": self.lambda_ref,
                "D_eff": self.D_eff}


def photonic_lo_reference(n_modes: int, n_photons: int, K: int = 4, n_ref: int = 2000,
                          n_holdout: int = 1000, seed: int = 0) -> EffectiveDimensionReference:
    """ORS_LO reference: Haar-random U(m) interferometers on the same dual-rail input."""
    from .quandela_photonic import QuandelaPhotonicReservoir
    template = QuandelaPhotonicReservoir(n_modes, n_photons, mesh_depth=1, seed=0).build()

    def sample(rng):
        return template.probabilities(unitary_group.rvs(n_modes, random_state=rng))

    return EffectiveDimensionReference(sample, K, n_ref, n_holdout, seed,
                                       label=f"Haar U({n_modes}), {n_photons} photons")


def qubit_haar_reference(n_qubits: int, K: int = 4, n_ref: int = 2000, n_holdout: int = 1000,
                         seed: int = 0) -> EffectiveDimensionReference:
    """Same estimator with Haar-random qubit states (validation: D_eff ≈ 2^n)."""
    D = 2 ** n_qubits

    def sample(rng):
        z = rng.normal(size=D) + 1j * rng.normal(size=D)
        return np.abs(z) ** 2 / np.sum(np.abs(z) ** 2)

    return EffectiveDimensionReference(sample, K, n_ref, n_holdout, seed, label=f"Haar {n_qubits} qubits")
