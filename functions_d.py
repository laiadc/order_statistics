from functools import partial
import numpy as np
from scipy import integrate


# ---------------------------------------------------------------------------
# Small NumPy replacements for the few QuTiP helpers used in the original file
# ---------------------------------------------------------------------------

def qeye(n: int):
    return np.eye(n, dtype=complex)


def sigmax():
    return np.array([[0, 1], [1, 0]], dtype=complex)


def sigmay():
    return np.array([[0, -1j], [1j, 0]], dtype=complex)


def sigmaz():
    return np.array([[1, 0], [0, -1]], dtype=complex)


def basis(dim: int, index: int):
    v = np.zeros(dim, dtype=complex)
    v[index] = 1.0
    return v


def tensor(*ops):
    """
    Kronecker product.

    Accepts either ``tensor(A, B, C)`` or ``tensor([A, B, C])`` for backward
    compatibility with the original QuTiP-based code.
    """
    if len(ops) == 1 and isinstance(ops[0], (list, tuple)):
        ops = tuple(ops[0])

    out = np.asarray(ops[0], dtype=complex)
    for op in ops[1:]:
        out = np.kron(out, np.asarray(op, dtype=complex))

    return out


def _normalize_state(psi):
    psi = np.asarray(psi, dtype=complex).reshape(-1)
    nrm = np.linalg.norm(psi)

    if nrm <= 0:
        raise ValueError("Cannot normalize a zero vector.")

    return psi / nrm


def _eigh_or_eig(A):
    A = np.asarray(A, dtype=complex)

    if np.allclose(A, A.conj().T):
        vals, vecs = np.linalg.eigh(A)
    else:
        vals, vecs = np.linalg.eig(A)

    return vals, vecs


def _project_to_symmetry_subspace(Op, symmetry, eigenval, aprox_symmetry=False):
    Op = np.asarray(Op, dtype=complex)
    symmetry = np.asarray(symmetry, dtype=complex)

    if aprox_symmetry:
        _, op_vecs = _eigh_or_eig(Op)
        s_arr = np.real(np.diag(op_vecs.conj().T @ symmetry @ op_vecs))
        subspace = np.where(np.round(s_arr, 1) == eigenval)[0]
        Op_trans = op_vecs.conj().T @ Op @ op_vecs
    else:
        s_evals, s_vecs = _eigh_or_eig(symmetry)
        subspace = np.where(np.isclose(s_evals, eigenval))[0]
        Op_trans = s_vecs.conj().T @ Op @ s_vecs

    if len(subspace) == 0:
        raise ValueError(f"No symmetry subspace found for eigenvalue {eigenval}.")

    return Op_trans[np.ix_(subspace, subspace)]


def sz_total(N: int, to_numpy=False):
    return sj_total(N, j="z", to_numpy=to_numpy)


def sx_total(N: int, to_numpy=False):
    return sj_total(N, j="x", to_numpy=to_numpy)


def sy_total(N: int, to_numpy=False):
    return sj_total(N, j="y", to_numpy=to_numpy)


def sj_total(N: int, j: str, to_numpy=False):
    """
    Total S_j operator from the sum of N spin-1/2 systems.

    The argument ``to_numpy`` is kept for backward compatibility. Since this
    version has no QuTiP dependency, the function always returns a NumPy array.
    """
    si = qeye(2)

    if j == "x":
        sj = 0.5 * sigmax()
    elif j == "y":
        sj = 0.5 * sigmay()
    elif j == "z":
        sj = 0.5 * sigmaz()
    else:
        raise ValueError("j must be one of 'x', 'y', or 'z'.")

    dim = 2**N
    sj_tot = np.zeros((dim, dim), dtype=complex)

    for n in range(N):
        op_list = [si for _ in range(N)]
        op_list[n] = sj
        sj_tot += tensor(op_list)

    return sj_tot


def sxz_alternate(N: int, to_numpy=False):
    si = qeye(2)
    sx = 0.5 * sigmax()
    sz = 0.5 * sigmaz()

    dim = 2**N
    out = np.zeros((dim, dim), dtype=complex)

    for n in range(N):
        op_list = [si for _ in range(N)]
        op_list[n] = sz if n % 2 == 0 else sx
        out += tensor(op_list)

    return out


def sxy_alternate(N: int, to_numpy=False):
    si = qeye(2)
    sx = 0.5 * sigmax()
    sy = 0.5 * sigmay()

    dim = 2**N
    out = np.zeros((dim, dim), dtype=complex)

    for n in range(N):
        op_list = [si for _ in range(N)]
        op_list[n] = sy if n % 2 == 0 else sx
        out += tensor(op_list)

    return out


def szy_alternate(N: int, to_numpy=False):
    si = qeye(2)
    sz = 0.5 * sigmaz()
    sy = 0.5 * sigmay()

    dim = 2**N
    out = np.zeros((dim, dim), dtype=complex)

    for n in range(N):
        op_list = [si for _ in range(N)]
        op_list[n] = sy if n % 2 == 0 else sz
        out += tensor(op_list)

    return out


def exchange_operator(dim: int, i: int, j: int):
    si = qeye(2)
    geye = [si for _ in range(dim)]

    H = np.zeros((2**dim, 2**dim), dtype=complex)

    for s in [sigmax(), sigmay(), sigmaz()]:
        g = geye.copy()
        g[i] = s
        g[j] = s
        H += tensor(g)

    H += tensor(geye)
    return H / 2


def parity_operator(n: int):
    si = qeye(2)
    geye = [si for _ in range(n)]

    p = tensor(geye)

    for m in range(n // 2):
        p = exchange_operator(n, m, n - m - 1) @ p

    return p


def parity_bra(psi, n=None):
    """Return P|psi>."""
    psi = np.asarray(psi, dtype=complex).reshape(-1)

    if n is None:
        n = int(np.log2(psi.shape[0]))

    p = parity_operator(n)
    return p @ psi


def to_even_state(psi):
    psi = np.asarray(psi, dtype=complex).reshape(-1)
    n = int(np.log2(psi.shape[0]))
    p = parity_operator(n)
    projected = (p + np.eye(psi.shape[0], dtype=complex)) @ psi

    return _normalize_state(projected)


class QOperator:
    """
    Minimal NumPy replacement for the original QuTiP-backed QOperator wrapper.
    """

    def __init__(self, op):
        self.op = np.asarray(op, dtype=complex)
        self.n = int(np.log2(self.op.shape[0]))
        self.transformations = []

    def to_numpy(self):
        return self.op

    def to_Qobj(self):
        """
        Kept for backward compatibility. Returns the NumPy matrix.
        """
        return self.op

    def to_sz_subspace(self, eigenval=0, aprox_symmetry=False):
        transformation = partial(
            self._to_subspace,
            symmetry=sz_total(self.n),
            eigenval=eigenval,
            aprox_symmetry=aprox_symmetry,
        )

        self.op = transformation(Op=self.op)
        self.transformations.append(transformation)
        return self

    def to_even_subspace(self, eigenval=1, aprox_symmetry=False):
        transformation = partial(
            self._to_subspace,
            symmetry=parity_operator(self.n),
            eigenval=eigenval,
            aprox_symmetry=aprox_symmetry,
        )

        self.op = transformation(Op=self.op)
        self.transformations.append(transformation)
        return self

    def _to_subspace(
        self,
        Op,
        symmetry,
        eigenval,
        aprox_symmetry,
        apply_prev_hamiltonian_transformations=True,
    ):
        if apply_prev_hamiltonian_transformations:
            for transformation in self.transformations:
                symmetry = transformation(
                    Op=symmetry,
                    apply_prev_hamiltonian_transformations=False,
                )

        return _project_to_symmetry_subspace(
            Op,
            symmetry,
            eigenval=eigenval,
            aprox_symmetry=aprox_symmetry,
        )


def to_even_subspace(H):
    H = np.asarray(H, dtype=complex)
    n = int(np.log2(H.shape[0]))
    P = parity_operator(n)

    return _project_to_symmetry_subspace(
        H,
        P,
        eigenval=1,
        aprox_symmetry=False,
    )


def to_sz_subspace(H, m=1):
    """
    Project H to the total-Sz subspace with eigenvalue m.
    """
    H = np.asarray(H, dtype=complex)
    n = int(np.log2(H.shape[0]))
    sz = sz_total(n)

    return _project_to_symmetry_subspace(
        H,
        sz,
        eigenval=m,
        aprox_symmetry=False,
    )


def is_even_state(psi):
    psi = np.asarray(psi, dtype=complex).reshape(-1)
    return np.allclose(psi, parity_bra(psi))


def odeintz(func, z0, t, **kwargs):
    """An odeint-like function for complex-valued differential equations."""
    _unsupported_odeint_args = ["Dfun", "col_deriv", "ml", "mu"]
    bad_args = [arg for arg in kwargs if arg in _unsupported_odeint_args]

    if bad_args:
        raise ValueError(f"The odeint argument {bad_args[0]!r} is not supported by odeintz.")

    z0 = np.array(z0, dtype=np.complex128, ndmin=1)

    def realfunc(x, t, *args):
        z = x.view(np.complex128)
        dzdt = func(z, t, *args)
        return np.asarray(dzdt, dtype=np.complex128).view(np.float64)

    result = integrate.odeint(realfunc, z0.view(np.float64), t, **kwargs)

    if kwargs.get("full_output", False):
        z = result[0].view(np.complex128)
        infodict = result[1]
        return z, infodict

    return result.view(np.complex128)


def wavefunction_phi(a, b, tmax, ntimes):
    """
    Output
    ------
    t.shape = (ntimes,)
    phi.shape = (ntimes, b.size)
    """
    a = np.asarray(a, dtype=complex)
    b = np.asarray(b, dtype=complex)

    def eq_syst(phi, t):
        dot_phi = np.zeros_like(b, dtype=complex)

        if b.size == 1:
            dot_phi[0] = 1j * a[0] * phi[0]
            return dot_phi

        dot_phi[0] = -b[1] * phi[1] + 1j * a[0] * phi[0]
        dot_phi[1:-1] = (
            b[1:-1] * phi[:-2]
            - b[2:] * phi[2:]
            + 1j * a[1:-1] * phi[1:-1]
        )
        dot_phi[-1] = b[-1] * phi[-2] + 1j * a[-1] * phi[-1]

        return dot_phi

    k = b.size
    phi0 = np.zeros(k, dtype=complex)
    phi0[0] = 1.0 + 0.0j

    t = np.linspace(0, tmax, ntimes)
    phi = odeintz(eq_syst, phi0, t)

    return t, phi


def k_complexity_from_phi(phi):
    phi = np.asarray(phi, dtype=complex)
    return np.sum(np.arange(phi.shape[1]) * np.abs(phi) ** 2, axis=1)


def k_complexity(a, b, tmax, ntimes=500):
    t, phi = wavefunction_phi(a, b, tmax, ntimes)
    kc = k_complexity_from_phi(phi)
    return t, kc


def computational_to_physic_base(psi):
    """
    Return single-qubit computational-basis probabilities.

    Parameters
    ----------
    psi
        State vector in the computational basis.

    Returns
    -------
    probabilities
        Array with shape ``(n, 2)``. Row q contains ``[P_q(0), P_q(1)]``.
    """
    psi = _normalize_state(psi)
    n = int(np.log2(psi.shape[0]))

    if 2**n != psi.shape[0]:
        raise ValueError("State length must be a power of two.")

    probs = np.abs(psi.reshape((2,) * n)) ** 2
    probabilities = []

    for qb in range(n):
        axes = tuple(ax for ax in range(n) if ax != qb)
        marginal = np.sum(probs, axis=axes)
        probabilities.append(marginal)

    return np.asarray(probabilities, dtype=float)


def computational_base_to_fstr(psi, unicode=True, simplify=False):
    probabilities = computational_to_physic_base(psi)

    if unicode:
        down, up, mixed = "↓", "↑", "⇅"
    else:
        down, up, mixed = "d", "u", "m"

    s_list = []

    for probs in probabilities:
        b0p, b1p = probs

        if np.isclose(b0p, 1):
            s = down
        elif np.isclose(b1p, 1):
            s = up
        else:
            if simplify:
                s = mixed
            else:
                s = f"({b0p:.2g}{down} + {b1p:.2g}{up})"

        s_list.append(s)

    return " ".join(s_list)


def chaometer(H):
    e = np.linalg.eigvalsh(np.asarray(H, dtype=complex))
    return chaometer_from_energies(e)


def chaometer_from_energies(e_arr):
    IWD = 0.5307
    IP = 0.3863

    e = np.sort(np.real(e_arr))
    s = np.diff(e)
    s = s[np.abs(s) > 1e-12]

    if len(s) < 2:
        return np.nan

    r = s[:-1] / s[1:]
    r_tilde = np.min(np.column_stack([r, 1 / r]), axis=1)
    eta = (np.mean(r_tilde) - IP) / (IWD - IP)

    return np.real(eta)


def __test_parity_operators():
    psi = tensor(
        basis(2, 0),
        basis(2, 1),
        basis(2, 1),
    )

    print(f"\n{psi = }")
    print(f"\n{parity_bra(psi) = }")
    print(f"\n{to_even_state(psi) = }")


def __test_computational_base_to_fstr():
    down = basis(2, 0)
    up = basis(2, 1)

    psi_single = _normalize_state(np.sqrt(1 / 4) * down + np.sqrt(3 / 4) * up)
    psi = tensor(psi_single, up, up)

    print(f"\n{computational_base_to_fstr(psi) = }")
    print(f"\n{computational_base_to_fstr(psi, simplify=True) = }")
