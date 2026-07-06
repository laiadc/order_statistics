import numpy as np

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


def tensor(ops):
    """Kronecker product of a list of operators."""
    out = np.asarray(ops[0], dtype=complex)
    for op in ops[1:]:
        out = np.kron(out, np.asarray(op, dtype=complex))
    return out


# matriz evolutiva unitaria Harper
def FF(i, j, k, n):
    xi1 = 0.0
    xi2 = 0.0
    return (
        np.exp(
            -1j
            * 2.0
            * np.pi
            * (float(i - 1) + xi1)
            * (float(j - 1) + xi2)
            / float(n)
        )
        * np.sqrt(1.0 / float(n))
    )


def maUH(k, n):
    UU = np.zeros((n, n), dtype=complex)
    MM = np.zeros((n, n), dtype=complex)
    U = np.zeros((n, n), dtype=complex)
    xi1 = 0.5

    for j in range(n):
        for i in range(n):
            phase = np.exp(
                1j
                * float(n)
                * k
                * np.cos(-2.0 * np.pi * (float(i - 1) + xi1) / float(n))
            )
            UU[i, j] = FF(i, j, k, n) * phase
            MM[i, j] = np.conjugate(FF(j, i, k, n)) * phase

    for j in range(n):
        for i in range(n):
            for l in range(n):
                U[i, j] += MM[i, l] * UU[l, j]

    return U


# matriz del standard map
def maUS(k, n):
    UU = np.zeros((n, n), dtype=complex)
    MM = np.zeros((n, n), dtype=complex)
    xi1 = 0.5
    xi2 = 0.5

    for j in range(n):
        for i in range(n):
            UU[i, j] = FF(i, j, k, n) * np.exp(
                -1j
                * float(n)
                * (k / (2 * np.pi))
                * np.cos(-2.0 * np.pi * (float(i - 1) + xi1) / float(n))
            )
            MM[i, j] = np.conjugate(FF(j, i, k, n)) * np.exp(
                -1j * np.pi * (i + xi2) ** 2 / float(n)
            )

    return MM @ UU


def Kcomplex(F, v, tie):
    Uf = F
    tt = []
    Kc = []

    for ii in range(tie - 1):
        kk = 0
        for i in range(len(v)):
            v1 = Uf @ v[0]
            v2 = np.conjugate(v[i]) @ v1
            kk += float(i) * np.abs(v2) ** 2

        Kc.append(kk)
        tt.append(ii + 1)
        Uf = F @ Uf

    return tt, Kc


def r_chaometer(ener, plotadjusted):
    ener = np.asarray(ener, dtype=float)
    ener = np.sort(ener)

    spacings = np.diff(ener)
    spacings = spacings[np.abs(spacings) > 1e-12]

    if len(spacings) < 2:
        return np.nan

    ratios = spacings[1:] / spacings[:-1]
    ratios = np.minimum(ratios, 1.0 / ratios)
    ra = np.mean(ratios)

    if plotadjusted:
        ra = (ra - 0.5307) / (0.3863 - 0.5307)

    return ra


def h_ising_transverse(
    N: int,
    hx: float,
    hz: float,
    jx: float,
    jy: float,
    jz: float,
    to_numpy: bool = False,
):
    """
    Transverse-field Ising-type Hamiltonian built with NumPy arrays.

    The argument ``to_numpy`` is kept for backward compatibility. Since this
    version has no QuTiP dependency, the function always returns a NumPy array.
    """
    si = qeye(2)
    sx = sigmax()
    sy = sigmay()
    sz = sigmaz()

    sx_list = []
    sy_list = []
    sz_list = []

    for n in range(N):
        op_list = [si for _ in range(N)]

        op_x = op_list.copy()
        op_x[n] = sx
        sx_list.append(tensor(op_x))

        op_y = op_list.copy()
        op_y[n] = sy
        sy_list.append(tensor(op_y))

        op_z = op_list.copy()
        op_z[n] = sz
        sz_list.append(tensor(op_z))

    dim = 2**N
    H = np.zeros((dim, dim), dtype=complex)

    for n in range(N):
        H += hz * sz_list[n]
        H += hx * sx_list[n]

    for n in range(N - 1):
        H += -jx * (sx_list[n] @ sx_list[n + 1])
        H += -jy * (sy_list[n] @ sy_list[n + 1])
        H += -jz * (sz_list[n] @ sz_list[n + 1])

    return H


def FO_state(
    H: np.ndarray,
    e0: np.ndarray,
    stop: int = None,
    start_from: tuple = None,
):
    """
    Full-orthonormalization Lanczos algorithm for states.

    Parameters
    ----------
    H
        Hermitian matrix as a NumPy array.
    e0
        Initial state vector.
    stop
        Maximum Krylov-basis size. Defaults to the Hilbert-space dimension.
    start_from
        Optional tuple ``(a_arr, b_arr, e_arr)`` for continuation.

    Returns
    -------
    a_arr, b_arr, e_arr
        Lanczos diagonal coefficients, off-diagonal coefficients, and Krylov
        basis vectors.
    """
    H = np.asarray(H, dtype=complex)
    e0 = np.asarray(e0, dtype=complex).reshape(-1)

    stop = stop or H.shape[0]

    def L(psi):
        return H @ psi

    def prod(x, y):
        return np.vdot(x, y)

    def norm(e):
        return np.sqrt(np.real(prod(e, e)))

    if start_from is not None:
        a_arr = list(start_from[0])
        b_arr = list(start_from[1])
        e_arr = [np.asarray(e, dtype=complex).reshape(-1) for e in start_from[2]]
        n = len(b_arr)
    else:
        e0_norm = norm(e0)
        if e0_norm <= 0:
            raise ValueError("Initial state e0 has zero norm.")

        e0 = e0 / e0_norm
        a_arr = [prod(e0, L(e0))]
        b_arr = [0.0]
        e_arr = [e0]
        n = 1

    bn = np.inf

    while bn > 1e-7:
        An = L(e_arr[n - 1])

        # Two Gram-Schmidt passes for numerical stability
        for _ in range(2):
            for en in e_arr:
                An = An - en * prod(en, An)

        bn = norm(An)

        if bn <= 1e-7:
            break

        Kn = An / bn
        an = prod(Kn, L(Kn))

        a_arr.append(an)
        b_arr.append(bn)
        e_arr.append(Kn)

        if n >= stop - 1:
            break

        n += 1

    return np.real(np.array(a_arr)), np.array(b_arr), np.array(e_arr)
