"""Explicit conventions: chemist ERIs, alpha modes then beta, little-endian qubits."""

import itertools, math
import numpy as np
from pyscf import fci, ao2mo


def strings(n, k):
    return np.array(fci.cistring.make_strings(range(n), k), dtype=np.int64)


def sector(n, ne):
    a, b = strings(n, ne[0]), strings(n, ne[1])
    return np.array([int(x) | (int(y) << n) for x in a for y in b], dtype=np.int64)


def apply(bits, ops):
    sign = 1
    for idx, creation in reversed(ops):
        occ = (bits >> idx) & 1
        if occ == creation:
            return None, 0
        if (bits & ((1 << idx) - 1)).bit_count() % 2:
            sign = -sign
        bits ^= 1 << idx
    return bits, sign


def terms(h, g):
    n = len(h)
    for p, q in itertools.product(range(n), repeat=2):
        for s in range(2):
            yield h[p, q], [(p + s * n, True), (q + s * n, False)]
    for p, q, r, s in itertools.product(range(n), repeat=4):
        for a, b in itertools.product(range(2), repeat=2):
            yield 0.5 * g[p, q, r, s], [
                (p + a * n, True),
                (r + b * n, True),
                (s + b * n, False),
                (q + a * n, False),
            ]


def explicit_hamiltonian(h, g, ne, offset=0):
    basis = sector(len(h), ne)
    lookup = {int(x): i for i, x in enumerate(basis)}
    if len(basis) > 500:
        raise ValueError("Explicit dense sector limited to 500 determinants")
    H = np.eye(len(basis)) * offset
    for coeff, ops in terms(h, g):
        if abs(coeff) < 1e-14:
            continue
        for col, bits in enumerate(basis):
            out, sign = apply(int(bits), ops)
            if sign and out in lookup:
                H[lookup[out], col] += coeff * sign
    assert np.max(abs(H - H.T)) < 1e-9
    return H


def jw_operator(h, g, offset=0):
    from qiskit.quantum_info import SparsePauliOp

    m = 2 * len(h)
    ident = SparsePauliOp("I" * m)
    out = offset * ident

    def ladder(idx, creation):
        labels = []
        for axis, c in [("X", 0.5), ("Y", -0.5j if creation else 0.5j)]:
            text = ["I"] * m
            for j in range(idx):
                text[m - 1 - j] = "Z"
            text[m - 1 - idx] = axis
            labels.append(("".join(text), c))
        return SparsePauliOp.from_list(labels)

    for coeff, ops in terms(h, g):
        if abs(coeff) < 1e-14:
            continue
        term = ident
        for idx, creation in ops:
            term = term.dot(ladder(idx, creation))
        out = out + coeff * term
    return out.simplify(atol=1e-12)


def fci_solve(h, g, ne, offset=0):
    n = len(h)
    dim = math.comb(n, ne[0]) * math.comb(n, ne[1])
    if dim > 100000:
        raise ValueError("FCI sector cap")
    solver = fci.direct_spin0.FCI() if ne[0] == ne[1] else fci.direct_spin1.FCI()
    solver.conv_tol = 1e-11
    solver.max_cycle = 100
    solver.max_space = 40
    solver.max_memory = 700
    e, c = solver.kernel(h, g, n, ne, ecore=offset)
    if not solver.converged:
        raise RuntimeError("FCI did not converge")
    ss = fci.spin_op.spin_square(c, n, ne)[0]
    dm = solver.make_rdm1(c, n, ne)
    return float(e), c, dm, float(ss)


def reduce_integrals(h, g, offset, active, nocc):
    n = len(h)
    active = list(active)
    frozen = [i for i in range(nocc) if i not in active]
    off = float(offset)
    for i in frozen:
        off += 2 * h[i, i]
        for j in frozen:
            off += 2 * g[i, i, j, j] - g[i, j, j, i]
    eff = h.copy()
    for p, q in itertools.product(range(n), repeat=2):
        for i in frozen:
            eff[p, q] += 2 * g[p, q, i, i] - g[p, i, i, q]
    ne = nocc - len(frozen)
    return (
        eff[np.ix_(active, active)],
        g[np.ix_(active, active, active, active)],
        off,
        (ne, ne),
    )


def local_rdm(ci, n, ne, sites):
    """Fermionic partial trace: move selected modes first, including permutation parity."""
    chosen = [m for i in sites for m in [i, n + i]]
    rest = [i for i in range(2 * n) if i not in chosen]
    order = chosen + rest
    dim = 1 << len(chosen)
    matrix = np.zeros((dim, 1 << len(rest)), complex)
    for bits, amp in zip(sector(n, ne), ci.ravel()):
        occ = [m for m in order if (int(bits) >> m) & 1]
        parity = (
            sum(
                occ[i] > occ[j] for i in range(len(occ)) for j in range(i + 1, len(occ))
            )
            % 2
        )
        left = sum(((int(bits) >> m) & 1) << k for k, m in enumerate(chosen))
        right = sum(((int(bits) >> m) & 1) << k for k, m in enumerate(rest))
        matrix[left, right] = (-1) ** parity * amp
    rho = matrix @ matrix.conj().T
    assert abs(np.trace(rho) - 1) < 1e-8
    assert np.linalg.eigvalsh(rho).min() > -1e-10
    return rho


def entropy(rho):
    p = np.linalg.eigvalsh(rho)
    p = p[p > 1e-14]
    return float(-np.dot(p, np.log(p)))


def information(ci, n, ne):
    s = np.array([entropy(local_rdm(ci, n, ne, [i])) for i in range(n)])
    mi = np.zeros((n, n))
    for i in range(n):
        for j in range(i):
            mi[i, j] = mi[j, i] = 0.5 * (
                s[i] + s[j] - entropy(local_rdm(ci, n, ne, [i, j]))
            )
    assert mi.min() > -1e-9
    return s, mi


def projector_weight(ci, n, ne, active, nocc):
    keep = []
    for bits in sector(n, ne):
        keep.append(
            all(
                ((int(bits) >> i) & 1) == int(i < nocc)
                and ((int(bits) >> (n + i)) & 1) == int(i < nocc)
                for i in range(n)
                if i not in active
            )
        )
    return float(np.sum(abs(ci.ravel()[keep]) ** 2))


def embed_dm(dm, active, n, nocc):
    out = np.diag([2.0 if i < nocc and i not in active else 0.0 for i in range(n)])
    out[np.ix_(active, active)] = dm
    return out
