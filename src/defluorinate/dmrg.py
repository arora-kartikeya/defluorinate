"""Original quimb MPO and sector-extraction kernels."""

import numpy as np
from quimb.operator import SparseOperatorBuilder, HilbertSpace
from .numerics import terms, sector, strings


def mpo(h, g, ne, penalty=10.0):
    n = len(h)
    builder = SparseOperatorBuilder(
        hilbert_space=HilbertSpace(range(2 * n)), jordan_wigner=True, atol=1e-13
    )
    for c, ops in terms(h, g):
        if abs(c) > 1e-13:
            builder.add_term(
                float(c), *[(("+" if create else "-"), i) for i, create in ops]
            )
    for spin in range(2):
        builder.add_term(penalty * ne[spin] ** 2)
        for i in range(n):
            builder.add_term(penalty * (1 - 2 * ne[spin]), ("n", i + spin * n))
            for j in range(i):
                builder.add_term(2 * penalty, ("n", i + spin * n), ("n", j + spin * n))
    return builder.build_mpo()


def reversed_bits(values, m):
    return sum(((values >> i) & 1) << (m - 1 - i) for i in range(m))


def ordered_indices(n, ne):
    bits = sector(n, ne)
    return reversed_bits(bits, 2 * n), np.ones(len(bits), dtype=int)


def ordered_mpo(h, g, ne):
    H = mpo(h, g, ne)
    before = H.max_bond()
    H.compress(cutoff=1e-12, cutoff_mode="rel")
    print("MPO compression", before, "to", H.max_bond(), flush=True)
    return H


def sector_coefficients(state, n, ne):
    s = state.copy()
    s.permute_arrays("lrp")
    arr = list(s.arrays)
    arr[0] = arr[0][None, :, :]
    arr[-1] = arr[-1][:, None, :]
    a, b = strings(n, ne[0]), strings(n, ne[1])
    L = np.ones((len(a), 1))
    R = np.ones((len(b), 1))
    for j in range(n):
        A = arr[j][:, :, (a >> j) & 1].transpose(2, 0, 1)
        L = np.einsum("sl,slr->sr", L, A, optimize=True)
    for j in range(n - 1, -1, -1):
        A = arr[n + j][:, :, (b >> j) & 1].transpose(2, 0, 1)
        R = np.einsum("slr,sr->sl", A, R, optimize=True)
    return L @ R.T
