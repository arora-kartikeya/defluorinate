"""Frequency-weighted SQD, configuration recovery, and a matched classical control."""

import math
from collections import Counter
import numpy as np
from qiskit_addon_sqd.fermion import solve_fermion
from qiskit_addon_sqd.configuration_recovery import recover_configurations
from .numerics import strings


def sci_energy(h, g, offset, a, b):
    e, state, occ, ss = solve_fermion(
        (np.sort(a), np.sort(b)),
        h,
        g,
        open_shell=True,
        spin_sq=0,
        shift=0.5,
        max_cycle=100,
        tol=1e-10,
    )
    return float(e + offset), float(ss)


def classical_product(H, n, ne, k):
    """Residual/diagonal-denominator ranking, product closure, HF initialization."""
    st = strings(n, ne[0])
    m = len(st)
    chosen = [0]
    eps = 1e-8
    while len(chosen) < k:
        ids = np.array([i * m + j for i in chosen for j in chosen])
        eig, v = np.linalg.eigh(H[np.ix_(ids, ids)])
        residual = H[:, ids] @ v[:, 0]
        denom = np.maximum(abs(np.diag(H) - eig[0]), eps)
        weights = (abs(residual) ** 2 / denom).reshape(m, m)
        score = weights.sum(0) + weights.sum(1)
        score[chosen] = -1
        chosen.append(int(np.argmax(score)))
    return np.sort(st[chosen])


def recover(samples, n, ne, seed):
    mat = np.array([[c == "1" for c in format(int(x), f"0{2*n}b")] for x in samples])
    # Regularized HF occupancy prior, never held-out exact populations.
    occ = np.array([0.95 if i < ne[0] else 0.05 for i in range(n)])
    rec, prob = recover_configurations(
        mat, np.ones(len(mat)) / len(mat), (occ, occ), *ne, rand_seed=seed
    )
    return (
        np.array([int("".join("1" if b else "0" for b in row), 2) for row in rec]),
        prob,
    )


def weighted_product(samples, weights, n, ne, cap):
    weights = np.asarray(weights, float)
    if (
        len(samples) != len(weights)
        or (weights < 0).any()
        or not np.isfinite(weights).all()
        or weights.sum() <= 0
    ):
        raise ValueError("Invalid sample weights")
    weights = weights / weights.sum()
    mask = (1 << n) - 1
    counts = Counter()
    valid = 0.0
    dets = set()
    for x, w in zip(samples, weights):
        a = int(x) & mask
        b = int(x) >> n
        if a.bit_count() == ne[0] and b.bit_count() == ne[1]:
            valid += float(w)
            counts[a] += float(w)
            counts[b] += float(w)
            dets.add(int(x))
    if not counts:
        return None, {"valid_fraction": 0.0, "dimension": 0, "observed_determinants": 0}
    k = min(math.isqrt(cap), len(counts))
    st = np.array(
        sorted(sorted(counts, key=lambda x: (-counts[x], x))[:k]), dtype=np.int64
    )
    return st, {
        "valid_fraction": valid,
        "dimension": k * k,
        "observed_determinants": len(dets),
        "selected_spin_strings": k,
    }


def readout_mitigate(samples, n, p):
    """Known synthetic independent readout inversion; clipping is explicitly biased."""
    size = 1 << (2 * n)
    freq = np.bincount(samples, minlength=size) / len(samples)
    inv = np.linalg.inv(np.array([[1 - p, p], [p, 1 - p]]))
    tensor = freq.reshape([2] * (2 * n))
    for axis in range(2 * n):
        tensor = np.moveaxis(np.tensordot(inv, tensor, axes=(1, axis)), 0, axis)
    quasi = tensor.ravel()
    clipped = np.maximum(quasi, 0)
    clipped /= clipped.sum()
    return np.arange(size), clipped, float(-quasi[quasi < 0].sum())


def sqd_state(h, g, offset, selected, ne=(2, 2)):
    """Reconstruct the physical CI state for residual and spin diagnostics."""
    import time

    start = time.perf_counter()
    n = len(h)
    e, state, occ, ss = solve_fermion(
        (selected, selected),
        h,
        g,
        open_shell=True,
        spin_sq=0,
        shift=0.5,
        max_cycle=100,
        tol=1e-10,
    )
    alpha, beta = strings(n, ne[0]), strings(n, ne[1])
    ix = {int(b): i for i, b in enumerate(alpha)}
    iy = {int(b): i for i, b in enumerate(beta)}
    ia = [ix[int(b)] for b in state.ci_strs_a]
    ib = [iy[int(b)] for b in state.ci_strs_b]
    ci = np.zeros((len(alpha), len(beta)))
    ci[np.ix_(ia, ib)] = state.amplitudes
    ci /= np.linalg.norm(ci)
    mask = np.zeros(ci.shape, bool)
    mask[np.ix_(ia, ib)] = True
    return float(e + offset), ci, mask, float(ss), time.perf_counter() - start
