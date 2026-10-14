import numpy as np
from defluorinate.numerics import *
from defluorinate.sqd import sci_energy, recover, classical_product
from test_numerics import h2


def test_recovery_bit_order_and_seed():
    samples = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15])
    a, p = recover(samples, 2, (1, 1), 11)
    b, q = recover(samples, 2, (1, 1), 11)
    assert np.array_equal(a, b)
    assert all(
        (int(x) & 3).bit_count() == 1 and (int(x) >> 2).bit_count() == 1 for x in a
    )
    assert abs(sum(p) - 1) < 1e-10


def test_nested_reduction_is_projection():
    rng = np.random.default_rng(19)
    n = 3
    h = rng.normal(size=(n, n))
    h = (h + h.T) / 2
    L = rng.normal(size=(2, n, n))
    L = (L + L.transpose(0, 2, 1)) / 2
    g = np.einsum("apq,ars->pqrs", L, L)
    H = explicit_hamiltonian(h, g, (2, 2), 2.7)
    hr, gr, o, ne = reduce_integrals(h, g, 2.7, [1, 2], 2)
    Hr = explicit_hamiltonian(hr, gr, ne, o)
    ids = sector(3, (2, 2))
    allowed = [i for i, b in enumerate(ids) if b & 1 and b & 8]
    assert np.allclose(
        np.linalg.eigvalsh(H[np.ix_(allowed, allowed)]),
        np.linalg.eigvalsh(Hr),
        atol=1e-10,
    )


def test_rectangular_virtual_transport_and_complement():
    from scipy.linalg import null_space
    from defluorinate.chemistry import orth_transport

    rng = np.random.default_rng(22)
    q, _ = np.linalg.qr(rng.normal(size=(12, 12)))
    old = q[:, :3]
    new = q[:, :8]
    moved, s = orth_transport(np.eye(12), old, new)
    rest = new @ null_space((new.T @ moved).T)
    full = np.column_stack([moved, rest])
    assert full.shape == (12, 8)
    assert np.allclose(full.T @ full, np.eye(8))
    assert np.allclose(moved, old)


def test_sqd_removes_spin_penalty_from_physical_energy():
    h, g, o, _ = h2()
    H = explicit_hamiltonian(h, g, (1, 1), o)
    energy, ss = sci_energy(h, g, o, np.array([1]), np.array([2]))
    assert abs(energy - H[1, 1]) < 1e-8
    assert abs(ss - 1) < 1e-8
