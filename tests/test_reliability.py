import numpy as np
import pytest
from defluorinate.numerics import explicit_hamiltonian, reduce_integrals, fci_solve
from defluorinate.projection import embed_ci, projection_mask
from defluorinate.reliability import linked_action, external_diagnostics


def problem(n=3):
    rng = np.random.default_rng(271)
    h = rng.normal(size=(n, n))
    h = (h + h.T) / 2
    a = rng.normal(size=(n, n, n, n))
    g = (
        a + a.transpose(1, 0, 2, 3) + a.transpose(0, 1, 3, 2) + a.transpose(1, 0, 3, 2)
    ) / 4
    g = (g + g.transpose(2, 3, 0, 1)) / 2
    return h, g


def test_sparse_links_against_independent_fermionic_dense():
    h, g = problem()
    H = explicit_hamiltonian(h, g, (1, 1), 1.7)
    c = np.random.default_rng(2).normal(size=(3, 3))
    c /= np.linalg.norm(c)
    assert np.allclose(
        linked_action(h, g, c, (1, 1), 1.7).ravel(), H @ c.ravel(), atol=1e-10
    )
    assert np.allclose(H, H.T)


def test_external_amplitudes_and_pt2_dense():
    h, g = problem()
    H = explicit_hamiltonian(h, g, (1, 1), 1.7)
    mask = np.zeros(9, bool)
    mask[[0, 1, 3, 4]] = True
    val, vec = np.linalg.eigh(H[np.ix_(mask, mask)])
    c = np.zeros(9)
    c[mask] = vec[:, 0]
    d, q, diag = external_diagnostics(h, g, 1.7, c, (1, 1), mask)
    exactq = (H @ c)[~mask]
    assert np.allclose(q.ravel()[~mask], exactq)
    assert np.isclose(d["rQ_Eh"], np.linalg.norm(exactq))
    assert np.isclose(d["pt2_Eh"], np.sum(exactq**2 / (val[0] - np.diag(H)[~mask])))
    assert d["internal_residual_Eh"] < 1e-10


def test_exact_subspace_and_wrong_branch():
    h = np.diag([0.0, 1.0])
    g = np.zeros((2,) * 4)
    c = np.zeros((2, 2))
    c[1, 1] = 1
    mask = c.astype(bool)
    d, _, _ = external_diagnostics(h, g, 0.0, c, (1, 1), mask)
    assert (
        d["rQ_Eh"] == 0 and d["pt2_Eh"] == 0 and d["energy"] == 2
    )  # wrong excited branch, no certificate!
    c[1, 1] = 0
    c[0, 0] = 1
    d, _, _ = external_diagnostics(h, g, 0.0, c, (1, 1), c.astype(bool))
    assert d["energy"] == 0


def test_weak_and_intruder_regularization():
    for gap in [1.0, 1e-12]:
        h = np.array([[0.0, 1e-3], [1e-3, gap]])
        g = np.zeros((2,) * 4)
        c = np.zeros((2, 2))
        c[0, 0] = 1
        d, _, _ = external_diagnostics(h, g, 0.0, c, (1, 1), c.astype(bool))
        if gap == 1.0:
            assert abs(d["pt2_Eh"] + 2e-6) < 1e-12
        else:
            assert (
                d["pt2_Eh"] is None
                and d["intruder_count"] > 0
                and np.isfinite(d["regularized_Eh"]["0.01"])
            )


def test_normalization_rejected():
    h, g = problem()
    c = np.ones((3, 3))
    with pytest.raises(ValueError):
        external_diagnostics(h, g, 0.0, c, (1, 1), np.ones((3, 3), bool))


def test_spin_and_offset():
    h, g = problem(2)
    e, c, dm, ss = fci_solve(h, g, (1, 1), 0.0)
    mask = np.ones_like(c, bool)
    d, _, _ = external_diagnostics(h, g, 13.0, c, (1, 1), mask)
    assert (
        abs(d["energy"] - e - 13) < 1e-9
        and abs(d["s_squared"] - ss) < 1e-9
        and d["rQ_Eh"] == 0
    )


def test_offsets_do_not_change_external_residual_or_correction():
    h, g = problem()
    c = np.zeros((3, 3))
    c[0, 0] = 1
    mask = c.astype(bool)
    d, q, _ = external_diagnostics(h, g, 0.0, c, (1, 1), mask)
    shifted, qq, _ = external_diagnostics(h, g, -523.0, c, (1, 1), mask)
    assert np.allclose(q, qq) and np.isclose(d["pt2_Eh"], shifted["pt2_Eh"], atol=1e-12)


def test_hardware_sqd_state_energy_and_mask_independently():
    from defluorinate.sqd import sqd_state

    h, g = problem(4)
    o = -2.37
    st = np.array([3, 5, 6, 9])
    e, ci, mask, ss, _ = sqd_state(h, g, o, st)
    H = explicit_hamiltonian(h, g, (2, 2), o)
    assert abs(np.vdot(ci.ravel(), H @ ci.ravel()) - e) < 1e-8
    assert np.linalg.norm(ci[~mask]) == 0 and abs(np.linalg.norm(ci) - 1) < 1e-10
    d, _, _ = external_diagnostics(h, g, o, ci, (2, 2), mask)
    assert abs(d["s_squared"] - ss) < 1e-8


def test_strong_coupling_pt2_can_worsen_energy():
    h = np.array([[0.0, 0.5], [0.5, 0.1]])
    g = np.zeros((2,) * 4)
    c = np.zeros((2, 2))
    c[0, 0] = 1
    d, _, _ = external_diagnostics(h, g, 0.0, c, (1, 1), c.astype(bool))
    ground = float(np.linalg.eigvalsh(explicit_hamiltonian(h, g, (1, 1)))[0])
    assert abs(d["energy"] + d["pt2_Eh"] - ground) > abs(d["energy"] - ground)
