import numpy as np
from defluorinate.numerics import explicit_hamiltonian, reduce_integrals
from defluorinate.projection import embed_ci, action, projection_mask, rdm_energy
from defluorinate.sqd import weighted_product
from defluorinate.selection import choose, score_arrays, options


def test_frozen_embedding_matches_independent_ladder_hamiltonian():
    rng = np.random.default_rng(712)
    n = 4
    h = rng.normal(size=(n, n))
    h = (h + h.T) / 2
    a = rng.normal(size=(3, n, n))
    a = (a + a.transpose(0, 2, 1)) / 2
    g = np.einsum("kpq,krs->pqrs", a, a) * 0.1
    off = 3.14
    H = explicit_hamiltonian(h, g, (2, 2), off)
    for ids in [[1, 3], [0, 2], [1, 2, 3]]:
        hr, gr, o, ne = reduce_integrals(h, g, off, ids, 2)
        small = explicit_hamiltonian(hr, gr, ne, o)
        dim = int(np.sqrt(len(small)))
        U = np.column_stack(
            [
                embed_ci(np.eye(len(small))[i].reshape(dim, dim), ids, 4, 2).ravel()
                for i in range(len(small))
            ]
        )
        assert np.max(abs(U.T @ H @ U - small)) < 1e-10
        c = rng.normal(size=(6, 6))
        c /= np.linalg.norm(c)
        assert (
            np.linalg.norm(action(h, g, off, c, (2, 2)).ravel() - H @ c.ravel()) < 1e-10
        )
        e, d1, d2 = rdm_energy(h, g, off, c, (2, 2))
        assert abs(e - c.ravel() @ H @ c.ravel()) < 1e-10


def test_weighted_recovery_preserves_shot_frequency_not_unique_uniform():
    # Three valid determinants with sharply unequal mass; unique-uniform selects wrong pair.
    shots = np.array([0b0101] * 80 + [0b1010] * 19 + [0b1001])
    raw, _ = weighted_product(shots, np.ones(100), 2, (1, 1), 1)
    weighted, _ = weighted_product(
        [0b0101, 0b1010, 0b1001], [0.80, 0.19, 0.01], 2, (1, 1), 1
    )
    assert raw.tolist() == weighted.tolist() == [1]
    weighted2, _ = weighted_product([0b0101, 0b1010], [0.1, 0.9], 2, (1, 1), 1)
    uniform, _ = weighted_product([0b0101, 0b1010], [0.5, 0.5], 2, (1, 1), 1)
    assert weighted2.tolist() == [2] and uniform.tolist() == [1]


def test_distinct_selector_inputs_change_unanchored_choices():
    a = np.array([0, 9, 3, 1, 0, 2, 8, 4.0])
    b = -a
    assert choose(a, 4) == [0, 1, 4, 6]
    assert choose(b, 4) == [0, 3, 4, 5]
    assert [len(options(s)) for s in [2, 4, 6]] == [1, 9, 9]
    angle = 0.3
    u = np.eye(8)
    u[:2, :2] = [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
    d = {
        "dm": u @ np.diag([1.9, 0.1, 2, 2, 0, 0, 0, 0]) @ u.T,
        "entropy": np.arange(8) / 10,
        "mutual_information": np.eye(8) * 0,
    }
    score = score_arrays([d], np.arange(8))
    assert not np.allclose(
        score["diagonal_population"], score["natural_occupation_informed"]
    )


def test_projector_is_idempotent_and_particle_sector_preserving():
    mask = projection_mask(4, (2, 2), [1, 3], 2)
    assert mask.sum() == 4
    c = np.arange(36).reshape(6, 6)
    assert np.array_equal((c * mask) * mask, c * mask)
