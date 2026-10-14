"""Matrix-free parent projections and explicit determinant embedding conventions."""

import numpy as np
from pyscf import fci
from .numerics import strings, sector


def action(h, g, offset, ci, ne):
    n = len(h)
    heff = fci.direct_spin1.absorb_h1e(h, g, n, ne, 0.5)
    return fci.direct_spin1.contract_2e(heff, ci, n, ne) + offset * ci


def projection_mask(n, ne, active, nocc):
    return np.array(
        [
            all(
                ((int(b) >> i) & 1) == int(i < nocc)
                and ((int(b) >> (n + i)) & 1) == int(i < nocc)
                for i in range(n)
                if i not in active
            )
            for b in sector(n, ne)
        ]
    ).reshape(len(strings(n, ne[0])), -1)


def embed_ci(ci, active, n, nocc):
    """Wedge frozen occupied orbitals with active determinants, tracking parity."""
    active = list(active)
    frozen = [i for i in range(nocc) if i not in active]
    k = nocc - len(frozen)
    small = strings(len(active), k)
    large = strings(n, nocc)
    lookup = {int(b): i for i, b in enumerate(large)}
    indices = []
    signs = []
    for b in small:
        ordered = frozen + [active[j] for j in range(len(active)) if (int(b) >> j) & 1]
        parity = (
            sum(
                ordered[i] > ordered[j]
                for i in range(len(ordered))
                for j in range(i + 1, len(ordered))
            )
            % 2
        )
        indices.append(lookup[sum(1 << i for i in ordered)])
        signs.append((-1) ** parity)
    out = np.zeros((len(large), len(large)))
    out[np.ix_(indices, indices)] = ci * np.outer(signs, signs)
    return out


def rdm_energy(h, g, offset, ci, ne):
    d1, d2 = fci.direct_spin1.make_rdm12(ci, len(h), ne)
    e = float(np.einsum("pq,qp", h, d1) + 0.5 * np.einsum("pqrs,pqrs", g, d2) + offset)
    return e, d1, d2
