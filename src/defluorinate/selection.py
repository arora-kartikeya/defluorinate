"""Pilot-informed selectors for the eight-orbital TFA parent."""

import itertools
import numpy as np


def choose(score, size, anchors=(0, 4)):
    active = list(anchors)
    for pool in [range(4), range(4, 8)]:
        active += sorted(
            [i for i in pool if i not in active], key=lambda i: (-score[i], i)
        )[: size // 2 - 1]
    return sorted(active)


def score_arrays(data, fock):
    pop = np.max(
        [np.minimum(np.diag(d["dm"]), 2 - np.diag(d["dm"])) for d in data], axis=0
    )
    nat = []
    for d in data:
        eig, u = np.linalg.eigh(d["dm"])
        nat.append((u * u) @ np.minimum(eig, 2 - eig))
    ent = np.max([d["entropy"] for d in data], axis=0)
    pair = np.max(
        [d["mutual_information"][:, [0, 4]].sum(axis=1) for d in data], axis=0
    )
    return {
        "equilibrium_informed": np.r_[fock[:4], -fock[4:]],
        "diagonal_population": pop,
        "natural_occupation_informed": np.max(nat, axis=0),
        "entropy": ent,
        "mutual_information": np.max(
            [d["mutual_information"].sum(axis=1) for d in data], axis=0
        ),
        "correlation_initial": ent / max(ent.max(), 1e-15)
        + pair / max(pair.max(), 1e-15),
    }


def options(size):
    return [
        sorted([0, 4] + list(a) + list(b))
        for a in itertools.combinations([1, 2, 3], size // 2 - 1)
        for b in itertools.combinations([5, 6, 7], size // 2 - 1)
    ]
