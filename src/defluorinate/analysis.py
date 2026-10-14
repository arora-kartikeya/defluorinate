"""Separate model-reduction diagnostics from reference evaluation."""

import argparse
from pathlib import Path
import numpy as np
from .io import load_model, save, sha
from .numerics import fci_solve, information, reduce_integrals
from .projection import embed_ci, projection_mask
from .reliability import external_diagnostics


def diagnose(model, active, out):
    h, g, o, ne = load_model(model)
    n = len(h)
    if n > 10:
        raise ValueError("Local diagnostic cap: ten spatial orbitals")
    if (
        sorted(set(active)) != active
        or not active
        or min(active) < 0
        or max(active) >= n
    ):
        raise ValueError("Active indices must be unique, sorted and in range")
    hh, gg, oo, reduced_ne = reduce_integrals(h, g, o, active, ne[0])
    energy, ci, dm, ss = fci_solve(hh, gg, reduced_ne, oo)
    embedded = embed_ci(ci, active, n, ne[0])
    mask = projection_mask(n, ne, active, ne[0])
    diagnostic, _, _ = external_diagnostics(h, g, o, embedded, ne, mask)
    result = {
        "parent_model_sha256": sha(model),
        "active": active,
        "nelec": reduced_ne,
        "reduced_energy_Eh": energy,
        **diagnostic,
        "scope": "No parent eigenpair used. An EN correction is heuristic and nonvariational; zero residual does not identify the ground-state branch.",
    }
    save(out, result)
    return result


def reference(model, out):
    h, g, o, ne = load_model(model)
    if len(h) > 10:
        raise ValueError("Local exact-reference cap: ten orbitals")
    e, ci, dm, ss = fci_solve(h, g, ne, o)
    save(
        out,
        {
            "model_sha256": sha(model),
            "energy_Eh": e,
            "s_squared": ss,
            "one_rdm": dm,
            "scope": "same-sector finite-Hamiltonian reference",
        },
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=["diagnose", "reference"])
    p.add_argument("model", type=Path)
    p.add_argument("--active", type=int, nargs="+")
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    if a.out.exists():
        raise FileExistsError("Use a new output path")
    if a.mode == "diagnose":
        if a.active is None:
            p.error("--active is required for diagnose")
        diagnose(a.model, a.active, a.out)
    else:
        reference(a.model, a.out)


if __name__ == "__main__":
    main()
