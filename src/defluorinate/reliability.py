"""Non-oracle residual and EN diagnostics using sparse excitation-link contractions.

No exact parent eigenpair enters these APIs. PySCF contracts the two-body
Hamiltonian through determinant excitation links, with O(sector dimension)
state storage and no dense parent Hamiltonian. All connected Q amplitudes are
retained (no coefficient screening); this is not constant cost in parent size.
"""

import time
import numpy as np
from pyscf import fci
from .numerics import strings, reduce_integrals, fci_solve, entropy, local_rdm
from .projection import embed_ci, projection_mask


def linked_action(h, g, ci, ne, offset=0.0):
    n = len(h)
    links = (
        fci.cistring.gen_linkstr_index_trilidx(range(n), ne[0]),
        fci.cistring.gen_linkstr_index_trilidx(range(n), ne[1]),
    )
    heff = fci.direct_spin1.absorb_h1e(h, g, n, ne, 0.5)
    return fci.direct_spin1.contract_2e(heff, ci, n, ne, link_index=links) + offset * ci


def external_diagnostics(h, g, offset, ci, ne, mask, eta_values=(0.0, 0.01, 0.05, 0.1)):
    tick = time.perf_counter()
    n = len(h)
    c = np.asarray(ci, float).reshape(len(strings(n, ne[0])), -1)
    mask = np.asarray(mask, bool).reshape(c.shape)
    if abs(np.vdot(c, c).real - 1) > 1e-8 or np.linalg.norm(c[~mask]) > 1e-10:
        raise ValueError("State must be normalized and supported in P")
    hc = linked_action(h, g, c, ne, offset)
    energy = float(np.vdot(c, hc).real)
    q = hc.copy()
    q[mask] = 0
    diag = fci.direct_spin1.make_hdiag(h, g, n, ne).reshape(c.shape) + offset
    den = energy - diag
    connected = (~mask) & (abs(q) > 1e-14)
    dq = den[connected]
    v2 = q[connected] ** 2
    small = abs(dq) < 0.05
    intruder = (dq >= 0) | small
    corrections = {}
    for eta in eta_values:
        if eta == 0:
            val = None if np.any(abs(dq) < 1e-10) else float(np.sum(v2 / dq))
        else:
            val = float(np.sum(v2 * dq / (dq * dq + eta * eta)))
        corrections[str(eta)] = val
    residual = hc - energy * c
    internal = float(np.linalg.norm(residual[mask]))
    ss = float(fci.spin_op.spin_square(c, n, ne)[0])
    return (
        {
            "energy": energy,
            "rQ_Eh": float(np.linalg.norm(q)),
            "internal_residual_Eh": internal,
            "full_residual_Eh": float(np.linalg.norm(residual)),
            "pt2_Eh": corrections["0.0"],
            "regularized_Eh": corrections,
            "connected_excluded": int(connected.sum()),
            "selected_dimension": int(mask.sum()),
            "parent_dimension": c.size,
            "min_abs_denominator_Eh": float(abs(dq).min()) if len(dq) else None,
            "intruder_count": int(intruder.sum()),
            "positive_denominator_count": int((dq >= 0).sum()),
            "small_denominator_weight": float(v2[small].sum()),
            "s_squared": ss,
            "diagnostic_seconds": time.perf_counter() - tick,
        },
        q,
        diag,
    )


def reduced_model(h, g, o, active):
    tick = time.perf_counter()
    hh, gg, oo, ne = reduce_integrals(h, g, o, active, 4)
    e, ci, dm, ss = fci_solve(hh, gg, ne, oo)
    solve = time.perf_counter() - tick
    embedded = embed_ci(ci, active, len(h), 4)
    mask = projection_mask(len(h), (4, 4), active, 4)
    d, q, diag = external_diagnostics(h, g, o, embedded, (4, 4), mask)
    assert abs(d["energy"] - e) < 1e-8 and d["internal_residual_Eh"] < 2e-5
    t = time.perf_counter()
    occ = np.linalg.eigvalsh(dm)
    s = sum(entropy(local_rdm(ci, len(active), ne, [i])) for i in range(len(active)))
    d.update(
        norb=len(active),
        active=active,
        solve_seconds=solve,
        occupation_indicator=float(np.minimum(occ, 2 - occ).sum()),
        entropy_indicator=s,
        baseline_seconds=time.perf_counter() - t,
    )
    return d, embedded, q, diag, mask


def predict(rows):
    """Frozen transparent thresholds; nothing fitted to reference errors."""
    for d in rows:
        eq = next(
            x
            for x in rows
            if x["parent_norb"] == d["parent_norb"]
            and x["model"] == d["model"]
            and x["multiplier"] == 1.0
        )
        d["en_predicted_signed_profile_mEh"] = (
            None
            if d["pt2_Eh"] is None or eq["pt2_Eh"] is None
            else -1000 * (d["pt2_Eh"] - eq["pt2_Eh"])
        )
        d["residual_predicted_signed_profile_mEh"] = 1000 * (
            d["rQ_Eh"] ** 2 - eq["rQ_Eh"] ** 2
        )
        d["flag_en"] = (
            d["intruder_count"] > 0
            or eq["intruder_count"] > 0
            or d["en_predicted_signed_profile_mEh"] is None
            or abs(d["en_predicted_signed_profile_mEh"]) > 1.6
        )
        d["flag_residual"] = abs(d["residual_predicted_signed_profile_mEh"]) > 1.6
        d["flag_size"] = d["norb"] < 6
        d["flag_occupation"] = d["occupation_indicator"] > 0.05
        d["flag_entropy"] = d["entropy_indicator"] > 0.2
    return rows
