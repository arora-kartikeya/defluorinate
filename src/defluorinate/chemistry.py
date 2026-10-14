"""PySCF integrals and the original localized TFA orbital construction."""

from pathlib import Path
import numpy as np
from scipy.linalg import eigh, svd
from pyscf import gto, scf, mcscf, ao2mo, lo, lib
from .io import save, sha

lib.num_threads(2)


def molecule(symbols, xyz, basis="def2-svp", spin=0):
    return gto.M(
        atom=list(zip(symbols, xyz)),
        basis=basis,
        unit="Angstrom",
        charge=0,
        spin=spin,
        verbose=0,
        max_memory=700,
    )


def scan_geometry(geo, mult):
    xyz = np.array(geo["xyz_angstrom"])
    c, f = geo["carbon_index"], geo["fluorine_index"]
    xyz[f] = xyz[c] + mult * (xyz[f] - xyz[c])
    for i in range(len(xyz)):
        for j in range(i):
            if np.linalg.norm(xyz[i] - xyz[j]) < 0.65:
                raise ValueError("Unintended short contact")
    return xyz


def orth_transport(cross, old, new):
    overlap = new.T @ cross.T @ old
    u, s, vh = svd(overlap, full_matrices=False)
    return new @ u @ vh, s


def initial_orbitals(mol, mf):
    S = mf.get_ovlp()
    vals, vec = eigh(S)
    sqrtS = (vec * np.sqrt(vals)) @ vec.T
    weights = (sqrtS @ mf.mo_coeff) ** 2
    labels = mol.ao_labels(fmt=False)
    core_ao = [i for i, (_, _, shell, _) in enumerate(labels) if shell == "1s"]
    core_score = weights[core_ao, :28].sum(axis=0)
    core = sorted(np.argsort(core_score)[-7:].tolist())
    if min(core_score[core]) < 0.65:
        raise ValueError("Chemical core identification ambiguous")
    val = [i for i in range(28) if i not in core]
    occ = lo.Boys(mol, mf.mo_coeff[:, val])
    occ.max_cycle = 150
    occmo = occ.kernel()
    # Lowest twelve virtual orbitals provide an explicitly finite virtual reservoir.
    vir = lo.Boys(mol, mf.mo_coeff[:, 28:40])
    vir.max_cycle = 150
    virmo = vir.kernel()
    mo = np.column_stack([mf.mo_coeff[:, core], occmo, virmo, mf.mo_coeff[:, 40:]])
    w = (sqrtS @ mo) ** 2
    pop = np.array(
        [
            w[mol.aoslice_by_atom()[i, 2] : mol.aoslice_by_atom()[i, 3]].sum(axis=0)
            for i in range(mol.natm)
        ]
    )
    # Two-centre character: geometric mean populations on selected C and F.
    cf = np.sqrt(pop[5] * pop[0])
    env = pop[5] + pop[0] + 0.15 * (pop[6] + pop[3] + pop[4])
    occidx = list(range(7, 28))
    viridx = list(range(28, 40))
    anchor_o = max(occidx, key=lambda i: (cf[i], -i))
    anchor_v = max(viridx, key=lambda i: (cf[i], -i))
    select = (
        lambda seq, anchor: [anchor]
        + sorted([i for i in seq if i != anchor], key=lambda i: (-env[i], i))[:3]
    )
    ids = select(occidx, anchor_o) + select(viridx, anchor_v)
    meta = {
        "core_canonical_indices": core,
        "core_1s_lowdin_weights": core_score[core],
        "parent_mo_ids": ids,
        "anchors_parent": [0, 4],
        "cf_character_score": cf[ids],
        "atomic_lowdin_populations": pop[:, ids],
        "occupied_localization": "Boys valence occupied separately",
        "virtual_localization": "Boys lowest 12 RHF virtuals separately",
        "finite_virtual_reservoir": 12,
        "parent_rationale": "Four occupied and four virtual localized orbitals ranked by reactive C/F plus neighbouring carboxyl populations; mandatory maximal two-centre C/F anchors",
        "fock_diagonal_hartree": np.diag(mo.T @ mf.get_fock() @ mo)[ids],
    }
    return mo, meta


def active_integrals(mf, mo, active, nelec):
    """Include nuclear and inactive-electron energy exactly once."""
    occupied = mf.mol.nelectron // 2
    inactive = [i for i in range(occupied) if i not in active]
    if mf.mol.nelectron - 2 * len(inactive) != sum(nelec):
        raise ValueError("Frozen electrons disagree with the requested active sector")
    coeff = mo[:, inactive + list(active)]
    cas = mcscf.CASCI(mf, len(active), nelec)
    h, offset = cas.get_h1eff(coeff)
    g = ao2mo.restore(1, cas.get_h2eff(coeff), len(active))
    return h, g, float(offset)


def build_h2(out, distance=0.735):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    mol = gto.M(
        atom=[("H", (0, 0, 0)), ("H", (0, 0, distance))],
        basis="sto-3g",
        verbose=0,
        max_memory=700,
    )
    mf = scf.RHF(mol).run(conv_tol=1e-11)
    if not mf.converged:
        raise RuntimeError("H2 RHF did not converge")
    h, g, o = active_integrals(mf, mf.mo_coeff, [0, 1], (1, 1))
    np.savez_compressed(out / "model.npz", h=h, g=g, offset=o, nelec=[1, 1])
    save(
        out / "chemistry.json",
        {
            "molecule": "H2",
            "basis": "sto-3g",
            "distance_angstrom": distance,
            "rhf_Eh": mf.e_tot,
            "norb": 2,
            "nelec": [1, 1],
            "source": "user-selected analytic two-atom geometry",
        },
    )
    return out / "model.npz"


def build_tfa(xyz_path, out):
    """Build the eight-orbital parent from an explicit input geometry.

    The original population selector assumes PubChem CID 6422 atom order:
    F,F,F,O,O,C,C,H. No geometry optimization or historical-run data is implicit.
    """
    from .numerics import reduce_integrals

    lines = Path(xyz_path).read_text().splitlines()
    if int(lines[0]) != 8:
        raise ValueError("TFA requires eight atoms")
    rows = [line.split() for line in lines[2:10]]
    symbols = [row[0] for row in rows]
    if symbols != ["F", "F", "F", "O", "O", "C", "C", "H"]:
        raise ValueError("Required atom order: F F F O O C C H; reactive bond C6-F1")
    xyz = np.array([[float(v) for v in row[1:4]] for row in rows])
    mol = molecule(symbols, xyz)
    if mol.nelectron != 56:
        raise ValueError("Expected neutral 56-electron TFA")
    mf = scf.RHF(mol).run(conv_tol=1e-10, max_cycle=150)
    if not mf.converged:
        raise RuntimeError("TFA RHF did not converge")
    mo, meta = initial_orbitals(mol, mf)
    h, g, o = active_integrals(mf, mo, meta["parent_mo_ids"], (4, 4))
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / "parent.npz", h=h, g=g, offset=o, nelec=[4, 4])
    # Frozen four-orbital choice used for the original 8-qubit campaign.
    hh, gg, oo, ne = reduce_integrals(h, g, o, [0, 2, 4, 6], 4)
    np.savez_compressed(out / "model.npz", h=hh, g=gg, offset=oo, nelec=ne)
    np.savez_compressed(out / "orbitals.npz", mo=mo)
    save(
        out / "chemistry.json",
        {
            "molecule": "TFA",
            "basis": "def2-svp",
            "input_sha256": sha(xyz_path),
            "symbols": symbols,
            "xyz_angstrom": xyz,
            "rhf_Eh": mf.e_tot,
            "orbital_metadata": meta,
            "reduced_parent_indices": [0, 2, 4, 6],
            "scope": "single user-supplied geometry; no implicit equilibrium optimization or curve transport",
        },
    )
    return out / "model.npz"
