import numpy as np
from pyscf import gto, scf, ao2mo
from defluorinate.numerics import *


def h2():
    mol = gto.M(atom="H 0 0 0; H 0 0 .74", basis="sto-3g", verbose=0)
    mf = scf.RHF(mol).run(conv_tol=1e-12)
    c = mf.mo_coeff
    return (
        c.T @ mf.get_hcore() @ c,
        ao2mo.restore(1, ao2mo.kernel(mol, c), 2),
        mol.energy_nuc(),
        mf.e_tot,
    )


def test_three_hamiltonians_and_offset():
    h, g, o, hf = h2()
    H = explicit_hamiltonian(h, g, (1, 1), o)
    e, c, dm, ss = fci_solve(h, g, (1, 1), o)
    assert abs(e - np.linalg.eigvalsh(H)[0]) < 1e-8
    assert abs(H[0, 0] - hf) < 1e-8
    op = jw_operator(h, g, o)
    ids = sector(2, (1, 1))
    mapped = op.to_matrix(sparse=True)[ids][:, ids].toarray()
    assert np.max(abs(mapped - H)) < 1e-8
    assert abs(np.trace(dm) - 2) < 1e-10 and abs(ss) < 1e-8
    hr, gr, off, ne = reduce_integrals(h, g, o, [1], 1)
    assert abs(off - hf) < 1e-8 and ne == (0, 0)


def test_projector_and_orbital_rdms():
    c = np.zeros((2, 2))
    c[0, 0] = 1
    assert projector_weight(c, 2, (1, 1), [], 1) == 1
    c = c[::-1, ::-1].copy()
    assert projector_weight(c, 2, (1, 1), [], 1) == 0
    c = np.eye(2) / np.sqrt(2)
    assert abs(projector_weight(c, 2, (1, 1), [], 1) - 0.5) < 1e-12
    s, mi = information(c, 2, (1, 1))
    assert np.allclose(s, np.log(2))
    assert abs(mi[0, 1] - np.log(2)) < 1e-12
    rho = local_rdm(c, 2, (1, 1), [0, 1])
    assert abs(np.trace(rho @ rho) - 1) < 1e-12


def test_bit_order():
    from qiskit import QuantumCircuit
    from qiskit.quantum_info import Statevector

    for bits in sector(3, (1, 1)):
        qc = QuantumCircuit(6)
        for i in range(6):
            if (int(bits) >> i) & 1:
                qc.x(i)
        assert np.argmax(abs(Statevector(qc).data)) == bits


def test_projector_explicit_diagonal_and_nonadjacent_rdm():
    rng = np.random.default_rng(7)
    ci = rng.normal(size=(3, 3))
    ci /= np.linalg.norm(ci)
    bits = sector(3, (1, 1))
    mask = np.array([not (int(b) & 2 or int(b) & 16) for b in bits])
    P = np.diag(mask.astype(float))
    assert (
        abs(projector_weight(ci, 3, (1, 1), [0, 2], 1) - ci.ravel() @ P @ ci.ravel())
        < 1e-12
    )
    # Nonadjacent sites (0,2), local order a0,b0,a2,b2. Check a0^+ a2.
    rho = local_rdm(ci, 3, (1, 1), [0, 2])
    observable = np.zeros((16, 16))
    for j in range(16):
        i, sign = apply(j, [(0, True), (2, False)])
        if sign:
            observable[i, j] = sign
    full = np.zeros((9, 9))
    lookup = {int(b): i for i, b in enumerate(bits)}
    for j, b in enumerate(bits):
        i, sign = apply(int(b), [(0, True), (2, False)])
        if sign and i in lookup:
            full[lookup[i], j] = sign
    assert abs(np.trace(rho @ observable) - ci.ravel() @ full @ ci.ravel()) < 1e-12
