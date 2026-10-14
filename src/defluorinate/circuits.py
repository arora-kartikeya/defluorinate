"""The study's bounded UCJ preparation and Qiskit circuit construction."""

import time
import numpy as np
import ffsim
from scipy.optimize import minimize
from qiskit import QuantumCircuit, transpile, qpy
from qiskit.quantum_info import Statevector
from .numerics import explicit_hamiltonian, sector
from .io import load_model, save, sha


def prepare(model, out, maxiter=40, seed=91):
    h, g, o, ne = load_model(model)
    n = len(h)
    if n > 4:
        raise ValueError(
            "Dense training is limited to four spatial orbitals (eight qubits)"
        )
    started = time.perf_counter()
    H = explicit_hamiltonian(h, g, ne, o)
    basis = sector(n, ne)
    reference = ffsim.hartree_fock_state(n, ne)
    count = ffsim.UCJOpSpinBalanced.n_params(n, 1)
    history = []

    def operator(x):
        return ffsim.UCJOpSpinBalanced.from_parameters(x, norb=n, n_reps=1)

    def cost(x):
        state = ffsim.apply_unitary(reference, operator(x), norb=n, nelec=ne)
        v = ffsim.qiskit.ffsim_vec_to_qiskit_vec(state, n, ne)[basis]
        value = float(np.vdot(v, H @ v).real)
        history.append(value)
        return value

    initial = np.random.default_rng(seed).uniform(-0.25, 0.25, count)
    fit = minimize(
        cost,
        initial,
        method="L-BFGS-B",
        bounds=[(-1.0, 1.0)] * count,
        options={"maxiter": maxiter, "ftol": 1e-10, "maxfun": 3000},
    )
    circuit = QuantumCircuit(2 * n)
    circuit.append(ffsim.qiskit.PrepareHartreeFockJW(n, ne), range(2 * n))
    circuit.append(ffsim.qiskit.UCJOpSpinBalancedJW(operator(fit.x)), range(2 * n))
    circuit = transpile(
        circuit,
        basis_gates=["rz", "sx", "x", "cx"],
        optimization_level=2,
        seed_transpiler=11,
    )
    sv = Statevector(circuit).data
    weight = float(np.sum(abs(sv[basis]) ** 2))
    energy = float(np.vdot(sv[basis], H @ sv[basis]).real)
    if abs(weight - 1) > 1e-8 or abs(energy - cost(fit.x)) > 1e-8:
        raise ValueError("Qiskit/ffsim sector or energy mapping mismatch")
    circuit.measure_all()
    with open(out / "circuit.qpy", "wb") as f:
        qpy.dump(circuit, f)
    save(
        out / "preparation.json",
        {
            "method": "one-layer UCJ; bounded random initialization; local ideal L-BFGS-B",
            "model_sha256": sha(model),
            "parameters": fit.x,
            "seed": seed,
            "optimizer_success": bool(fit.success),
            "optimizer_message": str(fit.message),
            "iterations": int(fit.nit),
            "evaluations": len(history),
            "energy_history_Eh": history,
            "direct_circuit_energy_Eh": energy,
            "sector_weight": weight,
            "seconds": time.perf_counter() - started,
            "qubits": circuit.num_qubits,
            "depth": circuit.depth(),
            "gates": dict(circuit.count_ops()),
            "exact_state_initialization": False,
            "bit_order": "qubits 0..n-1 alpha, n..2n-1 beta; printed strings beta then alpha",
            "cost_limit": "Dense classical objective; counted preparation cost; no speedup claim",
        },
    )
    return circuit
