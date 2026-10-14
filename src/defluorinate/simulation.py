"""Aer sampling, synthetic noise, and comparisons at matched subspace size."""

import time
from collections import Counter
import numpy as np
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, depolarizing_error, ReadoutError
from .io import save, load_model, sha
from .numerics import explicit_hamiltonian, fci_solve, strings
from .sqd import (
    recover,
    weighted_product,
    sci_energy,
    classical_product,
    readout_mitigate,
)


def noise_model(scale=1.0, readout_only=False):
    if not 0 <= scale < 25:
        raise ValueError("Noise scale must be in [0,25)")
    noise = NoiseModel()
    if not readout_only:
        noise.add_all_qubit_quantum_error(
            depolarizing_error(0.001 * scale, 1), ["sx", "x"]
        )
        noise.add_all_qubit_quantum_error(depolarizing_error(0.01 * scale, 2), ["cx"])
    p = 0.02 * scale
    noise.add_all_qubit_readout_error(ReadoutError([[1 - p, p], [p, 1 - p]]))
    return noise


def compare(samples, model, seed=11, caps=(4, 16, 36), readout_p=None):
    h, g, o, ne = load_model(model)
    n = len(h)
    if n > 4:
        raise ValueError("Comparison requires at most four spatial orbitals")
    H = explicit_hamiltonian(h, g, ne, o)
    exact = fci_solve(h, g, ne, o)[0]
    methods = [("filter", samples, np.ones(len(samples)), 0.0)]
    recovered, prob = recover(samples, n, ne, seed)
    methods.append(("weighted_recovery", recovered, prob, 0.0))
    if readout_p is not None:
        repaired, weights, negative = readout_mitigate(samples, n, readout_p)
        methods.append(("known_readout_inverse", repaired, weights, negative))
    rows = []
    for name, processed, weights, negative in methods:
        for cap in caps:
            tick = time.perf_counter()
            st, info = weighted_product(processed, weights, n, ne, cap)
            if st is None:
                rows.append(
                    {"arm": name, "cap": cap, "status": "no_valid_samples", **info}
                )
                continue
            energy, s2 = sci_energy(h, g, o, st, st)
            sqd_seconds = time.perf_counter() - tick
            tick = time.perf_counter()
            cs = classical_product(H, n, ne, len(st))
            classical, _ = sci_energy(h, g, o, cs, cs)
            classical_seconds = time.perf_counter() - tick
            us = np.sort(
                np.random.default_rng(seed).choice(
                    strings(n, ne[0]), len(st), replace=False
                )
            )
            uniform, _ = sci_energy(h, g, o, us, us)
            if min(energy, classical, uniform) < exact - 1e-8:
                raise ValueError("Variational lower-bound violation")
            rows.append(
                {
                    "arm": name,
                    "cap": cap,
                    "status": "completed",
                    **info,
                    "energy_Eh": energy,
                    "reference_Eh": exact,
                    "error_mEh": 1000 * (energy - exact),
                    "classical_error_mEh": 1000 * (classical - exact),
                    "uniform_error_mEh": 1000 * (uniform - exact),
                    "s_squared": s2,
                    "selected_spin_strings": st,
                    "classical_spin_strings": cs,
                    "sqd_seconds": sqd_seconds,
                    "classical_seconds": classical_seconds,
                    "negative_quasiprobability_mass": negative,
                }
            )
    return rows


def sample(circuit, model, out, shots=2048, seed=11):
    if not 1 <= shots <= 100000:
        raise ValueError("Shots must be in [1,100000]")
    domains = [
        ("ideal", None),
        ("synthetic_noise", noise_model()),
        ("readout_only", noise_model(readout_only=True)),
    ]
    rows = []
    for domain, noise in domains:
        start = time.perf_counter()
        sim = AerSimulator(
            method="statevector",
            noise_model=noise,
            max_parallel_threads=2,
            max_parallel_experiments=1,
        )
        result = sim.run(
            circuit, shots=shots, memory=True, seed_simulator=seed
        ).result()
        memory = result.get_memory()
        samples = np.array([int(b.replace(" ", ""), 2) for b in memory])
        save(
            out / f"raw_{domain}.json",
            {
                "domain": domain + "_simulator",
                "memory": memory,
                "counts": dict(Counter(memory)),
                "shots": shots,
                "seed": seed,
                "seconds": time.perf_counter() - start,
                "circuit_sha256": sha(out / "circuit.qpy"),
            },
        )
        for row in compare(
            samples, model, seed, readout_p=0.02 if domain == "readout_only" else None
        ):
            rows.append({"domain": domain + "_simulator", **row})
    save(out / "comparison.json", rows)
    save(
        out / "noise.json",
        {
            "kind": "synthetic independent channel",
            "depolarizing_sx_x": 0.001,
            "depolarizing_cx": 0.01,
            "readout_flip": 0.02,
            "readout_inverse": "known channel, no calibration; negative quasi-probabilities clipped and renormalized, introducing bias",
            "recovery": "one pass, regularized HF prior, probability preserving; classical mitigation, not quantum error correction",
        },
    )
    return rows
