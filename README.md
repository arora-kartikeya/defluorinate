# Defluorinate

**Team-6 (Kartikeya Arora, Debadrito Roy, Rodrigo Lopez, Abhinav Surendra)**

Qompute 2026 · Track 1: Molecules for a Livable Planet

Trifluoroacetic acid (TFA, CF₃COOH) is a small molecule with three carbon–fluorine bonds, detected in rainwater, rivers, groundwater and drinking water. It can form as certain refrigerants and pesticides break down; in water, its trifluoroacetate form spreads readily and resists degradation, complicating long-term protection of water supplies ([German Environment Agency](https://www.umweltbundesamt.de/en/publikationen/reducing-the-input-of-chemicals-into-waters)). Defluorinate studies how the molecule’s energy and electron arrangement change as one carbon–fluorine bond is stretched. Weakening the bond exposes errors that simplified calculations can hide near equilibrium, making this an early test of models intended for studying fluorine-removal reactions. We combined PySCF calculations, Qiskit simulations and **sample-based quantum diagonalization** on eight IBM qubits, then tested whether we could predict when a simplified molecular model would lose accuracy. Our classical rule estimates the electronic contributions left out; we fixed the rule and recorded its predictions before revealing the larger-model reference answers. The rule correctly identified **all 17 cases exceeding our energy-error target and all 7 meeting it**, across four reduced models at six bond lengths. The target was 1.6 millihartree (about 4.2 kJ/mol) for errors in energy changes along the curve. Applying the estimated correction reduced the worst error by about **94%**, bringing all 24 cases below that target. These are related tests of one molecule, and two missed failures in separate hardware reconstruction tests show that the rule is not a guarantee. Quantum-assisted and classical results were comparable, with no demonstrated quantum advantage. The contribution is a way to anticipate and reduce modeling errors before interpreting a quantum chemistry result. Environmental applications still require the dissolved ion, water interactions and experimental validation: this isolated, neutral-molecule study does not demonstrate pollutant removal. The repository contains code and tests; historical inputs and result datasets remain separate.

## Where the code lives

| File | Read it for |
| --- | --- |
| [chemistry.py](src/defluorinate/chemistry.py) | PySCF RHF, frozen-core integrals, localized TFA orbitals and orbital transport |
| [circuits.py](src/defluorinate/circuits.py) | Hartree–Fock preparation, UCJ ansatz, local optimization, Qiskit transpilation and mapping checks |
| [simulation.py](src/defluorinate/simulation.py) | Aer sampling, declared noise channels, recovery comparisons and classical controls |
| [sqd.py](src/defluorinate/sqd.py) | Qiskit sample-based quantum diagonalization adapter, frequency-weighted configuration selection, recovery and physical CI reconstruction |
| [hardware.py](src/defluorinate/hardware.py) | IBM Runtime planning, exact-manifest approval, one-job submission and retrieval |
| [isa.py](src/defluorinate/isa.py) | Physical-qubit to measurement-bit mapping and ideal checks of transpiled circuits |
| [selection.py](src/defluorinate/selection.py) | Equilibrium, occupation, entropy and mutual-information selectors for the eight-orbital parent |
| [numerics.py](src/defluorinate/numerics.py) | Fermionic Hamiltonian, exact references, orbital RDMs, entropies and frozen-space weights |
| [projection.py](src/defluorinate/projection.py) | Frozen-orbital embedding, projectors and independent energy checks |
| [reliability.py](src/defluorinate/reliability.py) | External residuals and Epstein–Nesbet estimates of omitted contributions |
| [analysis.py](src/defluorinate/analysis.py) | Separate CLI commands for reduced-model diagnostics and reference energies |
| [dmrg.py](src/defluorinate/dmrg.py) | Quimb MPO construction and physical particle-sector extraction |
| [__main__.py](src/defluorinate/__main__.py) | Monitored local workflow and resumable stages |
| [tests/](tests/) | Independent numerical, mapping, recovery and hardware-gate checks |

## Qiskit workflow

1. **Map the molecule.** PySCF supplies the molecular integrals and the energy contribution of the nuclei and frozen electrons. Jordan–Wigner mapping assigns two qubits to each spatial orbital, one for each electron spin. The four-orbital TFA model uses eight qubits.
2. **Prepare the circuit.** `ffsim.qiskit.PrepareHartreeFockJW` prepares the starting electron arrangement. `UCJOpSpinBalancedJW` adds a unitary cluster Jastrow layer that mixes arrangements while preserving electron counts. Parameters are optimized locally, and that classical preparation cost counts toward the experiment.
3. **Collect measurements.** Qiskit transpiles the circuit. Aer provides ideal and noisy simulations; IBM Runtime `SamplerV2` provides real-device samples after explicit approval of the exact circuit and resource manifest. Both paths preserve raw measurements and their mapping to molecular occupations.
4. **Calculate the energy.** The Qiskit sample-based quantum diagonalization add-on recovers configurations with the wrong electron count and solves the Hamiltonian in the selected configuration space. We compare this energy with exact references and classical selection at the same solver dimension.

The final energy comes from a classical calculation informed by quantum samples. Measurement count and selected-space size are different quantities. The tests check energy offsets, electron sectors, bit ordering and the hardware approval gate independently.
