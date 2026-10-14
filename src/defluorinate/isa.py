"""Check a transpiled circuit in measurement-bit order using only occupied wires."""

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector


def compact_distribution(qc):
    used = sorted(
        {
            qc.find_bit(q).index
            for inst in qc.data
            for q in inst.qubits
            if inst.operation.name != "barrier"
        }
    )
    if len(used) > 18:
        raise ValueError("Dry-run statevector cap: 18 actually used physical qubits")
    lookup = {p: i for i, p in enumerate(used)}
    small = QuantumCircuit(len(used))
    mapping = {}
    for inst in qc.data:
        if inst.operation.name == "barrier":
            continue
        qubits = [lookup[qc.find_bit(q).index] for q in inst.qubits]
        if inst.operation.name == "measure":
            mapping[qubits[0]] = qc.find_bit(inst.clbits[0]).index
        else:
            small.append(inst.operation, qubits)
    sv = Statevector(small).data
    probs = np.zeros(1 << qc.num_clbits)
    for bits, p in enumerate(abs(sv) ** 2):
        out = sum(((bits >> q) & 1) << c for q, c in mapping.items())
        probs[out] += p
    return probs, used, {str(used[q]): c for q, c in mapping.items()}
