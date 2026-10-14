"""Explicit-instance Runtime jobs, bound to reviewed circuits and resource limits.

Planning is read-only on IBM. Submission is a separate command and requires the
SHA-256 of the exact reviewed manifest. No account, instance, or approval is bundled.
"""

from pathlib import Path
import argparse
import math
import numpy as np
from qiskit import qpy
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2
from .io import read, save, sha, utc, load_model
from .isa import compact_distribution
from .numerics import sector


def service(account, instance):
    if not account or not instance:
        raise ValueError("Select a saved account and exact instance explicitly")
    client = QiskitRuntimeService(name=account, instance=instance)
    if client.active_instance() != instance:
        raise ValueError("Active IBM instance differs from requested instance")
    return client


def prepare(
    circuit_path, model_path, backend, out, account, instance, shots, max_seconds
):
    out = Path(out)
    if out.exists():
        raise FileExistsError("Use a new hardware plan directory")
    h, g, o, ne = load_model(model_path)
    n = len(h)
    if n > 4 or not 1 <= shots <= 100000 or not 1 <= max_seconds <= 60:
        raise ValueError(
            "Limits: eight logical qubits, 100000 shots, 60 runtime seconds per job"
        )
    with open(circuit_path, "rb") as f:
        circuit = qpy.load(f)[0]
    if circuit.num_qubits != 2 * n or circuit.num_clbits != 2 * n:
        raise ValueError("Circuit/model width mismatch")
    isa = generate_preset_pass_manager(
        backend=backend, optimization_level=3, seed_transpiler=11
    ).run(circuit)
    probabilities, used, mapping = compact_distribution(isa)
    original, _, _ = compact_distribution(circuit)
    probability_error = float(np.max(abs(probabilities - original)))
    if probability_error > 1e-8:
        raise ValueError("Transpilation changed measured probabilities")
    weight = float(probabilities[sector(n, ne)].sum())
    if abs(weight - 1) > 1e-8:
        raise ValueError("Transpiled ISA failed measurement-sector check")
    out.mkdir(parents=True)
    import shutil

    shutil.copy2(model_path, out / "model.npz")
    with open(out / "isa.qpy", "wb") as f:
        qpy.dump(isa, f)
    manifest = {
        "schema": 1,
        "created_at": utc(),
        "account": account,
        "instance": instance,
        "backend": backend.name,
        "jobs": 1,
        "shots": shots,
        "max_execution_time_seconds": max_seconds,
        "qpy": "isa.qpy",
        "qpy_sha256": sha(out / "isa.qpy"),
        "model": "model.npz",
        "model_sha256": sha(out / "model.npz"),
        "logical_qubits": 2 * n,
        "used_physical_qubits": used,
        "measurement_mapping": mapping,
        "depth": isa.depth(),
        "gate_counts": dict(isa.count_ops()),
        "ideal_sector_weight": weight,
        "maximum_probability_error": probability_error,
        "source_circuit_sha256": sha(circuit_path),
        "rep_delay_seconds": 0.00025,
        "cost_note": "Runtime limit is a bound, not a usage estimate. Allocation availability and any institutional monetary charge must be reviewed before submission. No automatic backend fallback.",
    }
    save(out / "manifest.json", manifest)
    return {
        "manifest": str(out / "manifest.json"),
        "sha256": sha(out / "manifest.json"),
        **manifest,
    }


def validate(manifest_path, approved_sha):
    path = Path(manifest_path)
    m = read(path)
    if not approved_sha or sha(path) != approved_sha:
        raise PermissionError("Exact manifest SHA-256 approval required")
    if (
        m.get("jobs") != 1
        or not 1 <= m.get("shots", 0) <= 100000
        or not 1 <= m.get("max_execution_time_seconds", 0) <= 60
    ):
        raise ValueError("Invalid reviewed resource limits")
    for filekey, hashkey in [("qpy", "qpy_sha256"), ("model", "model_sha256")]:
        file = (path.parent / m[filekey]).resolve()
        if not file.is_relative_to(path.parent.resolve()) or sha(file) != m[hashkey]:
            raise ValueError("Reviewed artifact path/hash mismatch")
    return m


def submit(manifest_path, approved_sha, acknowledge_cost=False):
    m = validate(manifest_path, approved_sha)
    if not acknowledge_cost:
        raise PermissionError("Explicit allocation/cost acknowledgement required")
    folder = Path(manifest_path).parent
    receipt = folder / "job.json"
    if receipt.exists():
        return read(receipt)
    if (folder / "submission_intent.json").exists():
        raise RuntimeError(
            "Unresolved submission intent. Inspect platform workloads before any retry; do not delete this file blindly."
        )
    client = service(m["account"], m["instance"])
    backend = client.backend(m["backend"], instance=m["instance"])
    if not backend.status().operational:
        raise RuntimeError("Reviewed backend is not operational")
    usage = client.usage()
    remaining = usage.get("usage_remaining_seconds")
    if remaining is None or remaining < m["max_execution_time_seconds"]:
        raise RuntimeError(
            "Remaining allocation is unknown or below the reviewed job limit"
        )
    with open(folder / m["qpy"], "rb") as f:
        circuit = qpy.load(f)[0]
    for item in circuit.data:
        if item.operation.name == "barrier":
            continue
        qargs = tuple(circuit.find_bit(q).index for q in item.qubits)
        if not backend.target.instruction_supported(
            operation_name=item.operation.name, qargs=qargs
        ):
            raise ValueError("Saved ISA is no longer supported")
    # O_EXCL prevents a concurrent or ambiguous retry from submitting a duplicate.
    with open(folder / "submission_intent.json", "x") as f:
        import json

        json.dump(
            {
                "at": utc(),
                "manifest_sha256": approved_sha,
                "status": "submission_attempt_started",
            },
            f,
        )
    sampler = SamplerV2(
        mode=backend,
        options={
            "max_execution_time": m["max_execution_time_seconds"],
            "execution": {"rep_delay": m["rep_delay_seconds"]},
            "environment": {"job_tags": ["defluorinate", approved_sha[:16]]},
        },
    )
    job = sampler.run([(circuit, None, m["shots"])])
    record = {
        "job_id": job.job_id(),
        "submitted_at": utc(),
        "manifest_sha256": approved_sha,
        "backend": m["backend"],
        "shots": m["shots"],
        "max_execution_time_seconds": m["max_execution_time_seconds"],
    }
    save(receipt, record)
    return record


def retrieve(manifest_path):
    folder = Path(manifest_path).parent
    m = read(manifest_path)
    receipt = read(folder / "job.json")
    if receipt["manifest_sha256"] != sha(manifest_path):
        raise ValueError("Manifest changed after submission")
    job = service(m["account"], m["instance"]).job(receipt["job_id"])
    status = str(job.status())
    save(
        folder / "status.json",
        {"at": utc(), "job_id": receipt["job_id"], "status": status},
    )
    if status not in ["DONE", "JobStatus.DONE"]:
        return {"status": status, "job_id": receipt["job_id"]}
    bits = job.result()[0].data.meas
    memory = bits.get_bitstrings()
    if len(memory) != m["shots"]:
        raise ValueError("Returned shot count disagrees with manifest")
    save(
        folder / "raw_hardware.json",
        {
            "domain": "ibm_hardware",
            "job_id": receipt["job_id"],
            "memory": memory,
            "counts": bits.get_counts(),
            "shots": bits.num_shots,
            "bit_order": "meas classical register, printed MSB first",
            "qpy_sha256": m["qpy_sha256"],
        },
    )
    save(folder / "usage.json", job.usage())
    save(folder / "metrics.json", job.metrics())
    from .simulation import compare

    samples = np.array([int(b, 2) for b in memory])
    rows = compare(samples, folder / m["model"])
    save(folder / "comparison.json", [{"domain": "ibm_hardware", **r} for r in rows])
    return {"status": "DONE", "job_id": receipt["job_id"], "shots": len(memory)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    plan = sub.add_parser("plan")
    plan.add_argument("--circuit", type=Path, required=True)
    plan.add_argument("--model", type=Path, required=True)
    plan.add_argument("--out", type=Path, required=True)
    plan.add_argument("--account", required=True)
    plan.add_argument("--instance", required=True)
    plan.add_argument("--backend", required=True)
    plan.add_argument("--shots", type=int, required=True)
    plan.add_argument("--max-seconds", type=int, required=True)
    send = sub.add_parser("submit")
    send.add_argument("manifest", type=Path)
    send.add_argument("--approve-sha", required=True)
    send.add_argument("--acknowledge-allocation-cost", action="store_true")
    get = sub.add_parser("retrieve")
    get.add_argument("manifest", type=Path)
    a = parser.parse_args()
    if a.action == "plan":
        client = service(a.account, a.instance)
        backend = client.backend(a.backend, instance=a.instance)
        print(
            prepare(
                a.circuit,
                a.model,
                backend,
                a.out,
                a.account,
                a.instance,
                a.shots,
                a.max_seconds,
            )
        )
    elif a.action == "submit":
        print(submit(a.manifest, a.approve_sha, a.acknowledge_allocation_cost))
    else:
        print(retrieve(a.manifest))


if __name__ == "__main__":
    main()
