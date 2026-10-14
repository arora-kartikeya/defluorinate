from pathlib import Path
import numpy as np
import pytest
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector
from defluorinate.chemistry import build_h2
from defluorinate.circuits import prepare
from defluorinate.simulation import sample
from defluorinate.io import read, load_model, sha, save
from defluorinate.numerics import sector
from defluorinate.isa import compact_distribution


def test_fresh_molecule_to_qiskit_samples_and_sqd(tmp_path):
    model = build_h2(tmp_path)
    h, g, o, ne = load_model(model)
    qc = prepare(model, tmp_path, maxiter=3)
    sv = Statevector(qc.remove_final_measurements(inplace=False))
    assert np.isclose(sv.probabilities()[sector(2, ne)].sum(), 1)
    rows = sample(qc, model, tmp_path, shots=128)
    assert {r["domain"] for r in rows} == {
        "ideal_simulator",
        "synthetic_noise_simulator",
        "readout_only_simulator",
    }
    for domain in ["ideal", "synthetic_noise", "readout_only"]:
        raw = read(tmp_path / f"raw_{domain}.json")
        assert len(raw["memory"]) == 128 and sum(raw["counts"].values()) == 128
    assert min(r["error_mEh"] for r in rows if r["status"] == "completed") >= -1e-5
    full = [r for r in rows if r.get("dimension") == 4]
    assert full and max(abs(r["error_mEh"]) for r in full) < 1e-5
    # Mutating the model is visible to resume/hardware identity checks.
    assert read(tmp_path / "preparation.json")["model_sha256"] == sha(model)


def test_compact_physical_to_classical_bit_order():
    qc = QuantumCircuit(12, 4)
    qc.x(9)
    qc.h(2)
    qc.cx(2, 7)
    for physical, classic in [(9, 0), (7, 1), (2, 2), (11, 3)]:
        qc.measure(physical, classic)
    p, used, mapping = compact_distribution(qc)
    assert np.allclose(p[[1, 7]], [0.5, 0.5])
    assert p.sum() == pytest.approx(1)
    assert mapping == {"9": 0, "7": 1, "2": 2, "11": 3}


def test_hardware_rejects_unapproved_or_changed_artifacts(tmp_path, monkeypatch):
    import defluorinate.hardware as hw

    # No IBM client may be constructed for an invalid authorization.
    monkeypatch.setattr(
        hw, "service", lambda *a: pytest.fail("Accessed IBM before validation")
    )
    (tmp_path / "isa.qpy").write_bytes(b"local fixture")
    (tmp_path / "model.npz").write_bytes(b"local fixture")
    m = {
        "jobs": 1,
        "shots": 512,
        "max_execution_time_seconds": 60,
        "qpy": "isa.qpy",
        "qpy_sha256": sha(tmp_path / "isa.qpy"),
        "model": "model.npz",
        "model_sha256": sha(tmp_path / "model.npz"),
    }
    path = tmp_path / "manifest.json"
    save(path, m)
    fingerprint = sha(path)
    with pytest.raises(PermissionError):
        hw.submit(path, "wrong", True)
    with pytest.raises(PermissionError):
        hw.submit(path, fingerprint, False)
    assert hw.validate(path, fingerprint) == m
    (tmp_path / "isa.qpy").write_bytes(b"changed")
    with pytest.raises(ValueError):
        hw.submit(path, fingerprint, True)


def test_hardware_uncertain_submission_cannot_retry(tmp_path, monkeypatch):
    import defluorinate.hardware as hw

    monkeypatch.setattr(
        hw, "service", lambda *a: pytest.fail("Accessed IBM for ambiguous retry")
    )
    for f in ["isa.qpy", "model.npz"]:
        (tmp_path / f).write_bytes(b"fixture")
    m = {
        "jobs": 1,
        "shots": 512,
        "max_execution_time_seconds": 60,
        "qpy": "isa.qpy",
        "qpy_sha256": sha(tmp_path / "isa.qpy"),
        "model": "model.npz",
        "model_sha256": sha(tmp_path / "model.npz"),
    }
    path = tmp_path / "manifest.json"
    save(path, m)
    save(tmp_path / "submission_intent.json", {"status": "unknown"})
    with pytest.raises(RuntimeError):
        hw.submit(path, sha(path), True)


def test_fake_backend_preflight_has_no_submission(tmp_path):
    from qiskit.providers.fake_provider import GenericBackendV2
    from qiskit import qpy
    from defluorinate.hardware import prepare as plan, validate

    model = build_h2(tmp_path)
    qc = QuantumCircuit(4)
    qc.x(0)
    qc.x(2)
    qc.measure_all()
    with open(tmp_path / "circuit.qpy", "wb") as f:
        qpy.dump(qc, f)
    backend = GenericBackendV2(
        num_qubits=5, basis_gates=["id", "rz", "sx", "x", "cx"], seed=11
    )
    dest = tmp_path / "plan"
    manifest = plan(
        tmp_path / "circuit.qpy",
        model,
        backend,
        dest,
        "mock-account",
        "mock-instance",
        512,
        60,
    )
    assert manifest["jobs"] == 1 and manifest["ideal_sector_weight"] == pytest.approx(1)
    assert validate(dest / "manifest.json", manifest["sha256"])["shots"] == 512
    assert not (dest / "job.json").exists()
