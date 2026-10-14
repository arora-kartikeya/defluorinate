"""Local experiment runner with one worker, checkpoints, and an RSS ceiling."""

import argparse
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

THREADS = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)
for name in THREADS:
    os.environ[name] = "2"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"


def worker(a):
    from .io import save, read, sha, environment, utc
    from .chemistry import build_h2, build_tfa
    from .circuits import prepare
    from .simulation import sample
    from .plots import plot_comparison
    from qiskit import qpy

    out = a.out
    out.mkdir(parents=True, exist_ok=True)
    source = {p.name: sha(p) for p in Path(__file__).parent.glob("*.py")}
    key = {
        "mode": a.mode,
        "shots": a.shots,
        "seed": a.seed,
        "maxiter": a.maxiter,
        "input_sha256": sha(a.xyz or a.model) if (a.xyz or a.model) else None,
        "source": source,
        "environment": environment(),
    }
    checkpoint = out / "checkpoint.json"
    if checkpoint.exists():
        old = read(checkpoint)
        if not a.resume:
            raise FileExistsError(
                "Use --resume for a matching checkpoint or choose a new output directory"
            )
        if old["key"] != key:
            raise ValueError(
                "Code, environment or inputs changed; choose a new output directory"
            )
        for name, fingerprint in old.get("artifacts", {}).items():
            if sha(out / name) != fingerprint:
                raise ValueError("Checkpoint artifact changed: " + name)
        completed = old["completed"]
    else:
        completed = []

    def mark(stage):
        completed.append(stage)
        artifacts = {
            str(p.relative_to(out)): sha(p)
            for p in out.rglob("*")
            if p.is_file()
            and p.name not in ["checkpoint.json", "run.json", "worker.log", ".lock"]
        }
        save(
            checkpoint,
            {"key": key, "completed": completed, "artifacts": artifacts, "at": utc()},
        )
        print("Completed", stage, flush=True)

    model = out / "model.npz"
    if "chemistry" not in completed:
        if a.mode == "demo":
            build_h2(out)
        elif a.mode == "tfa":
            build_tfa(a.xyz, out)
        else:
            from shutil import copy2

            copy2(a.model, model)
        mark("chemistry")
    if "preparation" not in completed:
        circuit = prepare(model, out, maxiter=a.maxiter, seed=91)
        mark("preparation")
    else:
        with open(out / "circuit.qpy", "rb") as f:
            circuit = qpy.load(f)[0]
    if "simulation" not in completed:
        sample(circuit, model, out, shots=a.shots, seed=a.seed)
        mark("simulation")
    if "figures" not in completed:
        plot_comparison(out)
        mark("figures")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=["demo", "tfa", "simulate"])
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--xyz", type=Path)
    p.add_argument("--model", type=Path)
    p.add_argument("--shots", type=int, default=2048)
    p.add_argument("--seed", type=int, default=11)
    p.add_argument("--maxiter", type=int, default=40)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--rss-gib", type=float, default=3.5)
    p.add_argument("--timeout", type=float, default=3600)
    p.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    a = p.parse_args()
    if a.mode == "tfa" and not a.xyz:
        p.error("tfa requires --xyz")
    if a.mode == "simulate" and not a.model:
        p.error("simulate requires --model")
    if not 0 < a.rss_gib <= 3.5 or not 0 < a.timeout <= 14400:
        p.error("RSS limit must be <=3.5 GiB; timeout <=4 hours")
    if not 1 <= a.maxiter <= 40 or not 1 <= a.shots <= 100000:
        p.error("maxiter must be 1..40; shots 1..100000")
    if a.worker:
        return worker(a)
    import fcntl
    import psutil
    from .io import save, utc

    a.out.mkdir(parents=True, exist_ok=True)
    # A shared per-user lock prevents two heavy workflows, even in different folders.
    lock_path = (
        Path(os.environ.get("TMPDIR", "/tmp")) / f"defluorinate-{os.getuid()}.lock"
    )
    with open(lock_path, "a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        start = time.monotonic()
        peak = 0
        reason = None
        with open(a.out / "worker.log", "a") as log:
            child = subprocess.Popen(
                [sys.executable, "-m", "defluorinate", *sys.argv[1:], "--worker"],
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            try:
                process = psutil.Process(child.pid)
                while child.poll() is None:
                    try:
                        peak = max(
                            peak,
                            process.memory_info().rss
                            + sum(
                                x.memory_info().rss
                                for x in process.children(recursive=True)
                            ),
                        )
                    except psutil.NoSuchProcess:
                        pass
                    if peak > a.rss_gib * 1024**3:
                        reason = "RSS ceiling exceeded"
                    if time.monotonic() - start > a.timeout:
                        reason = "Timeout exceeded"
                    if reason:
                        os.killpg(child.pid, signal.SIGTERM)
                        try:
                            child.wait(timeout=5)
                        except subprocess.TimeoutExpired:
                            os.killpg(child.pid, signal.SIGKILL)
                        break
                    time.sleep(0.25)
            except KeyboardInterrupt:
                os.killpg(child.pid, signal.SIGTERM)
                reason = "Interrupted"
            except Exception as exc:
                if child.poll() is None:
                    os.killpg(child.pid, signal.SIGTERM)
                reason = f"Monitor failed: {type(exc).__name__}"
            code = child.wait()
        record = {
            "at": utc(),
            "status": "completed" if code == 0 and reason is None else "failed",
            "returncode": code,
            "failure": reason,
            "peak_worker_rss_gib": peak / 1024**3,
            "seconds": time.monotonic() - start,
            "new_hardware_jobs": 0,
            "threads": 2,
        }
        save(a.out / "run.json", record)
        print(record)
        if record["status"] != "completed":
            raise SystemExit("Worker failed; see " + str(a.out / "worker.log"))


if __name__ == "__main__":
    main()
