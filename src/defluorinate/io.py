"""Small, atomic run records. Generated files stay outside source control."""

import datetime
import hashlib
import importlib.metadata
import json
import platform
from pathlib import Path
import numpy as np


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def utc():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def encode(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)


def save(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, default=encode, allow_nan=False) + "\n")
    tmp.replace(path)


def read(path):
    return json.loads(Path(path).read_text())


def environment():
    packages = {}
    for name in [
        "numpy",
        "scipy",
        "pyscf",
        "qiskit",
        "qiskit-aer",
        "qiskit-addon-sqd",
        "qiskit-ibm-runtime",
        "ffsim",
        "quimb",
    ]:
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            pass
    return {
        "python": platform.python_version(),
        "machine": platform.machine(),
        "packages": packages,
    }


def load_model(path):
    with np.load(path, allow_pickle=False) as p:
        h, g, offset = p["h"].copy(), p["g"].copy(), float(p["offset"])
        ne = tuple(map(int, p["nelec"]))
    n = len(h)
    if h.shape != (n, n) or g.shape != (n, n, n, n) or len(ne) != 2:
        raise ValueError("Expected h[n,n], g[n,n,n,n], scalar offset and nelec[2]")
    if ne[0] != ne[1] or not 0 < ne[0] < n:
        raise ValueError(
            "This workflow supports partially occupied, spin-balanced sectors"
        )
    if not all(np.isfinite(x).all() for x in [h, g, np.array(offset)]):
        raise ValueError("Nonfinite integral")
    if not np.allclose(h, h.T, atol=1e-10):
        raise ValueError("One-electron matrix must be real symmetric")
    return h, g, offset, ne
