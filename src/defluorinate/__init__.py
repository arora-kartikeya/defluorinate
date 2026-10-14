"""Small-active-space chemistry and Qiskit SQD experiments."""

import os

for _name in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
):
    os.environ[_name] = "2"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
