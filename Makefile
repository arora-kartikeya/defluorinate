PYTHON ?= python
export PYTHONPATH := src
export OMP_NUM_THREADS := 2
export OPENBLAS_NUM_THREADS := 2
export MKL_NUM_THREADS := 2
export VECLIB_MAXIMUM_THREADS := 2
export XLA_PYTHON_CLIENT_PREALLOCATE := false
export MPLCONFIGDIR := $(CURDIR)/output/.matplotlib

.PHONY: test demo

test:
	$(PYTHON) -m pytest -q

demo:
	$(PYTHON) -m defluorinate demo --out output/h2 --shots 512 --resume
