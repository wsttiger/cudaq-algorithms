# Suzuki-Trotter Hamiltonian simulation (pure Python)

Product-formula time evolution for Hamiltonians expressed as sums of Pauli
strings, implemented entirely in Python on top of CUDA-Q: term extraction,
host-side planning and ordering, resource estimation, and the circuit
primitive itself. Importing the package registers it under the
`cudaq.algorithms` namespace.

```
cudaq_algorithms/trotter.py     term extraction, plans, resources, apply_trotter kernel
cudaq_algorithms/sim_utils.py   simulation-only helpers (statevector evolution)
conftest.py                     shared pytest configuration
test_trotter.py                 dense-reference test suite
example_trotter_chemistry.py    chemistry-style end-to-end example
```

Run:

```bash
PYTHONPATH=/path/to/cudaq python3 -m pytest -q .
PYTHONPATH=/path/to/cudaq python3 example_trotter_chemistry.py
```

The simulation target defaults to `qpp-cpu`; override with CUDA-Q's
standard `CUDAQ_DEFAULT_SIMULATOR` variable (e.g. `nvidia-fp64` for the
GPU statevector simulator).

## API

```python
import cudaq_algorithms                     # registers cudaq.algorithms
from cudaq.algorithms import trotter

# Flexible Hamiltonian input: SpinOperator, single spin term,
# {"XZI...": coeff} mapping, or (coeff, word) pairs.
plan = trotter.make_trotter_plan(
    hamiltonian, time=0.8, steps=4, order=2,
    ordering=trotter.TrotterOrdering.COEFFICIENT_MAGNITUDE_DESCENDING)

plan.kernel()       # ready @cudaq.kernel(): |0...0> -> evolved state
plan.resources()    # TrotterResourceEstimate (rotations, CNOT proxy, ...)
plan.num_terms, plan.identity_coefficient, plan.words, plan.coefficients
```

Supported product-formula orders: 1 (first order), 2 (symmetric second
order), and 4 (Forest-Ruth, built from symmetric steps with time fractions
`FOREST_RUTH_W1`, `FOREST_RUTH_W0`, `FOREST_RUTH_W1`).

For composition inside a custom kernel, the flattened primitive is the
escape hatch:

```python
@cudaq.kernel
def my_kernel(coeffs: list[float], words: list[cudaq.pauli_word],
              t: float, steps: int, order: int):
    q = cudaq.qvector(4)
    # ... state preparation ...
    trotter.apply_trotter(coeffs, words, t, steps, order, q)
```

### Identity terms

For `H = c I + H'`, the circuit applies the product formula for `H'` only;
`exp(-i c t)` cannot be realized as a circuit on the evolved register. The
phase is an unobservable global phase for a single unconditioned evolution
but a real relative phase for controlled or interference-based algorithms —
`plan.identity_coefficient` reports it so callers can account for it.

### Simulation helpers (`sim_utils`)

Statevector-based conveniences live in `cudaq.algorithms.sim_utils`,
clearly separated from the hardware-shaped API (nothing in the library
classes calls `cudaq.get_state`):

```python
from cudaq.algorithms import sim_utils

evolved = sim_utils.evolve(plan, ket)   # approximates exp(-i H t)|ket>,
                                        # identity phase included
```

## Testing

The suite pins correctness against independent dense references: exact
matrix exponentials via diagonalization, and an explicit Pauli-rotation
simulator for per-order product formulas. Coverage includes kernel
interop with flattened arguments, invalid-input no-op behavior, per-order
error thresholds, asymptotic error-scaling slope fits (order-p error ~
dt^p), exactness for commuting Hamiltonians, and every accepted input
form of the term-extraction front end.

## Known CUDA-Q Python constraints

Two upstream compiler behaviors shape the implementation:

- `return` inside a Python kernel is silently ignored
  ([cuda-quantum#4845](https://github.com/NVIDIA/cuda-quantum/issues/4845));
  the `apply_trotter` body is a single positively-guarded block instead of
  early-return guards.
- Captured empty lists cannot be marshaled
  ([cuda-quantum#4847](https://github.com/NVIDIA/cuda-quantum/issues/4847));
  identity-only plans special-case their kernel factory.
