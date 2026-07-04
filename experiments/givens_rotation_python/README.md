# Pure-Python Givens-rotation Slater determinant prep — API-surface experiment

A self-contained, Python-only replication of the
`given_rotation_state_prep_phase2` branch: Givens elimination schedules (real
and complex), plan construction/validation, resource estimation, and the
state-preparation kernels — no compiled `cudaq-algorithms` bindings, only
`cudaq` and `numpy`. Companion to the `experiments/lcu_python` and
`experiments/suzuki_trotter_python` prototypes (same philosophy).

```
givens_py.py                    the module: kernels + schedules + plan + resources
test_givens_py.py               ports every test from tests/python/test_givens_stateprep.py
example_slater_determinant.py   port of examples/stateprep/givens_slater_determinant.py
```

Run:

```bash
PYTHONPATH=/path/to/cudaq python3 -m pytest -q .
PYTHONPATH=/path/to/cudaq python3 example_slater_determinant.py
```

Target defaults to `qpp-cpu`; override with `LCU_PY_TARGET` (e.g.
`nvidia-fp64`).

## Functionality parity with the C++/bindings branch

| given_rotation_state_prep_phase2 | here |
|---|---|
| `apply_givens_rotation` / `apply_phase_givens_rotation` kernels | ✅ upstream signatures, same sign convention (inlined `givens_rotation(-theta)`), non-adjacent pairs no-op |
| `prepare_slater_determinant` / `prepare_complex_slater_determinant` | ✅ upstream signatures, invalid flattened inputs no-op |
| real + complex `make_givens_rotation_schedule` (elimination order, phase extraction, final phases) | ✅ one implementation, real/complex branch per upstream |
| automatic real/complex dispatch (numpy dtype kind, nested-list complex entries) | ✅ |
| orthonormality validation (normalized / orthogonal / rectangular, 100x tolerance) | ✅ |
| `SlaterDeterminantPlan` + `validate_slater_determinant_plan` (adjacency, range, phase-shape rules) | ✅ mutable dataclass, default-constructible like the binding type |
| both resource estimators | ✅ same formulas |
| Python test suite (17 tests) + example | all ported |

Prototype ergonomics on top:

```python
import givens_py as stateprep

plan = stateprep.make_slater_determinant_plan(occupied_orbitals)  # numpy or lists
plan.kernel()      # ready @cudaq.kernel() — real/complex chosen automatically
plan.state()       # one-call simulated statevector
plan.resources()   # GivensStatePrepResources
```

The module-level kernels remain the escape hatch for composing inside user
kernels (e.g. state prep before time evolution), exactly as upstream.

## Kernel-language findings (new in this experiment)

1. **Python `exp_pauli` does not accept individual qubit operands** — the
   C++ two-qubit form `exp_pauli(theta, "YX", q[a], q[b])` fails with "too
   many values"; only `(angle, register, word)` is supported.
2. **Runtime-contiguous `qview` slices work**: `qubits[a:a + 2]` with a
   runtime `a` is a valid register for `exp_pauli`. Because this library
   only ever emits *adjacent* Givens rotations (validated on the host), the
   two-qubit form maps onto slices exactly, preserving the upstream kernel
   signatures. A future non-adjacent variant would need host-side
   full-width Pauli words instead.
3. The early-return miscompile (see the Suzuki-Trotter prototype README /
   upstream issue draft 4) applies here too: the no-op guards in the
   preparation kernels are positive if-blocks.

## Deliberate scope cuts (matching the upstream branch)

- No integration with the UCCSD/operator-pool stateprep surface the C++
  branch's `stateprep/__init__.py` also re-exports — those are separate
  features; this prototype is the Givens/Slater portion only.
- Basis rotations beyond Slater determinants (e.g. general orbital
  rotations applied to correlated states) are out of scope, as upstream.
