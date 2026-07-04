# Pure-Python Suzuki-Trotter — API-surface experiment

A self-contained, Python-only replication of the `add_suzuki_trotter` branch:
term extraction, planning/ordering, resource estimation, and the
product-formula circuit itself, with no compiled `cudaq-algorithms` bindings —
only `cudaq` and `numpy`. Companion to the `experiments/lcu_python` prototype
on the `expt_lcu_python` branch (same philosophy, same conventions).

```
trotter_py.py                  the module: kernel + terms + plan + resources
test_trotter_py.py             ports every test from tests/python/test_trotter.py
example_trotter_chemistry.py   port of examples/hamiltonian_simulation/trotter_chemistry.py
```

Run:

```bash
PYTHONPATH=/path/to/cudaq python3 -m pytest -q .
PYTHONPATH=/path/to/cudaq python3 example_trotter_chemistry.py
```

The simulation target defaults to `qpp-cpu`; override with `LCU_PY_TARGET`
(e.g. `nvidia-fp64`). Initial-state construction goes through `state_from`,
which matches the input dtype to the active target's precision.

## Functionality parity with the C++/bindings branch

| add_suzuki_trotter | here |
|---|---|
| `apply_trotter` device kernel (orders 1, 2, 4; invalid inputs no-op) | `trotter_py.apply_trotter`, identical signature |
| `make_trotter_terms` (identity split out, imaginary rejection, padding) | ✅ |
| `make_trotter_plan` / `TrotterPlan` / `TrotterOrdering` | ✅ same fields + validation |
| `estimate_trotter_resources` (plan or flattened) | ✅ same formulas |
| Python test suite (14 tests: interop, no-ops, order thresholds, slope fit, commuting exactness, 4-qubit) | all ported |
| `trotter_chemistry.py` example | ported |

Prototype-style ergonomics on top:

```python
import trotter_py as trotter

plan = trotter.make_trotter_plan(
    {"XI": 0.7, "IZ": 0.4, "II": -0.2},   # dict / SpinOperator / (coeff, word) pairs
    time=0.8, steps=4, order=2,
    ordering=trotter.TrotterOrdering.COEFFICIENT_MAGNITUDE_DESCENDING)

plan.kernel()       # ready @cudaq.kernel(state) — no argument threading
plan.evolve(psi)    # one-call simulation, identity phase INCLUDED by default
plan.resources()    # TrotterResourceEstimate
```

`plan.evolve` can do something the circuit primitive cannot: reintroduce the
identity phase `exp(-i * identity_coefficient * t)` host-side, so its output
approximates the full `exp(-i H t)|psi>` and comparisons against exact
evolution need no phase alignment.

Design choice: flattened `words` are plain **strings** host-side (readable,
comparable, and accepted directly as `list[cudaq.pauli_word]` kernel
arguments); `plan.kernel()` converts to `cudaq.pauli_word` only for capture,
where plain strings cannot be lowered.

## Kernel-language findings (new in this experiment)

1. **`return` inside a Python `@cudaq.kernel` is silently ignored** — gates
   after an `if cond: return` execute regardless of the condition. This is
   the most serious AST-bridge finding so far: the C++-mirroring early-return
   guards compiled and *appeared* to work because zero-trip loops masked
   them, until the unsupported-order guard produced a wrong circuit instead
   of a no-op. `apply_trotter`'s body is one positively-guarded if-block as a
   result. (The `if n == 0: return` guards in the `lcu_python` prototype
   kernels are the same latent hazard — unreachable there only because the
   factories special-case those paths.)
2. `exp_pauli` works in Python kernels with `pauli_word` lists as arguments
   and as factory captures; plain strings work as arguments but cannot be
   *captured* ("Cannot handle conversion of python type <class 'str'>").
3. `cudaq.pauli_word` is opaque in Python (`str()` returns the repr), while
   `SpinOperatorTerm.get_pauli_word()` returns a plain `str` — hence the
   strings-host-side design choice.

## Deliberate scope cuts (matching the upstream branch)

- No controlled Trotter evolution — the upstream docs list it as future work
  (needed for phase estimation, Hadamard tests, Krylov moments). The
  combined-register pattern from the LCU prototype's controlled family would
  extend here directly.
- Phase/ordering optimizations beyond coefficient-magnitude sorting are out
  of scope, as upstream.
