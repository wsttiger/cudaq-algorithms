# Pure-Python Hartree-Fock + fixed-parameter UCC — API-surface experiment

A self-contained, Python-only replication of the
`add_hf_fixed_param_ucc_state_prep` branch — no compiled `cudaq-algorithms`
bindings, only `cudaq` and `numpy`. Companion to the other
`experiments/*_python` prototypes (same philosophy and conventions).

```
hf_ucc_py.py           kernels + HF helpers + operator pools + UCC plans
test_hf_ucc_py.py      ports the upstream test suite + pool/plan coverage
example_hf_ucc.py      fixed-parameter UCCSD prep vs dense reference
```

Run:

```bash
PYTHONPATH=/path/to/cudaq python3 -m pytest -q .
PYTHONPATH=/path/to/cudaq python3 example_hf_ucc.py
```

Target defaults to `qpp-cpu`; override with `LCU_PY_TARGET` (e.g.
`nvidia-fp64`).

## Functionality parity with the C++/bindings branch

| add_hf_fixed_param_ucc_state_prep | here |
|---|---|
| `hartree_fock` / `hartree_fock_occupation` kernels | ✅ upstream signatures |
| `make_hartree_fock_occupation` (closed shell + open-shell interleaved alpha/beta) | ✅ incl. the open-shell {0,1,2,4} fix the branch exists for |
| occupation validation + both resource estimators | ✅ |
| `fixed_parameter_ucc` kernel (nested word/coefficient groups, one parameter per excitation) | ✅ upstream nested-list signature |
| `FixedParameterUccPlan` + validation + both generic constructors | ✅ |
| `make_fixed_parameter_uccsd/uccgsd/upccgsd_plan` | ✅ |
| resource estimator | ✅ same formulas |
| upstream Python test suite | all ported (dense operator-exponential reference included) |

Because the plan constructors depend on the library's operator pools, the
needed subset is ported too (built with `cudaq.spin` algebra):
`get_uccsd_excitations`, `make_uccsd_operator_pool`,
`make_uccgsd_operator_pool` / `get_uccgsd_pauli_lists`,
`make_upccgsd_operator_pool` / `get_upccgsd_pauli_lists`. All pools are
checked densely for Hermitian generators, and the UCCSD kernel path is
validated against dense matrix exponentials of the pool — an independent
reference, not another cudaq kernel.

Prototype ergonomics on top:

```python
import hf_ucc_py as stateprep

plan = stateprep.make_fixed_parameter_uccsd_plan(4, 2, [0.11, -0.04, 0.28])
plan.kernel(num_electrons=2)                  # HF reference + UCC product
plan.kernel(occupied_orbitals=[0, 1, 2, 4])   # open-shell reference
plan.state(num_electrons=2)                   # one-call statevector
plan.resources()
```

## Kernel-language findings (new in this experiment)

1. **Nested lists work as kernel arguments** — `list[list[cudaq.pauli_word]]`
   and `list[list[float]]` marshal correctly, so the upstream
   `fixed_parameter_ucc` signature ports unchanged.
2. **Nested lists cannot be captured** (`getElementType(): incompatible
   function arguments`) — the plan factory therefore flattens the grouped
   data (theta*coefficient per rotation) into flat lists before capture.

## Deliberate scope cuts

- The CEO operator pool and the variational `uccsd`/`uccgsd`/`upccgsd`
  ansatz kernels from `main` are not ported — the branch under replication
  only *consumes* the pools listed above; this prototype is the HF +
  fixed-parameter feature, not the whole stateprep subsystem.
- No parameter optimization, as upstream (the primitive is deliberately
  non-variational).
