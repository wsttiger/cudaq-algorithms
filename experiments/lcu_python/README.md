# Pure-Python PauliLCU — API-surface experiment

A self-contained, Python-only implementation of the Pauli LCU block-encoding
feature, built to explore what an intuitive Python API looks like. No compiled
`cudaq-algorithms` bindings are used — only `cudaq` and `numpy`.

```
pauli_lcu_py.py       the prototype module
test_pauli_lcu_py.py  dense-reference correctness tests (pytest, qpp-cpu)
demo.py               printed walkthrough of the API surface
```

Run:

```bash
PYTHONPATH=/path/to/cudaq python3 -m pytest -q test_pauli_lcu_py.py
PYTHONPATH=/path/to/cudaq python3 demo.py
```

## The API surface

```python
import pauli_lcu_py as lcu

# Flexible construction: dict, SpinOperator, or (coeff, word) pairs.
enc = lcu.PauliLCU({"ZI": 0.70, "IZ": -0.43, "XX": 0.19, "YZ": 0.11})
enc = lcu.PauliLCU(spin_op, num_qubits=2)
enc = lcu.PauliLCU([(0.7, "ZI"), (-0.43, "IZ")])

# Inspection
enc.num_system, enc.num_ancilla, enc.num_terms
enc.alpha                  # LCU 1-norm (alias: enc.normalization)
enc.terms                  # [(coeff, word), ...]
enc.constant_term          # sum of identity terms

# Kernel factories — no flattening in user code
kernel = enc.encode_kernel()          # @cudaq.kernel(state): full U_A
walked = enc.walk_kernel(power=3)     # PREPARE + W^3 + UNPREPARE

# Simulation conveniences
good = enc.good_subspace(cudaq.get_state(kernel, state))
hpsi_over_alpha = enc.action(psi)     # (H/alpha)|psi> in one call

# Escape hatch — compose the module-level kernels in your own kernel
angles, controls, ops, lengths, signs = enc.kernel_args
lcu.prepare(...); lcu.select(...); lcu.unprepare(...)
lcu.apply(...); lcu.reflect_about_zero(...); lcu.walk(...)
```

## What the experiment demonstrated

1. **No arity ladder needed.** CUDA-Q Python kernels take a whole `qview` as a
   control register (`x.ctrl(ancilla, target)`,
   `z.ctrl(reg.front(n-1), reg[n-1])`), so every multi-controlled gate that
   needs a 10-branch if/else ladder in the C++ device kernels is one line
   here — and there is no 10-ancilla cap.

2. **The factory-closure pattern works.** Kernels defined inside a factory can
   capture the flattened arrays *and* call module-level kernels. Users never
   see the seven-list threading; `kernel_args` remains for power users who
   compose inside their own kernels (demo section 6 shows both paths agreeing
   bit for bit).

3. **Kernel-boundary quirks found along the way** (all worked around, all
   worth upstream reports):
   - Empty `list` kernel arguments fail with "Cannot infer runtime argument
     type" — with zero ancillas the flattened control data is empty, so the
     factories special-case the single-term encoding.
   - (From the earlier struct probe:) `@dataclass` kernel arguments work,
     including list fields, **except** a `list[int]` field containing a
     negative value fails with `std::bad_cast` — the blocker for passing one
     aggregated kernel-args object instead of flat lists.

4. **Correctness carried over.** The tests mirror the C++/binding test
   methodology: dense Pauli-sum action match, spin-op/dict input equivalence,
   the single-term negative-coefficient sign regression (the PR #4 A.1 bug),
   identity-term policy, and qubitization Chebyshev moments
   `<T_2k(H/alpha)>` on an asymmetric spectrum.

## Deliberate scope cuts

- No QSVT phase-sequence layer (that composition belongs to the existing
  `qsvt` module; this experiment is the block-encoding core plus walks).
- No controlled variants (`controlled_select` etc.) — nothing here prevents
  them; the variadic-`ctrl` pattern extends directly.
- `action()`/`good_subspace()` are simulation-only conveniences and say so.

## Relation to the real library

This prototype is the "section 3.2 + 3.4" direction from
`API_ERGONOMICS_NOTES.md` taken to its endpoint: aggregate at the host level,
flatten only inside library-owned factories. If adopted, the natural landing
spot is `python/cudaq_algorithms/` as a Python-only layer over the existing
C++ core (per the notes' design decision), with the pure-Python kernels here
replaced by calls to the already-bound device kernels — the factories and the
`PauliLCU` construction/inspection surface carry over unchanged.
