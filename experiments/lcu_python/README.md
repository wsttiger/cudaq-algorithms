# Pure-Python PauliLCU / qubitization / QSVT — API-surface experiment

A self-contained, Python-only implementation of the Pauli LCU block-encoding
stack — block encoding, qubitization walks and moment measurement, and QSVT —
built to explore what an intuitive Python API looks like. No compiled
`cudaq-algorithms` bindings are used, only `cudaq` and `numpy`.

```
pauli_lcu_py.py                    block encoding: kernels + PauliLCU
qubitization_py.py                 walks, observables, Walk (moments)
qsvt_py.py                         PhaseSequence, QSVT, host response model
test_pauli_lcu_py.py               dense-reference tests (pytest, qpp-cpu)
test_qubitization_py.py            walk/moment/adjoint tests
test_qsvt_py.py                    response/convention/QSPPACK tests
demo.py                            printed walkthrough of the LCU surface
example_hamiltonian_simulation.py  QSVT time evolution (needs qsppack, scipy)
example_quantum_lanczos.py         full QEL: moments -> Krylov -> energy
```

Run:

```bash
PYTHONPATH=/path/to/cudaq python3 -m pytest -q .
PYTHONPATH=/path/to/cudaq python3 demo.py
PYTHONPATH=/path/to/cudaq python3 example_hamiltonian_simulation.py
PYTHONPATH=/path/to/cudaq python3 example_quantum_lanczos.py
```

The simulation target defaults to `qpp-cpu` and is overridable everywhere via
`LCU_PY_TARGET`, e.g. `LCU_PY_TARGET=nvidia-fp64` for the GPU statevector
simulator (verified: all tests and both examples pass on an RTX 6000 Ada).
Use an fp64 target for the test suite — the default `nvidia` target is fp32
and legitimately misses the 1e-8..1e-10 tolerances (the few failures there
are precision, not correctness). State construction goes
through `pauli_lcu_py.state_from`, which matches the input dtype to the
active target's precision (`cudaq.complex()`), since fp32 simulators reject
complex128 initial-state data.

## The API surface

### Block encoding (`pauli_lcu_py`)

```python
import pauli_lcu_py as lcu

# Flexible construction: dict, SpinOperator, or (coeff, word) pairs.
enc = lcu.PauliLCU({"ZI": 0.70, "IZ": -0.43, "XX": 0.19, "YZ": 0.11})
enc = lcu.PauliLCU(spin_op, num_qubits=2)
enc = lcu.PauliLCU([(0.7, "ZI"), (-0.43, "IZ")])

enc.num_system, enc.num_ancilla, enc.num_terms
enc.alpha                  # LCU 1-norm (alias: enc.normalization)
enc.terms                  # [(coeff, word), ...]
enc.constant_term          # sum of identity terms

kernel = enc.encode_kernel()          # @cudaq.kernel(state): full U_A
good = enc.good_subspace(cudaq.get_state(kernel, state))
hpsi_over_alpha = enc.action(psi)     # (H/alpha)|psi> in one call
```

### Qubitization (`qubitization_py`)

```python
import qubitization_py as qub

walk = qub.Walk(enc)
walk.kernel(power=3)                  # PREPARE + W^3 + UNPREPARE
walk.kernel(power=3, uncompute=False) # ... without UNPREPARE
walk.adjoint_kernel(power=2)          # (W dagger)^2
walk.roundtrip_kernel(power=2)        # W^2 (W dagger)^2 == identity

walk.moment(psi, k)                   # <T_k(H/alpha)> via cudaq.observe
walk.moments(psi, 8)                  # QEL even/odd observable convention

qub.reflection_observable(enc)        # 2|0..0><0..0| - I  (SpinOperator)
qub.select_observable(enc)            # sum_i sign_i |i><i| x P_i
```

### QSVT (`qsvt_py`)

```python
import qsvt_py as qsvt

seq = qsvt.PhaseSequence(phases)                      # projector convention
seq = qsvt.PhaseSequence(phases, convention="qsp")    # QSPPACK convention,
                                                      # converted automatically
seq = qsvt.PhaseSequence(phases, walk_directions=["forward", "adjoint"])

transformer = qsvt.QSVT(enc)
kernel = transformer.kernel(seq)      # @cudaq.kernel(state)
good = transformer.transform(psi, seq)

# Host model. NOTE the convention: x is the plain scaled eigenvalue
# lambda/alpha — the walk's -H/alpha sign is folded into the model, so
# transform(eigvec) == evaluate_response(seq, lambda/alpha) * eigvec
# with no caller-side negation.
value = qsvt.evaluate_response(seq, x)

evolved = qsvt.recover_real_time_evolution(cos_state, sin_state,
                                           cos_phases, sin_phases)
```

Escape hatch at every level: the module-level kernels (`lcu.prepare`,
`lcu.select`, `lcu.apply`, `lcu.reflect_about_zero`, `qub.adjoint_walk`,
`qsvt.signal_phase`, `qsvt.apply_phase_sequence`, ...) compose inside user
kernels with `enc.kernel_args` supplying the flattened arrays.

## What the experiment demonstrated

1. **No arity ladder needed.** CUDA-Q Python kernels take a whole `qview` as
   a control register (`x.ctrl(ancilla, target)`,
   `z.ctrl(reg.front(n-1), reg[n-1])`, `r1.ctrl(...)`), so every
   multi-controlled gate that needs a 10-branch if/else ladder in the C++
   device kernels is one line here — and there is no 10-ancilla cap.

2. **The factory-closure pattern works, including cross-module kernel
   calls.** Kernels defined inside factories capture the flattened arrays and
   call module-level kernels imported from other files. Users never see the
   seven-list threading; `kernel_args` remains for power users.

3. **Observable-based workflows fit naturally.** The reflection and SELECT
   observables are built as `cudaq.SpinOperator`s from the encoding metadata,
   so `Walk.moment` is a real `cudaq.observe` measurement — the same circuit
   and operator a hardware run would use — not statevector slicing.

4. **A friendlier host-model convention is possible.** `evaluate_response`
   folds the walk's `-H/alpha` sign into its 2x2 step, so users evaluate at
   the plain scaled eigenvalue instead of remembering to negate (the C++
   helper's documented `-x` gotcha). Verified against the circuits in the
   tests.

5. **Kernel-boundary quirks found along the way** (worked around; worth
   upstream reports):
   - Empty `list` kernel arguments fail with "Cannot infer runtime argument
     type" — zero-ancilla encodings and degree-0 sequences are special-cased.
   - `@dataclass` kernel arguments work, including list fields, **except** a
     `list[int]` field containing a negative value fails with
     `std::bad_cast` — the blocker for passing one aggregated kernel-args
     object instead of flat lists.
   - A control set cannot mix a bare qubit with a `qview`, and
     `cudaq.control` of a composite (kernel-calling) kernel fails — hence
     the combined-register convention for the controlled family.

6. **Correctness carried over.** Tests mirror the library methodology: dense
   Pauli-sum action match, the single-term negative-coefficient sign
   regression (the PR #4 A.1 bug), Chebyshev moments on an asymmetric
   spectrum via both observables, adjoint walks inverting forward walks, the
   QSVT good-subspace block matching the host model column by column
   (including mixed walk directions), qsp/qsvt convention equivalence, and a
   QSPPACK Hamiltonian-simulation run reaching ~1e-15 state error.

### Controlled variants

The full controlled family is implemented: `controlled_select`,
`controlled_reflect_about_zero/prepare`, `controlled_walk`/
`controlled_adjoint_walk`, `Walk.controlled_kernel` /
`controlled_roundtrip_kernel`, and `QSVT.controlled_kernel` via
`apply_controlled_phase_sequence`. One language constraint shaped the API:
a CUDA-Q Python control set cannot mix a bare qubit with a `qview`
("invalid argument type for control operand"), and `cudaq.control(...)` of a
kernel that calls other kernels fails ("Could not successfully apply kernel
specialization"). The controlled kernels therefore take a single
**combined register** whose qubit 0 is the external control and whose
remaining qubits are the ancilla/signal register — every control set is then
a view of that register. Uncontrolled PREPARE pairs wrap the controlled
SELECT, so everything collapses to the identity at control |0> (verified in
tests for walks, roundtrips, and sequences, both control states).

## Deliberate scope cuts

- `action()`/`transform()`/`good_subspace()` are simulation conveniences and
  say so; the kernel factories and observables are the hardware-shaped path.
- Phase *generation* stays external (QSPPACK), matching the library's scope
  decision.

## Relation to the real library

This prototype is the "section 3.2 + 3.4" direction from
`API_ERGONOMICS_NOTES.md` taken to its endpoint: aggregate at the host level,
flatten only inside library-owned factories. If adopted, the natural landing
spot is `python/cudaq_algorithms/` as a Python-only layer over the existing
C++ core (per the notes' design decision), with the pure-Python kernels here
replaced by calls to the already-bound device kernels — the object surfaces
(`PauliLCU`, `Walk`, `QSVT`, `PhaseSequence`) carry over unchanged. The one
semantic divergence to reconcile deliberately: `evaluate_response`'s
sign-folded convention vs the C++ `evaluate_qsvt_response(-x)` convention.
