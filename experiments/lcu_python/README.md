# Pure-Python PauliLCU / qubitization / QSVT

A self-contained, Python-only implementation of the Pauli LCU block-encoding
stack — block encoding, qubitization walks and moment measurement, and QSVT,
each with controlled variants — used to evaluate the Python API surface for
these algorithms. Only `cudaq` and `numpy` are required; the compiled
`cudaq-algorithms` bindings are not used.

```
cudaq_algorithms/                  the package; importing it registers
                                   the cudaq.algorithms namespace
  pauli_lcu.py                     block encoding: kernels + PauliLCU
  qubitization.py                  walks, observables, Walk (moments)
  qsvt.py                          PhaseSequence, QSVT
sim_utils.py                       simulation-only helpers (tests/examples)
conftest.py                        shared pytest target fixture
test_pauli_lcu.py                  dense-reference tests (pytest, qpp-cpu)
test_qubitization.py               walk/moment/adjoint/controlled tests
test_qsvt.py                       response/convention/QSPPACK tests
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
CUDA-Q's standard `CUDAQ_DEFAULT_SIMULATOR` variable, e.g.
`CUDAQ_DEFAULT_SIMULATOR=nvidia-fp64` for the GPU statevector simulator
(verified: all tests and both examples pass on an RTX 6000 Ada). Use an fp64
target for the test suite — the `nvidia` target is fp32 and misses the
1e-8..1e-10 tolerances (the failures are precision, not correctness). State
construction goes through `state_from`, which matches the input dtype to the
active target's precision (`cudaq.complex()`), since fp32 simulators reject
complex128 initial-state data.

## The API surface

Importing `cudaq_algorithms` registers the package as `cudaq.algorithms`:

```python
import cudaq
import cudaq_algorithms  # registers cudaq.algorithms
from cudaq.algorithms import PauliLCU, Walk, QSVT, PhaseSequence
```

### Block encoding

```python
# Flexible construction: dict, SpinOperator, or (coeff, word) pairs.
enc = PauliLCU({"ZI": 0.70, "IZ": -0.43, "XX": 0.19, "YZ": 0.11})
enc = PauliLCU(spin_op, num_qubits=2)
enc = PauliLCU([(0.7, "ZI"), (-0.43, "IZ")])

enc.num_system, enc.num_ancilla, enc.num_terms
enc.alpha                  # LCU 1-norm (alias: enc.normalization)
enc.terms                  # [(coeff, word), ...]
enc.constant_term          # sum of identity terms

kernel = enc.encode_kernel()          # @cudaq.kernel(state): full U_A
```

### Qubitization

```python
walk = Walk(enc)
walk.kernel(power=3)                  # PREPARE + W^3 + UNPREPARE
walk.kernel(power=3, uncompute=False) # ... without UNPREPARE
walk.adjoint_kernel(power=2)          # (W dagger)^2
walk.roundtrip_kernel(power=2)        # W^2 (W dagger)^2 == identity
walk.controlled_kernel(power=2)       # controlled walks (see below)

walk.moment(psi, k)                   # <T_k(H/alpha)> via cudaq.observe
walk.moments(psi, 8)                  # QEL even/odd observable convention

from cudaq.algorithms import reflection_observable, select_observable
reflection_observable(enc)            # 2|0..0><0..0| - I  (SpinOperator)
select_observable(enc)                # sum_i sign_i |i><i| x P_i
```

### QSVT

```python
seq = PhaseSequence(phases)                      # projector convention
seq = PhaseSequence(phases, convention="qsp")    # QSPPACK convention,
                                                 # converted automatically
seq = PhaseSequence(phases, walk_directions=["forward", "adjoint"])

transformer = QSVT(enc)
kernel = transformer.kernel(seq)                 # @cudaq.kernel(state)
controlled = transformer.controlled_kernel(seq)  # controlled sequence

from cudaq.algorithms import recover_real_time_evolution
evolved = recover_real_time_evolution(cos_state, sin_state,
                                      cos_phases, sin_phases)
```

Sign convention: the walk block encodes `-H/alpha`, and the circuits fold
the sign in, so on an eigenstate with eigenvalue `lambda` the good-subspace
block implements `p(lambda / alpha)` — no caller-side negation. The 2x2
signal-model reference implementing this convention lives in `test_qsvt.py`
(`reference_response`), where it serves as the test oracle.

### Simulation helpers (`sim_utils` — tests/examples only)

`cudaq.get_state` is a simulator-only API, so nothing in the library package
calls it. Statevector-based conveniences live in `sim_utils` and are
imported only by the tests, demo, and examples:

```python
import sim_utils as sim

good = sim.good_subspace(enc, state)     # postselect the |0..0>-ancilla block
hpsi = sim.action(enc, psi)              # (H/alpha)|psi>
out = sim.transform(transformer, psi, seq)
psi0 = sim.state_from(ket)               # precision-aware cudaq.State
```

(`Walk.moment`/`moments` stay in the library: they measure through
`cudaq.observe`, which is a hardware-legitimate path.)

Escape hatch at every level: the module-level kernels (`prepare`, `select`,
`apply`, `reflect_about_zero`, `adjoint_walk`, `signal_phase`,
`apply_phase_sequence`, ...) compose inside user kernels, with
`enc.kernel_args` supplying the flattened arrays they take as arguments.

## Implementation notes

1. Multi-controlled gates use CUDA-Q Python's variadic control support
   (`x.ctrl(ancilla, target)`, `z.ctrl(reg.front(n-1), reg[n-1])`), so there
   is no ancilla-count cap.

2. Kernels defined inside factory methods capture the flattened arrays and
   call module-level kernels imported from other modules; `kernel_args`
   remains available for composing inside user kernels.

3. `Walk.moment` is an observable-based `cudaq.observe` measurement — the
   same circuit and operator a hardware run would use — not statevector
   slicing.

4. Controlled kernels use a combined-register convention: a CUDA-Q Python
   control set cannot mix a bare qubit with a `qview` ("invalid argument
   type for control operand"), and `cudaq.control(...)` of a kernel that
   calls other kernels fails ("Could not successfully apply kernel
   specialization"). The controlled kernels therefore take a single register
   whose qubit 0 is the external control and whose remaining qubits are the
   ancilla/signal register — every control set is then a view of that
   register. Uncontrolled PREPARE pairs wrap the controlled SELECT, so
   everything collapses to the identity at control |0> (verified in tests
   for walks, roundtrips, and sequences, both control states).

5. Kernel-boundary limitations worked around (see `UPSTREAM_ISSUES.md` for
   the full list of filed issues):
   - Empty `list` kernel arguments fail with "Cannot infer runtime argument
     type" — zero-ancilla encodings and degree-0 sequences are special-cased.
   - A `@dataclass` kernel argument with a `list[int]` field containing a
     negative value fails with `std::bad_cast` — this blocks passing one
     aggregated kernel-args object instead of flat lists.

6. Test methodology: dense Pauli-sum action match, the single-term
   negative-coefficient sign regression, Chebyshev moments on an asymmetric
   spectrum via both observables, adjoint walks inverting forward walks,
   controlled walks/sequences against their uncontrolled references at both
   control states, the QSVT good-subspace block matching the 2x2 signal
   model column by column (including mixed walk directions), qsp/qsvt
   convention equivalence, and a QSPPACK Hamiltonian-simulation run reaching
   ~1e-15 state error.

## Scope

- Simulation conveniences (`action`/`transform`/`good_subspace`) live in
  `sim_utils`, outside the library package; the kernel factories and
  observables are the hardware-shaped path.
- Phase *generation* stays external (QSPPACK).

## Relation to the library

The natural landing spot for this surface is `python/cudaq_algorithms/` as a
Python layer over the existing C++ core, with the pure-Python kernels here
replaced by calls to the bound device kernels — the object surfaces
(`PauliLCU`, `Walk`, `QSVT`, `PhaseSequence`) carry over unchanged, exposed
under the `cudaq.algorithms` namespace. One semantic divergence to reconcile
deliberately: this implementation folds the walk's `-H/alpha` sign into its
response convention, whereas the C++ host helper `evaluate_qsvt_response`
expects the caller to negate.
