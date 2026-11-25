# Quantum-word-driven fermionic rotations

Public APIs: `fermion_basis_change(..., programmable=True, angle_bits=b)` and
`FermionBasisChange.kernel()`, `adjoint_kernel()`, `controlled_kernel(adjoint=...)`.
The separate-view `split_kernel(adjoint=...)` is a development extension.
Source: `basis_change/_factory.py`, `_kernels.py`.

## Workflow

The existing backend reads each quantum word through bit-controlled rotations.
It is programmable angle consumption; it is not phase-gradient synthesis.
`kernel()` takes the combined system/parameter layout specified by the plan;
`controlled_kernel()` puts the enable at index zero, shifting that layout by one.
`split_kernel()` takes `(system, parameters)` and respects layout gaps relative
to `num_system_qubits`. Mint kernels on the host and capture only kernels/data
inside device code. No lookup is implicit in the plan.

Compile rows with one structure, pack `encoded_words`, load with
`QROM(...).kernel(address, ladder, words)`, apply inverse basis change, middle
parity and forward change in the derived order, then unlookup. A zero angle
row still applies parity unless the middle action is disabled. Distinguish
lookup padding, one-body sentinels and signs from physical zero rotations.

## Verification

`pytest -q tests/python/test_fermion_basis_change_qrom.py
tests/python/test_fermion_basis_change_split.py` exercises lookup/consume/undo.
Change the loaded word with the host schedule fixed; test coherent addresses,
controls, both spin mode maps, a three-orbital cascade, multi-occupation and
whole-output cleanup. Use determinant-based references independent of the
rotation tape. The application must actually call the programmable consumer;
an available library API or unused QROM is not integration evidence.
