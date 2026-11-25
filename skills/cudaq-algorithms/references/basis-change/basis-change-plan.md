# Basis-change planning and angle encoding

Public APIs: `cudaq_algorithms.basis_change.fermion_basis_change`,
`GivensStructure.partial`, `GivensStructure.full`, `compile_givens_basis_change`,
`AngleEncoding`. Source: `basis_change/_factory.py`, `_plan.py`, `_encoding.py`.

## Workflow

`plan = fermion_basis_change(vector, programmable=True, angle_bits=b,
structure=GivensStructure.partial(n, mode_map=modes))` produces a fixed schedule
and finite words; it does not allocate or look up quantum words. A selected
column leaves an unconstrained completion; a full transformation specifies all
columns. Inspect `schedule`, `encoded_words`, `represented_parameters`,
`parameter_registers`, `num_system_qubits`, `num_qubits`, `quantization_error`.
Physical mode maps retain intervening Jordan-Wigner parity.

`AngleEncoding(b)` uses unsigned little-endian full-turn words, bit weight
`2*pi*2**(j-b)`, nearest-even rounding, width 1..52. Host double rounding has the
limits documented by `encode`; a standalone Pauli rotation is not automatically
periodic in 2*pi. Different lookup rows must use compatible schedules/layouts.

## Verification

Run `pytest -q tests/python/test_fermion_basis_change_plan.py` and the
planning cases in `test_fermion_basis_change*.py`. Inspect the actual test names
in the checkout. Compare the rounded schedule with the ideal operator under
`quantization_error`; compare circuit to rounded target separately. Independent
determinant tests must retain phases under control. A host plan alone is not an
executed programmable operation.
