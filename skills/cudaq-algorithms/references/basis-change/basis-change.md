# Fermionic basis changes

## Selectable operation contracts

| Operation + object | Public entry point | Kind / layer | Capabilities | Record |
| --- | --- | --- | --- | --- |
| compile / a full or selected-column orbital transformation | `fermion_basis_change`, `GivensStructure`, `compile_givens_basis_change` | host plan | physical mode maps; fixed parameter structure | [planning and encoding](basis-change-plan.md) |
| apply / a basis change driven by quantum angle words | `FermionBasisChange.kernel`, `split_kernel`, `controlled_kernel` | device kernel factory | programmable words, inverse, full-Fock phases | [programmable consumer](basis-change-programmable.md) |
| apply / word-driven rotations using catalytic arithmetic | `FermionBasisChange.phase_gradient_kernel` | device kernel factory | clean arithmetic scratch and reusable gradient | [phase-gradient backend](basis-change-phase-gradient.md) |

## Workflow

Import `cudaq_algorithms.basis_change` directly. A Slater preparation and a
full-Fock basis transformation have different contracts. Compile one compatible
structure for all coherently selected rows; load encoded words with QROM and
consume them in the programmable backend. See [THC](../thc/thc-fidelity.md) only
when that paper is requested.

## Verification

Use independent exterior-power/determinant matrices, not only one-particle
agreement. Include the vacuum, multiple occupation, spectators, coherent
controls, exact number phases and scratch restoration. Source is
`python/cudaq_algorithms/basis_change/`; tests are `test_fermion_basis_change*.py`.
