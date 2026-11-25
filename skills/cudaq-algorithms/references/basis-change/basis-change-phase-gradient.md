# Phase-gradient basis-change backend

Development API: `FermionBasisChange.phase_gradient_kernel(adjoint=False)`;
requires a programmable plan with contiguous parameter words. Source:
`basis_change/_kernels.py`, `_factory.py`.

## Workflow

Kernel signature: `(system, parameters, gradient, scratch)`. Initialize
`gradient[b]` once with [PhaseGradient](../primitives/primitives-phase-gradient.md);
reserve `b+1` clean scratch. The inverse restores phases as well as occupations.
The resource is returned unchanged on the promised catalyst sector.

The existing Givens tape is reused. Opposite-angle Rz pairs become R1 operations
whose word-dependent scalar phases cancel across the complete pair; number
phases remain R1. Each variable R1 costs `4*b-2` actual CDKM Toffolis. QROM,
resource preparation and FT synthesis are separate. This implementation does
not establish the paper's cheaper measurement-assisted adder prefactor.

Optional `parameter_slots=(...)` packs only those schedule words, in the named
order. The caller must prove every omitted encoded column is identically zero
over all selected addresses. This reduces storage while preserving quantum
selection; it is not permission to capture changing angles on the host.

`one_particle_gradient_kernel` is an explicit two-mode, one-particle restriction
on one logical qubit. Its logical basis labels occupations in mode 0/1; it is
not general-N, a full-Fock circuit, or a molecular singlet simulation.

## Verification

Run `pytest -q tests/python/test_fermion_basis_change_gradient.py
tests/python/test_one_particle_gradient.py` (confirm paths in the checkout).
Compare all complex Fock columns at feasible widths against independent
determinants, with catalyst and scratch in the joint expected output. Check
inverse, vacuum and double-occupation phases without fitting a global phase.
Arithmetic widths and compiler prices are tested in `test_phase_gradient.py`.
`test_gradient_compact.py` checks quantum-selected compact words against the
full-Fock reference and preserves vacuum/determinant phases.
