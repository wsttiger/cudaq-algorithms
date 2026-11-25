# Catalytic variable phase

Public development API: `cudaq_algorithms.primitives.PhaseGradient(b)`.
Source: `python/cudaq_algorithms/primitives/_phase_gradient.py`.

## Workflow

`prepare_kernel()` / `unprepare_kernel()` act on `gradient[b]` and prepare
`2**(-b/2) sum_j exp(-2*pi*i*j/2**b)|j>`. `phase_kernel(adjoint=False)` takes
`(word[b], target:qubit, gradient[b], scratch[b+1])`. Conditional masking plus
modular add/subtract kicks back `exp(2*pi*i*word/2**b)` on target=1. The word,
gradient and clean scratch are restored. Views must be disjoint. The resource
must be prepared; the kernel has no such promise for an arbitrary catalyst.

Cost is `4*b-2` logical Toffolis per variable phase (tested from the emitted
circuit), not the paper's optimized adder price. Resource preparation has b H
and b specified dyadic R1 rotations, once per reusable resource; approximate
FT synthesis of these rotations is a separate declared assumption.

## Verification

`pytest -q tests/python/test_phase_gradient.py` checks coherent word branches,
signed inverse, joint catalyst restoration and compiler counts at multiple
widths. Compare circuit versus finite-word phase exactly, and finite-word versus
ideal target under a separately derived rounding bound. Never attribute this
cost to the distinct bit-controlled-rotation backend.
