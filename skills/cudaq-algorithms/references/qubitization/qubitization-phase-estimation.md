# Textbook binary quantum phase estimation

Public baseline API: `cudaq_algorithms.PhaseEstimation(encoding)`.
Source: `python/cudaq_algorithms/phase_estimation.py`.

## Workflow

`kernel(num_bits,state_prep=None,measure=False)` constructs a binary
controlled-walk ladder and inverse QFT. With `state_prep`, it is a zero-argument
gate-built circuit; without it, supply a `cudaq.State`. Measure=True measures
the phase register only. `energies_from_counts(counts,num_bits,top=...)` folds
mirror bins. CUDA-Q strings list bit zero first, so index=`int(s[::-1],2)`.
Energy is `-alpha*cos(2*pi*j/2**m)`; restore separate scalar shifts afterwards.
Input overlap determines spectral weights, not automatic ground-state selection.

Cost is `2**m-1` controlled walk calls. It is packaged, but does not implement
the paper's optimized unary reflection schedule. Large ladders require the
documented simulation stack helper; inspect the source before executing them.

## Verification

`pytest -q tests/python/test_phase_estimation.py` compares phase marginals and
signed decoding against independent small spectral/Fourier references. Never
select a bin using a reference eigenvalue. Phase resolution and shot confidence
are separate, and no accuracy finer than the chosen grid is implied.
