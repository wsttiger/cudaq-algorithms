# Sine-window phase control with shared unary reflections

Development API: `cudaq_algorithms.UnaryPhaseEstimation(encoding,queries)`.
Source: `python/cudaq_algorithms/unary_phase_estimation.py`.

## Workflow

For q>=1, the shared unary schedule uses q Hermitian SELECT calls to choose
`W**(2*t-q)`, t=0..q. One additional controlled W on parity b produces consecutive
powers `W**(2*t+b-q)`. Omitting parity aliases opposite signed energies.
The optimal sine amplitudes cover `L=2*(q+1)` states, padded with zero for IQFT;
q need not be a power of two. A global W**(-q) changes no measured histogram.

`control_kernel()` prepares the sine window. `multiplex_kernel()` takes
`(system,work)` with phase, unary and encoding registers. `kernel(state_prep=...,
measure=True)` builds the full circuit; `energies_from_counts` uses the same
signed walk decoder as textbook QPE. `.resources` exposes query counts,
unary cost and register sizes. Prepared-frame encodings expose `select_kernel`
and `controlled_reflection_kernel`; generic encodings use the full Hermitian
block and a zero-frame reflection, adding preparation work.

For measurement/feed-forward encodings on CUDA-Q 0.17, use
`kernel(state_prep=prep,measure=True,return_measurements=True)` with `cudaq.run`.
It returns the phase bits explicitly per shot. `cudaq.sample` rejects kernels
that branch on measurement results; the coherent path can use the named
`phase` register from a normal measured kernel.

The sine-state tree uses floating-point RY rotations; FT synthesis is separate.
Unary ladders use actual coherent uncompute prices. Extra parity costs one
controlled walk. These deviations must remain visible in resource claims.

## Verification

`pytest -q tests/python/test_unary_phase_estimation.py` compares signed physical
spectra and independently computed sine-window Fourier distributions, including
nonpower q, endpoints and padded control support. Check actual selected powers
and coherent branches when extending the schedule. Library IQFT omits reversal
swaps; this protocol reverses its ordinary little-endian input before IQFT.
