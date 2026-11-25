# Alias PREPARE over valid THC pairs

API: `cudaq_algorithms.thc_prepare.THCPairPrepare(one_body,z,keep_bits=2,
qrom_variant='select',block_size=None)`. Source: `thc_prepare.py`.

## Workflow

Weights are Z_mm/2 on the diagonal, Z_mn off diagonal, and signed one-body
coefficients on the sentinel. Host alias tables use d valid bins, not 2**m.
`kernel()` / `adjoint_kernel()` take `(pair,garbage,scratch)`. Garbage retains
alternate pair, keep, sign, alternate sign, coin and comparison flag;
`num_garbage=2*k+2*b+3`. Scratch `num_scratch` is clean at the boundary.
`table_probabilities` is the realized finite-bit marginal, `alpha` the exact
weight one-norm. `discretization_bound` is per-bin absolute error.

`uniform_nonnegative` permits a uniform-source specialization only when the
realized alias probabilities are exactly uniform and all weights nonnegative.
The signed/history path remains available. SELECT must use the selected sign
and both pair/order branches, with the one-body exception. Generic
`AliasSamplingPrepare` starts from power-of-two bins and cannot be substituted
without deriving the changed domain and tables.

## Verification

`pytest -q tests/python/test_thc_prepare.py` enumerates coherent alias histories,
signs and rounded tables; checks inverse and scratch. Additional application
tests must establish diagonal/off-diagonal multiplicity and sentinel action.
Histories retained by contract are not scratch-cleanup failures.
