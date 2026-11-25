# Uniform valid pair domain

API: `cudaq_algorithms.thc_prepare.THCPairDomain(num_spatial,rank)`.
Zero-based valid pairs are `0<=mu<=nu<M` and `nu=M,0<=mu<n`.
`pair_width` fits both the sentinel M and n-1. There are
`d=M*(M+1)//2+n` valid bins. Source: `thc_prepare.py`.

## Workflow

`kernel()` and `adjoint_kernel()` take `(pair[2*k],scratch[num_scratch])`.
First k bits are mu, next k nu. Reversible inequalities/equality mark success;
an analytic success-amplitude RY plus Grover iterations prepares uniform valid
pairs with scratch clean. `amplification` reports rounds and angle. The RY
uses floating-point execution; its FT synthesis cost is separate. The n=2,M=1
predicate has a small exact specialization.

## Verification

`pytest -q tests/python/test_thc_prepare.py -k valid_pair_domain` compares all
prepared amplitudes with `1/sqrt(d)` and verifies inverse and scratch for
multiple triangular/sentinel domains. Check forward preparation as well as
roundtrip. An incompatible generic padded alias source is not this state.
