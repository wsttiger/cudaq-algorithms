# Contiguous pair index

API: `cudaq_algorithms.thc_prepare.TriangularIndex(pair_width,output_width)`.
Source: `thc_prepare.py`. `kernel()` takes `(pair[2*k],output[m],scratch)` and
XORs `(nu*(nu+1)//2+mu) mod 2**m` into any initial output. Width m<=2k.

## Workflow

Use the zero-based index consistently with `THCPairDomain.pairs`; first
triangular pairs, then one-body sentinel bins. Generic schoolbook multiply,
shift and add use O(k**2) gates and `5*k+4` scratch, reversed after copying the
index. k=1 specializes to two CNOTs and one Toffoli with no scratch.

## Verification

`pytest -q tests/python/test_thc_prepare.py -k triangular_index` uses coherent
inputs and nonzero XOR targets, checks classical arithmetic and full scratch
cleanup, then applies the index twice. A basis-only zero-target check would
miss part of this contract.
