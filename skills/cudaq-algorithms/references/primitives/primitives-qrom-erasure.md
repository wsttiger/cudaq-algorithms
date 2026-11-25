# Measurement-assisted erasure of a known lookup

Public development APIs: `QROM.measurement_eraser(block_size=None)` and
`cudaq_algorithms.primitives.QROMErase(data,address_bits,output_bits,block_size)`.
Source: `primitives/_qrom_erasure.py`, `_qrom.py`.

## Workflow

The input output-register must hold the clean lookup value `f(address)`.
`kernel()` takes `(address, output, scratch)` and erases that output with
X-basis measurements, reset and feed-forward phase correction. It preserves
coherent addresses entangled with a reference. This is not an inverse for
arbitrary XOR targets; use the self-inverse coherent QROM for those. Do not put
the measured kernel under quantum control or `cudaq.adjoint`.

Block size k is a power of two up to `2**(address_bits-1)`; omitted k minimizes
the actual correction cost. Scratch needs `k+address_bits` clean qubits.
`correction_kernel()` takes `(address,scratch,measured:list[bool])` for explicit
branch verification. Zero-pad nonpower tables. Exact correction price is
`W(high,ceil(N/k)) + 2*W(low,k)`, independent of output width, where W is the
coherent unary-iteration price. Forward lookup, measured output bits and
feed-forward are separate. This does not replace every coherent unary AND
uncompute with the paper's optimized measurement-assisted version.

## Verification

`pytest -q tests/python/test_qrom_erasure.py` covers all outcomes on a small
table, actual measured trajectories entangled with a reference and compiler
counts across output widths. Verify the corrected joint state and all restored
scratch. Merely quoting an erase formula or having an unused eraser proves no
application-level erasure saving.
