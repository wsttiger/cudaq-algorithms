# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unary-iteration QROM: coherent classical-table lookup.

``QROM(data, address_bits, output_bits)`` mints the Babbush-style
(`arXiv:1805.03662`) lookup ``|k>|y> -> |k>|y XOR data[k]>``: a unary
iteration over the address register whose per-address body CNOTs the set
bits of ``data[k]`` into the output register. Cost: at most
``2 (len(data) - 1)`` Toffolis and ``address_bits`` clean ladder
ancillas. Addresses ``k >= len(data)`` read as zero (the iteration never
enters them).

The write phase is X-only, so the lookup is exactly its own inverse
(``U^2 = I``, XOR-ing the same table twice); there is no separate adjoint
kernel — apply ``kernel()`` again to uncompute (the property is pinned by
``tests/python/test_sparse_qrom.py``).

This is the plain unary-iteration QROM; a SELECT-SWAP (``QROAM``)
variant can later slot in behind the same ``kernel()`` /
register-accounting surface without touching consumers.
"""

from __future__ import annotations

from collections.abc import Sequence

from ._unary_iteration import unary_iteration_kernels

__all__ = ["QROM"]


class QROM:
    """Coherent lookup of a classical integer table (see module docstring).

    Parameters
    ----------
    data
        The table: non-negative integers, one per address, each fitting
        in ``output_bits`` bits.
    address_bits
        Address register width (``len(data) <= 2^address_bits``).
    output_bits
        Output register width.

    The minted kernel has signature ``(address: qview, ladder: qview,
    output: qview)`` — all little-endian, ``ladder`` being
    ``num_ladder = address_bits`` clean ancillas (|0> in, |0> out). The
    three views may live anywhere (no contiguity requirement between
    them), which is what lets callers weave the QROM into their own
    ancilla layouts.
    """

    def __init__(self, data: Sequence[int], address_bits: int,
                 output_bits: int) -> None:
        if int(address_bits) != address_bits or address_bits < 1:
            raise ValueError("address_bits must be a positive integer")
        if int(output_bits) != output_bits or output_bits < 1:
            raise ValueError("output_bits must be a positive integer")
        address_bits = int(address_bits)
        output_bits = int(output_bits)
        entries = [int(v) for v in data]
        if len(entries) == 0:
            raise ValueError("data must be non-empty")
        if any(int(v) != v for v in data):
            raise ValueError("data entries must be integers")
        if len(entries) > (1 << address_bits):
            raise ValueError(
                f"data has {len(entries)} entries but address_bits="
                f"{address_bits} addresses only {1 << address_bits}")
        if any(v < 0 for v in entries):
            raise ValueError("data entries must be non-negative")
        if max(entries) >= (1 << output_bits):
            raise ValueError(
                f"data entry {max(entries)} does not fit in output_bits="
                f"{output_bits} bits (max representable "
                f"{(1 << output_bits) - 1})")

        def body(k: int) -> list[tuple[str, int]]:
            return [("x", t) for t in range(output_bits)
                    if (entries[k] >> t) & 1]

        # X-only body: the walk is an involution, skip the adjoint mint.
        self._walk = unary_iteration_kernels(address_bits,
                                             len(entries),
                                             body,
                                             include_adjoint=False)
        self._data = tuple(entries)
        self._output_bits = output_bits

    @property
    def data(self) -> tuple[int, ...]:
        return self._data

    @property
    def num_address(self) -> int:
        return self._walk.num_address

    @property
    def num_ladder(self) -> int:
        """Clean ladder ancillas the kernel needs (|0> in, |0> out)."""
        return self._walk.num_ladder

    @property
    def num_output(self) -> int:
        return self._output_bits

    @property
    def toffoli_count(self) -> int:
        return self._walk.toffoli_count

    def kernel(self):
        """The lookup kernel ``(address, ladder, output)`` — self-inverse."""
        return self._walk.kernel

    def __repr__(self) -> str:
        return (f"QROM(entries={len(self._data)}, "
                f"address_bits={self.num_address}, "
                f"output_bits={self.num_output}, "
                f"toffolis={self.toffoli_count})")
