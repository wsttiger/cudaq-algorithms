# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""QROM: coherent classical-table lookup, three constructions in one surface.

``QROM(data, address_bits, output_bits)`` mints the lookup
``|k>|y> -> |k>|y XOR data[k]>`` behind one kernel signature
``(address: qview, ladder: qview, output: qview)`` — all little-endian
(qubit 0 = LSB, ``docs/conventions.md``), ``ladder`` being ``num_ladder``
clean ancillas (|0> in, |0> out). Addresses ``k >= len(data)`` read as
zero. Three constructions sit behind ``variant=``:

- ``"select"`` — the plain unary-iteration QROM (Babbush et al.,
  `arXiv:1805.03662`): one fused tree walk over the address register
  whose per-address body CNOTs the set bits of ``data[k]`` into the
  output. ``num_ladder = address_bits``.
- ``"select_swap"`` — SELECT-SWAP / QROAM (Low, Kliuchnikov, Schaeffer,
  `arXiv:1812.00954`): the table is split into ``ceil(N / B)`` blocks of
  ``B = block_size`` entries; one unary-iteration walk over the *block
  index* (the high ``address_bits - log2(B)`` address bits) writes the
  visited block's ``B`` entries into ``B`` clean ``output_bits``-wide
  block registers, a swap network controlled on the low ``log2(B)``
  address bits routes the selected entry's register into block slot 0,
  and CNOTs copy slot 0 into the output. The routing and the writes are
  then undone unitarily. ``num_ladder = (address_bits - log2(B)) +
  B * output_bits`` (walk lines first, then the block registers).
- ``"select_copy"`` — SELECT-COPY (Motlagh & Pocrnic,
  `arXiv:2605.20334`, Sec. II.B), adapted here to clean ancillas and a
  fully unitary sandwich. The paper's insight: the select_swap middle —
  route block ``r`` into slot 0, copy slot 0 to the output, route back —
  *is* one multiplexed copy, ``|q>|r>|y> (x)_j |phi_j> ->
  |q>|r>|y XOR phi_r> (x)_j |phi_j>``, implementable as a second unary-
  iteration walk over the low ``log2(B)`` address bits whose per-``r``
  body Toffoli-copies block register ``r`` into the output. Following
  the paper's refinement, the first block-index walk writes entry
  ``(q, 0)`` straight into the output (leaf-controlled CNOTs — free) and
  ``data[q, r] XOR data[q, 0]`` into block register ``r`` for
  ``r = 1..B-1``, the multiplexed copy runs over those ``B - 1``
  registers (its ``r = 0`` body is empty), and a final block-index walk
  unwrites the registers only (X-only, self-inverse). This replaces the
  two swap networks (``2 b (B - 1)`` Toffolis plus a ``4 b (B - 1)``
  CNOT storm) with one copy walk (``W(log2(B), B) + b (B - 1)``
  Toffolis, no routing CNOTs) and drops one block register:
  ``num_ladder = max(address_bits - log2(B), log2(B)) +
  (B - 1) * output_bits`` (a shared walk-line pool — the block-index and
  low-address walks run sequentially and both restore their lines — then
  the block registers).

  ``alpha`` **sequential bit packets** (same paper, Sec. II.D)
  generalize select_copy: the ``b``-bit table is sliced into ``alpha``
  packets of ``ceil(b/alpha)`` and ``floor(b/alpha)`` bits (widest
  packets first — ``alpha`` need not divide ``b``, no padding), looked
  up as ``alpha`` sequential narrow select_copies writing disjoint
  output slices, all sharing one ``(B - 1) * ceil(b/alpha)``-bit
  register bank. Because packet ``s``'s register unwrite walk and
  packet ``s + 1``'s register write walk are adjacent X-only walks over
  the same block index, they fuse into ONE transition walk whose body
  XOR-loads ``content_s XOR content_{s+1}`` (the paper's Sec. II.C
  sequential-QROM trick, materialized coherently): the chain of
  ``alpha`` packets costs ``alpha + 1`` block-index walks, not
  ``2 alpha``. Narrower registers mean ``alpha`` times larger blocks
  fit a fixed ancilla budget — the point of the construction (see the
  cost notes below for when it wins). ``alpha = 1`` *is* plain
  select_copy.
- ``"auto"`` (default) — price all three variants (every power-of-two
  ``block_size``, and for select_copy every ``alpha`` in
  ``[1, output_bits]``) with the exact Toffoli model below, drop
  candidates whose ancilla count exceeds ``max_ancillas`` (when given),
  and take the cheapest; ties prefer ``"select"`` (fewest ancillas),
  then ``"select_copy"`` over ``"select_swap"`` (no routing-CNOT
  storm), then smaller ``alpha`` (fewer walks, fewer CNOTs).

Toffoli accounting (documented contract, pinned by the resource tests;
``W(n, m)`` is the fused walk count from
:mod:`cudaq_algorithms.primitives._unary_iteration`, ``3 * 2^n / 2 - 5``
on full uncontrolled trees):

- ``"select"``: ``W(address_bits, len(data))``.
- ``"select_swap"``: ``2 * W(address_bits - log2(B), ceil(len(data)/B))
  + 2 * output_bits * (B - 1)``. Each controlled register swap is
  ``output_bits`` Fredkins, each Fredkin one Toffoli plus two CNOTs; the
  binary routing network is ``B - 1`` register swaps per direction, and
  both the routing and the block writes run twice (compute + unitary
  uncompute).
- ``"select_copy"`` (``alpha`` packets; ``alpha = 1`` is the plain
  construction): ``(alpha + 1) * W(address_bits - log2(B),
  ceil(len(data)/B)) + alpha * W(log2(B), B) + output_bits * (B - 1)``.
  ``alpha + 1`` block-index walks (write, ``alpha - 1`` fused
  transitions, unwrite), one multiplexed-copy walk per packet, and one
  Toffoli per copied bit per non-zero block — the copy term
  ``sum_s width_s * (B - 1) = output_bits * (B - 1)`` is independent of
  ``alpha``. Ancillas: ``max(address_bits - log2(B), log2(B)) +
  (B - 1) * ceil(output_bits / alpha)``.

Against select_swap at the same block size, select_copy trades the
``2 b (B - 1)`` routing Toffolis for ``b (B - 1) + W(log2(B), B)`` — the
per-bit routing term is exactly halved, which is `arXiv:2605.20334`'s
half-cost claim for the multiplexed copy (in the paper's
measured-uncompute accounting the copy's own iterator costs ``B - 3``,
so the copy wins there for every ``b``). In this library's unitary
accounting the copy's iterator is the coherent walk,
``W(log2(B), B) = 3B/2 - 5`` for ``B >= 8``, so select_copy is strictly
cheaper than select_swap whenever ``W(log2(B), B) < output_bits *
(B - 1)`` — always for ``output_bits >= 2``, and for ``output_bits = 1``
only below ``B = 8`` (tie at 8): single-bit tables with large optimal
blocks are the one regime select_swap still wins, which is why all
three variants stay priced under ``"auto"``.

The papers' headline counts (``N - 1`` lookup Toffolis for SELECT,
``ceil(N/B) + output_bits * (B - 1)`` for QROAM) price ancilla
uncomputation at zero via measurement-and-fixup, and `arXiv:2605.20334`
additionally borrows *dirty* ancillas and restores them by measurement.
These primitives stay strictly unitary on clean ancillas (house rules:
statevector-testable, inverse-composable, no mid-circuit measurement),
so the coherent counts above are what the minted kernels actually cost;
the default ``block_size`` optimizes the coherent model
(``~ sqrt(3 N / (2 output_bits))`` for select_swap,
``~ sqrt(6 N / (2 output_bits + 3))`` for select_copy, rather than the
papers' ``~ sqrt(N / output_bits)``).

When do bit packets win? At fixed ``B`` the packet cost is strictly
increasing in ``alpha`` (each increment adds one block walk and one
copy walk), so with unconstrained ancillas ``alpha = 1`` dominates
pointwise and ``"auto"`` never picks packets. The paper's headline —
``(1 + 1/alpha) N / lambda``, halving toward ``(1 + 1/b) N / lambda``
at ``alpha = b`` — is a statement about a *fixed ancilla budget*
("when constrained by the number of dirty qubits"): the
``alpha``-times-narrower registers let ``B`` grow ``~ alpha``-fold
within the same budget, shrinking the dominant ``N/B`` walks faster
than the ``(alpha + 1)`` prefactor grows. The same tradeoff holds
coherently: pass ``max_ancillas`` and ``"auto"`` optimizes
``(variant, B, alpha)`` under the bound, picking ``alpha > 1``
exactly where it wins (e.g. ``N = 1024``, ``b = 4``, budget 24 clean
ancillas: ``alpha = 4, B = 16`` costs 591 Toffolis vs 772 for the best
``alpha = 1`` fit). The paper's dirty-qubit borrowing and measured
"Restore" (its Fig. 3 / Appendix B) remain out of scope on house
rules (clean ancillas, no measurement); the sequential fusion and the
packet slicing themselves carry over intact — see the records in
``PLAN_primitives_promotion.md`` for the full reconciliation.
Back-to-back lookups of *different* tables are
:class:`cudaq_algorithms.primitives.QROMChain`.

All variants are their own inverse on the clean-ancilla sector the
kernel contract requires (ladder |0> in): every construction restores
its ancillas and XORs ``data[k]`` into the output, so applying
``kernel()`` again XORs it back out — pinned by
``tests/python/test_primitives_qrom.py``. For ``"select"``,
``"select_swap"`` and ``"select_copy"`` at ``alpha = 1`` the inverse is
exact as a *global* unitary too: the ``"select"`` walk XORs the same
table twice, the ``"select_swap"`` sandwich ``U = W S C S^-1 W``
squares to identity (``W`` and ``C`` are involutions and the copy
commutes with itself through the routing), and the ``"select_copy"``
sandwich ``U = W1 C W2`` does too: ``W2 W1 = D`` (the leaf-controlled
direct write of block entry 0 into the output — walks over commuting
X-only bodies factor), ``C`` and ``D`` commute (both X-target the
output; neither touches the other's controls), and ``C``, ``D``,
``W1``, ``W2`` are involutions, so
``U^2 = W1 C D C W2 = W1 D W2 = W1 W1 = I``. At ``alpha > 1`` the
interleaved copy walks read register bits that the transition walks
rewrite, so the global-involution factoring above does not go through
and only the (contractual, tested) clean-sector inverse is claimed.
"""

from __future__ import annotations

from collections.abc import Sequence

from ._unary_iteration import (_OP_BODY_X, _OP_CCX, _OP_CCX_ADDR_ADDR,
                               _OP_CCX_CTRL, _OP_CCX_LADDER_LADDER_TARGET,
                               _OP_CX_ADDR_ADDR, _OP_CX_ADDR_LADDER,
                               _OP_CX_LADDER_LADDER, _OP_CX_LADDER_TARGET,
                               _OP_DESCRIPTIONS, _OP_X_ADDR, _TOFFOLI_OPCODES,
                               _emit_walk, _mint_interpreter,
                               _walk_toffoli_count, unary_iteration_kernels)

__all__ = ["QROM"]

_VARIANTS = ("auto", "select", "select_swap", "select_copy")
_BLOCK_VARIANTS = ("select_swap", "select_copy")

# Walk opcodes whose (a, b) operands index the address register, used to
# shift the block-index walk onto the high address bits.
_ADDRESS_OPERANDS = {
    _OP_X_ADDR: (0, ),
    _OP_CX_ADDR_LADDER: (0, ),
    _OP_CCX: (1, ),
    _OP_CCX_CTRL: (1, ),
    _OP_CX_ADDR_ADDR: (0, 1),
    _OP_CCX_ADDR_ADDR: (0, 1),
}


def _select_swap_cost(address_bits: int, num_entries: int, output_bits: int,
                      block_size: int) -> int:
    """Documented Toffoli price of the select_swap variant."""
    high_bits = address_bits - (block_size.bit_length() - 1)
    num_blocks = -(-num_entries // block_size)
    return (2 * _walk_toffoli_count(high_bits, num_blocks) + 2 * output_bits *
            (block_size - 1))


def _select_copy_cost(address_bits: int,
                      num_entries: int,
                      output_bits: int,
                      block_size: int,
                      alpha: int = 1) -> int:
    """Documented Toffoli price of select_copy with ``alpha`` packets."""
    low_bits = block_size.bit_length() - 1
    high_bits = address_bits - low_bits
    num_blocks = -(-num_entries // block_size)
    return ((alpha + 1) * _walk_toffoli_count(high_bits, num_blocks) +
            alpha * _walk_toffoli_count(low_bits, block_size) + output_bits *
            (block_size - 1))


def _select_swap_ancillas(address_bits: int, output_bits: int,
                          block_size: int) -> int:
    """Clean ancillas (num_ladder) of the select_swap variant."""
    low_bits = block_size.bit_length() - 1
    return address_bits - low_bits + block_size * output_bits


def _select_copy_ancillas(address_bits: int,
                          output_bits: int,
                          block_size: int,
                          alpha: int = 1) -> int:
    """Clean ancillas (num_ladder) of select_copy with ``alpha`` packets."""
    low_bits = block_size.bit_length() - 1
    packet = -(-output_bits // alpha)
    return max(address_bits - low_bits, low_bits) + (block_size - 1) * packet


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
    variant
        ``"auto"`` (default), ``"select"``, ``"select_swap"`` or
        ``"select_copy"`` — see the module docstring for the
        constructions and their exact prices.
    block_size
        SELECT-SWAP / SELECT-COPY block size: a power of two in
        ``[2, 2^(address_bits - 1)]``. Only valid with
        ``variant="select_swap"`` or ``variant="select_copy"``; ``None``
        there picks the power of two minimizing the documented Toffoli
        count.
    alpha
        Number of sequential bit packets (`arXiv:2605.20334`, Sec. II.D)
        in ``[1, output_bits]``; need not divide ``output_bits`` (the
        slices are balanced, widest first). Only valid with
        ``variant="select_copy"``; ``None`` there enumerates every
        ``alpha`` alongside ``block_size`` (with unconstrained ancillas
        the enumeration provably lands on ``alpha = 1`` — packets win
        only under ``max_ancillas``, see the module docstring).
    max_ancillas
        Optional bound on ``num_ladder``: candidates needing more clean
        ancillas are dropped before the price comparison, and a
        ``ValueError`` is raised when nothing fits (or when explicitly
        forced parameters exceed the bound).

    The three views of the minted kernel may live anywhere (no contiguity
    requirement between them), which is what lets callers weave the QROM
    into their own ancilla layouts. Whatever the variant, applying the
    kernel again uncomputes the lookup (see the module docstring for the
    exact inverse claims).
    """

    def __init__(self,
                 data: Sequence[int],
                 address_bits: int,
                 output_bits: int,
                 *,
                 variant: str = "auto",
                 block_size: int | None = None,
                 alpha: int | None = None,
                 max_ancillas: int | None = None) -> None:
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
        if variant not in _VARIANTS:
            raise ValueError(
                f"variant must be one of {_VARIANTS}, got {variant!r}")
        if block_size is not None:
            if variant not in _BLOCK_VARIANTS:
                raise ValueError(
                    "block_size is only valid with variant='select_swap' or "
                    f"variant='select_copy', got variant={variant!r}")
            if (int(block_size) != block_size or block_size < 2
                    or block_size & (block_size - 1) != 0):
                raise ValueError("block_size must be a power of two >= 2, got "
                                 f"{block_size}")
            block_size = int(block_size)
            if block_size > (1 << (address_bits - 1)):
                raise ValueError(
                    f"block_size {block_size} needs at least one block-"
                    f"index address bit: max is 2^(address_bits - 1) = "
                    f"{1 << (address_bits - 1)}")
        if alpha is not None:
            if variant != "select_copy":
                raise ValueError(
                    "alpha (bit packets) is only valid with "
                    f"variant='select_copy', got variant={variant!r}")
            if int(alpha) != alpha or not 1 <= alpha <= output_bits:
                raise ValueError(
                    f"alpha must be an integer in [1, output_bits = "
                    f"{output_bits}], got {alpha}")
            alpha = int(alpha)
        if max_ancillas is not None:
            if int(max_ancillas) != max_ancillas or max_ancillas < 1:
                raise ValueError("max_ancillas must be a positive integer, "
                                 f"got {max_ancillas}")
            max_ancillas = int(max_ancillas)

        # Enumerate the full strategy space (variant, block_size, alpha),
        # price every candidate with the exact documented models, filter
        # by the ancilla bound, and take the cheapest. Ties: select
        # (fewest ancillas), then select_copy over select_swap (no
        # routing-CNOT storm), then smaller alpha, then smaller blocks.
        block_candidates = [1 << s for s in range(1, address_bits)
                            ] if block_size is None else [block_size]
        alpha_candidates = list(range(1, output_bits +
                                      1)) if alpha is None else [alpha]
        candidates = []  # (cost, rank, alpha, block, variant, ancillas)
        if variant in ("auto", "select"):
            candidates.append((_walk_toffoli_count(address_bits, len(entries)),
                               0, 0, 0, "select", address_bits))
        if variant in ("auto", "select_copy"):
            for size in block_candidates:
                for a in alpha_candidates:
                    candidates.append(
                        (_select_copy_cost(address_bits, len(entries),
                                           output_bits, size,
                                           a), 1, a, size, "select_copy",
                         _select_copy_ancillas(address_bits, output_bits, size,
                                               a)))
        if variant in ("auto", "select_swap"):
            for size in block_candidates:
                candidates.append(
                    (_select_swap_cost(address_bits, len(entries), output_bits,
                                       size), 2, 0, size, "select_swap",
                     _select_swap_ancillas(address_bits, output_bits, size)))
        if max_ancillas is not None:
            fitting = [c for c in candidates if c[5] <= max_ancillas]
            if not fitting:
                cheapest_fit = min(c[5] for c in candidates)
                raise ValueError(
                    f"no {variant!r} construction fits max_ancillas="
                    f"{max_ancillas}: the smallest candidate needs "
                    f"{cheapest_fit} clean ancillas")
            candidates = fitting
        _, _, best_alpha, best_size, variant, _ = min(candidates)

        self._data = tuple(entries)
        self._num_address = address_bits
        self._output_bits = output_bits
        self._variant = variant
        self._block_size = best_size if variant in _BLOCK_VARIANTS else None
        self._alpha = best_alpha if variant == "select_copy" else None
        if variant == "select":
            self._build_select()
        elif variant == "select_swap":
            self._build_select_swap()
        else:
            self._build_select_copy()

    def _build_select(self) -> None:
        entries = self._data
        output_bits = self._output_bits

        def body(k: int) -> list[tuple[str, int]]:
            return [("x", t) for t in range(output_bits)
                    if (entries[k] >> t) & 1]

        # X-only body: the walk is an involution, skip the adjoint mint.
        walk = unary_iteration_kernels(self._num_address,
                                       len(entries),
                                       body,
                                       include_adjoint=False)
        self._kernel = walk.kernel
        self._num_ladder = walk.num_ladder
        self._ops = tuple(walk.ops)
        self._toffoli_count = walk.toffoli_count

    def _build_select_swap(self) -> None:
        entries = self._data
        b = self._output_bits
        size = self._block_size
        low_bits = size.bit_length() - 1
        high_bits = self._num_address - low_bits
        num_blocks = -(-len(entries) // size)

        def block_write(j: int) -> list[tuple[str, int]]:
            gates = []
            for i in range(size):
                k = j * size + i
                word = entries[k] if k < len(entries) else 0
                gates.extend(
                    ("x", i * b + t) for t in range(b) if (word >> t) & 1)
            return gates

        walk_ops = _emit_walk(high_bits, num_blocks, False, block_write)
        # Rebase the walk onto the combined kernel: block-index address
        # wires sit above the low routing bits, and the walk's "target"
        # (the block registers) lives in the ladder view after the walk
        # lines, so the leaf-controlled body X becomes a ladder-to-ladder
        # CNOT.
        ops = []
        for op in walk_ops:
            opcode, a, bb, c = op
            if opcode == _OP_BODY_X:
                ops.append((_OP_CX_LADDER_LADDER, a, high_bits + bb, 0))
                continue
            operands = [a, bb, c]
            for position in _ADDRESS_OPERANDS.get(opcode, ()):
                operands[position] += low_bits
            ops.append((opcode, *operands))
        walk_ops = ops

        # Binary routing network: after stage s (controlled on
        # address[s]), block slot i (i = 0 mod 2^(s+1)) holds the entry
        # at low-address offset (i + a_s 2^s + ... + a_0); slot 0 ends
        # holding the selected entry. Each controlled register swap is b
        # Fredkins: cswap(c; u, v) = cx(v, u) ccx(c, u, v) cx(v, u).
        swap_ops = []
        for s in range(low_bits):
            for i in range(0, size, 1 << (s + 1)):
                for t in range(b):
                    u = high_bits + i * b + t
                    v = high_bits + (i + (1 << s)) * b + t
                    swap_ops.append((_OP_CX_LADDER_LADDER, v, u, 0))
                    swap_ops.append((_OP_CCX, u, s, v))
                    swap_ops.append((_OP_CX_LADDER_LADDER, v, u, 0))
        copy_ops = [(_OP_CX_LADDER_TARGET, high_bits + t, t, 0)
                    for t in range(b)]

        # W S C S^-1 W: write, route, copy, unroute, unwrite — clean
        # ancillas and an exactly self-inverse lookup.
        ops = (walk_ops + swap_ops + copy_ops + list(reversed(swap_ops)) +
               walk_ops)
        self._kernel = _mint_interpreter(ops, controlled=False, has_work=False)
        self._num_ladder = high_bits + size * b
        self._ops = tuple(ops)
        self._toffoli_count = sum(1 for op in ops if op[0] in _TOFFOLI_OPCODES)

    def _build_select_copy(self) -> None:
        entries = self._data
        b = self._output_bits
        size = self._block_size
        alpha = self._alpha
        low_bits = size.bit_length() - 1
        high_bits = self._num_address - low_bits
        num_blocks = -(-len(entries) // size)
        # The block-index and low-address walks run sequentially and both
        # restore their ladder lines, so they share one line pool; the
        # B - 1 block registers (block 0 rides the direct output write)
        # sit after it, one packet wide.
        pool = max(high_bits, low_bits)
        # Balanced packet slices, widest first (alpha need not divide b);
        # the shared register bank is one packet stride per block.
        widths = [
            b // alpha + (1 if s < b % alpha else 0) for s in range(alpha)
        ]
        offsets = [sum(widths[:s]) for s in range(alpha)]
        packet = widths[0]

        def packet_word(s: int, k: int) -> int:
            word = entries[k] if k < len(entries) else 0
            return (word >> offsets[s]) & ((1 << widths[s]) - 1)

        def register_word(s: int, j: int, i: int) -> int:
            # Block register i - 1's content while packet s is loaded:
            # packet s of entry (j, i) XOR'd against entry (j, 0)
            # (arXiv:2605.20334's Sel1 refinement, Sec. II.B).
            return packet_word(s, j * size + i) ^ packet_word(s, j * size)

        def block_body(prev: int | None, nxt: int | None):
            # Body for one block-index walk: direct-write the next
            # packet's entry (j, 0) into its output slice (tokens < b)
            # and XOR-transition the register bank from packet ``prev``'s
            # contents to packet ``nxt``'s (tokens >= b) — the paper's
            # Sec. II.C sequential fusion: unwrite and write share one
            # walk. prev=None is the first write, nxt=None the final
            # unwrite.
            def body(j: int) -> list[tuple[str, int]]:
                gates = []
                if nxt is not None:
                    base = packet_word(nxt, j * size)
                    gates.extend(("x", offsets[nxt] + t)
                                 for t in range(widths[nxt])
                                 if (base >> t) & 1)
                for i in range(1, size):
                    word = 0
                    if prev is not None:
                        word ^= register_word(prev, j, i)
                    if nxt is not None:
                        word ^= register_word(nxt, j, i)
                    gates.extend(("x", b + (i - 1) * packet + t)
                                 for t in range(packet) if (word >> t) & 1)
                return gates

            return body

        def block_walk(prev: int | None, nxt: int | None) -> list:
            # Rebase the block-index walk exactly as select_swap does
            # (walk lines lead the ladder view, block-index address wires
            # sit above the low bits); direct output writes stay
            # leaf-controlled body X's, register writes become
            # ladder-to-ladder CNOTs.
            ops = []
            for op in _emit_walk(high_bits, num_blocks, False,
                                 block_body(prev, nxt)):
                opcode, a, bb, c = op
                if opcode == _OP_BODY_X:
                    if bb < b:
                        ops.append((_OP_BODY_X, a, bb, 0))
                    else:
                        ops.append((_OP_CX_LADDER_LADDER, a, pool + bb - b, 0))
                    continue
                operands = [a, bb, c]
                for position in _ADDRESS_OPERANDS.get(opcode, ()):
                    operands[position] += low_bits
                ops.append((opcode, *operands))
            return ops

        def copy_walk(s: int) -> list:
            # The multiplexed copy for packet s: one walk over the low
            # address bits (its wires are address[0 .. low_bits) already —
            # no rebase) whose body at r >= 1 Toffoli-copies block
            # register r - 1 into output slice s; the r = 0 body is empty
            # (the direct write covered it).
            def copy_body(r: int) -> list[tuple[str, int]]:
                if r == 0:
                    return []
                return [("x", (r - 1) * packet + t) for t in range(widths[s])]

            ops = []
            for op in _emit_walk(low_bits, size, False, copy_body):
                opcode, a, bb, c = op
                if opcode == _OP_BODY_X:
                    ops.append((_OP_CCX_LADDER_LADDER_TARGET, a, pool + bb,
                                offsets[s] + bb % packet))
                else:
                    ops.append(op)
            return ops

        # FW_0 C_0 FW_1 C_1 ... C_{alpha-1} FW_alpha: write packet 0,
        # copy it out, XOR-transition to packet 1, copy it out, ...,
        # unwrite packet alpha - 1 — alpha + 1 block walks for alpha
        # packets. At alpha = 1 this is exactly the W1 C W2 sandwich.
        ops = block_walk(None, 0)
        for s in range(alpha):
            ops += copy_walk(s)
            ops += block_walk(s, s + 1 if s + 1 < alpha else None)
        self._kernel = _mint_interpreter(ops, controlled=False, has_work=False)
        self._num_ladder = pool + (size - 1) * packet
        self._ops = tuple(ops)
        self._toffoli_count = sum(1 for op in ops if op[0] in _TOFFOLI_OPCODES)

    @property
    def data(self) -> tuple[int, ...]:
        return self._data

    @property
    def num_address(self) -> int:
        return self._num_address

    @property
    def num_ladder(self) -> int:
        """Clean ancillas the kernel needs (|0> in, |0> out).

        ``address_bits`` walk lines for ``"select"``; walk lines plus the
        ``block_size * output_bits`` block registers for
        ``"select_swap"``; a ``max(high_bits, log2(block_size))`` shared
        walk-line pool plus ``(block_size - 1) * ceil(output_bits /
        alpha)`` block registers for ``"select_copy"``.
        """
        return self._num_ladder

    @property
    def num_output(self) -> int:
        return self._output_bits

    @property
    def variant(self) -> str:
        """The minted construction: 'select', 'select_swap' or 'select_copy'."""
        return self._variant

    @property
    def block_size(self) -> int | None:
        """SELECT-SWAP / SELECT-COPY block size (``None`` for select)."""
        return self._block_size

    @property
    def alpha(self) -> int | None:
        """Number of sequential bit packets (``None`` unless select_copy)."""
        return self._alpha

    @property
    def toffoli_count(self) -> int:
        return self._toffoli_count

    def kernel(self):
        """The lookup kernel ``(address, ladder, output)`` — self-inverse."""
        return self._kernel

    def describe(self) -> str:
        """Decode the minted lookup into a human-readable gate listing.

        A header stating the chosen construction and its price, then one
        line per gate in execution order — exactly what the interpreter
        kernel replays; lines tagged ``# Toffoli`` are the gates
        ``toffoli_count`` counts. This is the intended way to *read* a
        minted lookup (the kernel source is a generic interpreter).
        """
        lines = [
            _OP_DESCRIPTIONS[op].format(a=a, b=b, c=c)
            for op, a, b, c in self._ops
        ]
        return "\n".join([f"QROM lookup: {self!r}"] + lines)

    def __repr__(self) -> str:
        return (f"QROM(entries={len(self._data)}, "
                f"address_bits={self.num_address}, "
                f"output_bits={self.num_output}, "
                f"variant={self._variant!r}, "
                f"block_size={self._block_size}, "
                f"alpha={self._alpha}, "
                f"ancillas={self._num_ladder}, "
                f"toffolis={self.toffoli_count})")
