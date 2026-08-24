# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""QROMChain: sequential lookups of several tables over one address.

Back-to-back table lookups sharing an address register are the shape of
SELECT circuits for factorized Hamiltonians — in tensor-hypercontraction
(THC) block encodings one looks up rotation data, applies the caller's
rotation on the output, uncomputes the lookup, and repeats for the next
factor. ``QROMChain(tables, address_bits, output_bits)`` mints the whole
sequence for ``m = len(tables)`` lookups as ``m + 1`` kernels, following
Motlagh & Pocrnic, "Halving the cost of QROM" (`arXiv:2605.20334`,
Sec. II.C): because every lookup is an X-only XOR into the output,
unloading table ``j`` and loading table ``j + 1`` is ONE lookup of the
difference table ``tables[j] XOR tables[j + 1]``.

The interface (the THC-facing contract)
---------------------------------------

The caller's operations between lookups are arbitrary quantum circuits,
so the chain cannot be one closed kernel; it mints a list of ``m + 1``
step kernels instead, each with the uniform QROM signature
``(address: qview, ladder: qview, output: qview)`` (all little-endian,
``docs/conventions.md``; one clean ladder of width ``num_ladder`` and
one ``output_bits``-wide output register are shared by every step). The
use pattern, with ``ks = chain.kernels()``::

    ks[0](address, ladder, output)      # output == tables[0][k]
    ... caller's operation 0 on the output ...
    ks[1](address, ladder, output)      # output == tables[1][k]
    ... caller's operation 1 ...
    ...
    ks[m - 1](address, ladder, output)  # output == tables[m - 1][k]
    ... caller's operation m - 1 ...
    ks[m](address, ladder, output)      # output restored (XOR'd clean)

Invariant: after step kernel ``j`` (``0 <= j <= m - 1``) the output
holds ``initial XOR tables[j][k]`` for address ``k`` (tables are XOR'd
in, exactly like ``QROM``; addresses ``k >= len(tables[j])`` read as
zero), the ladder is restored to |0>, and the caller's operation ``j``
may act arbitrarily on the *output* register (it must leave the
address and ladder alone — the next transition assumes the address is
intact and the ladder clean). The final kernel ``ks[m]`` XORs
``tables[m - 1][k]`` back out, returning the output to whatever the
caller's operations left, table-free.

Two modes, behaviorally identical:

- ``fused=True`` (default, the paper's Sec. II.C optimization): step 0
  is ``QROM(tables[0])``, step ``j`` for ``1 <= j <= m - 1`` is
  ``QROM(tables[j - 1] XOR tables[j])`` (entrywise, shorter tables
  zero-padded), and step ``m`` is ``QROM(tables[m - 1])`` — ``m + 1``
  full lookups in total.
- ``fused=False`` (the naive semantics reference): step ``j`` for
  ``1 <= j <= m - 1`` replays ``QROM(tables[j - 1])`` (the unload — the
  lookup is self-inverse from a clean ladder) followed by
  ``QROM(tables[j])`` (the load) inside one flat kernel — ``2 m`` full
  lookups in total.

Coherent Toffoli accounting (pinned by the resource tests): every
lookup is priced by the exact per-variant models in
:mod:`cudaq_algorithms.primitives._qrom` (each step auto-priced
independently unless the caller forces ``variant`` / ``block_size`` /
``alpha``, which forward to every step). For tables of equal length the
per-lookup price ``C`` is shape-only, so

    fused:  (m + 1) C        vs        naive:  2 m C,

the paper's ``(m + 1)``-vs-``2 m`` load count carried into the coherent
setting intact — fused is strictly cheaper for every ``m >= 2`` (equal
at ``m = 1``, where both modes are load + unload), saving exactly
``(m - 1) C``. A further coherent refinement is possible and
deliberately not taken: a step's trailing register-unwrite walk touches
only the address/ladder and therefore commutes with the caller's
operation on the output, so it could migrate into the next step's
kernel and fuse with that step's register-write walk (saving one
``W(high_bits, ceil(N/B))`` per boundary, ``m + 2`` block walks
instead of ``2 (m + 1)`` for select_copy steps) — at the price of step
kernels that are no longer self-contained QROMs; see the record in
``PLAN_primitives_promotion.md``.

Every step kernel restores the ladder to |0> and XORs a fixed table
into the output, so from the contractual clean-ladder sector each step
kernel is its own inverse (applying it twice XORs the same table
twice); the fused steps are full ``QROM`` lookups and inherit its
(stronger, ``alpha = 1``) global involution claims.
"""

from __future__ import annotations

from collections.abc import Sequence

from ._qrom import QROM
from ._unary_iteration import _mint_interpreter

__all__ = ["QROMChain"]


class QROMChain:
    """Sequential lookups of ``m`` tables over one address register.

    Parameters
    ----------
    tables
        The lookup tables, one per step: each a non-empty sequence of
        non-negative integers fitting in ``output_bits`` bits. Lengths
        may differ (each at most ``2^address_bits``); addresses beyond a
        table's length read as zero.
    address_bits, output_bits
        Shared address register width and the single shared output
        register width (every table is read into the same register, so
        one width serves the whole chain).
    fused
        ``True`` (default): ``m + 1`` lookups via difference tables
        (`arXiv:2605.20334`, Sec. II.C). ``False``: the naive
        ``2 m``-lookup reference with identical behavior.
    variant, block_size, alpha, max_ancillas
        Forwarded to every step's :class:`QROM` (default: each step
        auto-priced independently).

    See the module docstring for the step-kernel contract and the use
    pattern; ``kernels()`` returns the ``m + 1`` step kernels.
    """

    def __init__(self,
                 tables: Sequence[Sequence[int]],
                 address_bits: int,
                 output_bits: int,
                 *,
                 fused: bool = True,
                 variant: str = "auto",
                 block_size: int | None = None,
                 alpha: int | None = None,
                 max_ancillas: int | None = None) -> None:
        if isinstance(tables,
                      (str, bytes)) or not isinstance(tables, Sequence):
            raise ValueError(
                "tables must be a sequence of lookup tables (one table "
                f"per chain step), got {type(tables).__name__}")
        tables = list(tables)
        if len(tables) == 0:
            raise ValueError("tables must contain at least one table")
        cleaned = []
        for j, table in enumerate(tables):
            if isinstance(table,
                          (str, bytes)) or not isinstance(table, Sequence):
                raise ValueError(
                    f"tables[{j}] must be a sequence of integers, got "
                    f"{type(table).__name__}")
            cleaned.append(list(table))
        # Range/width/length validation is QROM's; wrap its errors with
        # the offending step so chain callers get a loud, located error.
        qrom_options = dict(variant=variant,
                            block_size=block_size,
                            alpha=alpha,
                            max_ancillas=max_ancillas)

        def mint(table, label):
            try:
                return QROM(table, address_bits, output_bits, **qrom_options)
            except ValueError as error:
                raise ValueError(f"QROMChain {label}: {error}") from error

        self._fused = bool(fused)
        if self._fused:
            # Load tables[0]; transition j XOR-loads the difference
            # tables[j - 1] ^ tables[j] (shorter tables zero-padded, as
            # out-of-range addresses read zero); unload tables[m - 1].
            step_tables = [cleaned[0]]
            for j in range(1, len(cleaned)):
                prev, this = cleaned[j - 1], cleaned[j]
                length = max(len(prev), len(this))
                step_tables.append([(prev[k] if k < len(prev) else 0) ^
                                    (this[k] if k < len(this) else 0)
                                    for k in range(length)])
            step_tables.append(cleaned[-1])
            self._steps = tuple(
                mint(table, f"fused step {j} of {len(step_tables)}")
                for j, table in enumerate(step_tables))
            self._kernels = tuple(step.kernel() for step in self._steps)
            step_toffolis = tuple(step.toffoli_count for step in self._steps)
        else:
            self._steps = tuple(
                mint(table, f"table {j} of {len(cleaned)}")
                for j, table in enumerate(cleaned))
            # Transition j replays the previous lookup (the unload) then
            # the next one (the load) inside one flat interpreter kernel.
            kernels = [self._steps[0].kernel()]
            step_toffolis = [self._steps[0].toffoli_count]
            for j in range(1, len(self._steps)):
                ops = list(self._steps[j - 1]._ops) + list(self._steps[j]._ops)
                kernels.append(
                    _mint_interpreter(ops, controlled=False, has_work=False))
                step_toffolis.append(self._steps[j - 1].toffoli_count +
                                     self._steps[j].toffoli_count)
            kernels.append(self._steps[-1].kernel())
            step_toffolis.append(self._steps[-1].toffoli_count)
            self._kernels = tuple(kernels)
            step_toffolis = tuple(step_toffolis)

        self._num_tables = len(cleaned)
        self._num_address = self._steps[0].num_address
        self._output_bits = output_bits
        self._num_ladder = max(step.num_ladder for step in self._steps)
        self._step_toffolis = step_toffolis
        self._toffoli_count = sum(step_toffolis)

    @property
    def num_tables(self) -> int:
        """Number of chained lookups ``m`` (``kernels()`` has ``m + 1``)."""
        return self._num_tables

    @property
    def num_address(self) -> int:
        return self._num_address

    @property
    def num_ladder(self) -> int:
        """Clean ancillas one shared ladder register needs (|0> in and
        out around every step kernel): the maximum over the steps'
        ``QROM.num_ladder`` — a wider ladder view is safe for every step
        (unused lines are never touched)."""
        return self._num_ladder

    @property
    def num_output(self) -> int:
        return self._output_bits

    @property
    def fused(self) -> bool:
        return self._fused

    @property
    def steps(self) -> tuple:
        """The underlying ``QROM`` lookups: the ``m + 1`` difference-table
        lookups when fused, the ``m`` per-table lookups when naive."""
        return self._steps

    @property
    def step_toffoli_counts(self) -> tuple:
        """Toffolis per step kernel (length ``m + 1``, sums to
        ``toffoli_count``)."""
        return self._step_toffolis

    @property
    def toffoli_count(self) -> int:
        return self._toffoli_count

    def kernels(self) -> tuple:
        """The ``m + 1`` step kernels ``(address, ladder, output)``, in
        application order (see the module docstring's use pattern)."""
        return self._kernels

    def describe(self) -> str:
        """One block per step kernel: the step's role in the chain and
        its underlying lookup's gate listing (see ``QROM.describe``)."""
        blocks = [f"QROM chain: {self!r}"]
        if self._fused:
            roles = (["step 0: load tables[0]"] + [
                f"step {j}: fused unload tables[{j - 1}] + load tables[{j}] "
                f"(difference-table lookup)"
                for j in range(1, self._num_tables)
            ] + [
                f"step {self._num_tables}: unload tables[{self._num_tables - 1}]"
            ])
            for role, step in zip(roles, self._steps):
                blocks.append(f"-- {role}\n{step.describe()}")
        else:
            blocks.append(f"-- step 0: load tables[0]\n"
                          f"{self._steps[0].describe()}")
            for j in range(1, self._num_tables):
                blocks.append(
                    f"-- step {j}: unload tables[{j - 1}] then load "
                    f"tables[{j}] (two lookups replayed back to back)\n"
                    f"{self._steps[j - 1].describe()}\n"
                    f"{self._steps[j].describe()}")
            blocks.append(f"-- step {self._num_tables}: unload "
                          f"tables[{self._num_tables - 1}]\n"
                          f"{self._steps[-1].describe()}")
        return "\n".join(blocks)

    def __repr__(self) -> str:
        return (f"QROMChain(tables={self._num_tables}, "
                f"address_bits={self._num_address}, "
                f"output_bits={self._output_bits}, "
                f"fused={self._fused}, "
                f"ancillas={self._num_ladder}, "
                f"toffolis={self._toffoli_count})")
