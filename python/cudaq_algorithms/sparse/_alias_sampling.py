# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Coherent alias sampling: PREPARE-with-garbage for weighted indices.

``AliasSamplingPrepare(weights, mu)`` mints the Babbush-style
(`arXiv:1805.03662`, Sec. III.D) state preparation whose *index-register
marginal* is the normalized weight distribution ``w / lambda`` up to a
``mu``-bit discretization: classical Vose preprocessing builds integer
keep/alias tables, and the circuit is

1. uniform superposition over the ``K = 2^num_index`` bins (``weights``
   is zero-padded to a power of two; padded bins carry exactly zero
   probability),
2. a QROM lookup of ``(alias_k, keep_k)`` at bin ``k``,
3. a comparator of ``keep_k`` against a uniform ``mu``-bit reference
   register (built from the CDKM register adders of ``_arithmetic``:
   subtract on a ``mu+1``-bit extension, copy the borrow, add back), and
4. a controlled swap of the bin index with the alias when the reference
   is not below ``keep_k``.

PREPARE with garbage — read this before consuming
-------------------------------------------------

The output is **not** ``sum_k sqrt(w_k / lambda) |k> |0...0>``: the
alias/keep/reference/flag registers stay *entangled* with the index (only
the amplitude *magnitudes* marginalize to ``w / lambda``). That is
exactly the coherent-alias-sampling contract: qubitization and V2's
``SparseLCUEncoding`` tolerate the garbage only because every use is the
symmetric sandwich ``PREPARE ... PREPARE^dagger`` — the garbage registers
are uncomputed by the hand-written ``adjoint_kernel()`` applied to the
same registers, never discarded, never reflected over while dirty, and
never consumed by any other circuit element in between. Anything that
needs a clean ``sum sqrt(p_k) |k>`` (e.g. amplitude arithmetic on the
index alone) must use a different preparation.

Discretization: the integer tables represent bin probabilities exactly as
``W_k / (K 2^mu)`` (integer Vose is exact, and full bins are stored
self-aliased so the ``keep = 2^mu`` edge loses nothing); the only error
is rounding ``w_k / lambda`` to the integers ``W_k``, bounded per bin by
``1.5 / (K 2^mu)`` (half-ulp rounding plus at most one unit of residual
redistribution) — exposed as ``discretization_bound`` and derived, never
assumed, in the tests.

Kernel signatures (little-endian; ``docs/conventions.md``): both
``kernel()`` and ``adjoint_kernel()`` are ``(index: qview, garbage:
qview)`` with ``index`` of width ``num_index`` and ``garbage`` of width
``num_garbage`` laid out as ``[alias(num_index) | keep(mu) | keep_pad |
ref(mu) | ref_pad | flag | ladder(num_index) | carry]``. Both registers
must be |0...0> on entry to ``kernel()``; ``adjoint_kernel()`` is the
literal gate-reversal (no ``cudaq.adjoint``, cuda-quantum#4897/#4898) and
returns them to |0...0>. These kernels call the QROM and CDKM sub-kernels
and are therefore NOT flat: do not place them under ``cudaq.control``
(control-variant generation rejects nested kernel calls) — consumers
control SELECT, not PREPARE.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import cudaq
import numpy as np

from ._arithmetic import add_register, subtract_register
from ._qrom import QROM
from ._sparse_oracle import _retain

__all__ = ["AliasSamplingPrepare"]


def _vose_tables(probabilities: Sequence[float],
                 mu: int) -> tuple[list[int], list[int], list[float]]:
    """Integer Vose preprocessing at ``mu`` bits.

    Returns ``(keep, alias, table_probabilities)`` over the padded bins:
    ``keep[k] in [0, 2^mu - 1]``, ``alias`` a bin index (full bins are
    self-aliased), and ``table_probabilities[k] = W_k / (K 2^mu)`` — the
    *exact* probability the circuit realizes for bin ``k``.
    """
    num_bins = len(probabilities)
    unit = 1 << mu
    total = num_bins * unit

    # Integer weights summing exactly to K * 2^mu: round, then push the
    # rounding residual back one unit at a time onto the bins whose
    # rounding moved the furthest in the residual's direction (weights
    # stay non-negative: a downward push targets a bin with W >= 1).
    ideal = [p * total for p in probabilities]
    weights = [int(round(v)) for v in ideal]
    residual = total - sum(weights)
    step = 1 if residual > 0 else -1
    order = sorted(range(num_bins),
                   key=lambda k: step * (ideal[k] - weights[k]),
                   reverse=True)
    cursor = 0
    for _ in range(abs(residual)):
        while step < 0 and weights[order[cursor % num_bins]] == 0:
            cursor += 1
        weights[order[cursor % num_bins]] += step
        cursor += 1

    # Integer Vose: pair small bins (W < 2^mu) with large ones (W > 2^mu).
    keep = [unit] * num_bins
    alias = list(range(num_bins))
    remaining = list(weights)
    small = [k for k in range(num_bins) if remaining[k] < unit]
    large = [k for k in range(num_bins) if remaining[k] > unit]
    while small and large:
        s = small.pop()
        l = large[-1]
        keep[s] = remaining[s]
        alias[s] = l
        remaining[l] -= unit - remaining[s]
        if remaining[l] <= unit:
            large.pop()
            if remaining[l] < unit:
                small.append(l)
    # Bins left at exactly 2^mu are full: stored self-aliased with the
    # keep clamped into mu bits — self-aliasing makes the keep value
    # irrelevant (both comparator branches land on the same bin), so the
    # clamp is exact, not an approximation.
    keep = [min(v, unit - 1) for v in keep]
    table = [w / total for w in weights]
    return keep, alias, table


class AliasSamplingPrepare:
    """Coherent alias-sampling PREPARE (see the module docstring).

    Parameters
    ----------
    weights
        Non-negative, finite weights with a positive sum; zero entries
        are allowed (their bins get exactly zero probability). Padded
        with zero-weight bins to the next power of two.
    mu
        Keep-threshold precision in bits (>= 1); the per-bin
        discretization error is bounded by ``discretization_bound``.
    """

    def __init__(self, weights: Sequence[float], mu: int) -> None:
        values = [float(w) for w in weights]
        if len(values) == 0:
            raise ValueError("weights must be non-empty")
        if any(not math.isfinite(w) for w in values):
            raise ValueError("weights must be finite")
        if any(w < 0.0 for w in values):
            raise ValueError("weights must be non-negative")
        lam = sum(values)
        if not lam > 0.0:
            raise ValueError("weights sum to zero: nothing to prepare")
        if int(mu) != mu or mu < 1:
            raise ValueError("mu must be a positive integer (mu >= 1)")
        mu = int(mu)

        num_index = max(1, (len(values) - 1).bit_length())
        num_bins = 1 << num_index
        padded = values + [0.0] * (num_bins - len(values))
        probabilities = [w / lam for w in padded]
        keep, alias, table = _vose_tables(probabilities, mu)

        self._lam = lam
        self._mu = mu
        self._num_index = num_index
        self._num_bins = num_bins
        self._num_weights = len(values)
        self._keep = tuple(keep)
        self._alias = tuple(alias)
        self._probabilities = np.asarray(probabilities)
        self._table_probabilities = np.asarray(table)

        # QROM data: alias in the low num_index bits, keep above it.
        self._qrom = QROM(
            [alias[k] | (keep[k] << num_index) for k in range(num_bins)],
            address_bits=num_index,
            output_bits=num_index + mu)
        self._build_kernels()

    # ------------------------------------------------------------------
    # Kernel construction
    # ------------------------------------------------------------------

    def _build_kernels(self) -> None:
        # Garbage layout offsets (see the module docstring); everything
        # unpacked into scalar locals (no tuple/self capture in kernels).
        m = self._num_index
        mu = self._mu
        k0 = m  # keep (keep_pad at k0 + mu)
        r0 = m + mu + 1  # ref (ref_pad at r0 + mu)
        flag = m + 2 * mu + 2
        l0 = m + 2 * mu + 3
        c0 = l0 + m  # carry
        qrom_kernel = self._qrom.kernel()

        @cudaq.kernel
        def sparse_alias_prepare(index: cudaq.qview, garbage: cudaq.qview):
            # Uniform superpositions over bins and the mu-bit reference.
            for b in range(m):
                h(index[b])
            for b in range(mu):
                h(garbage[r0 + b])
            # (alias_k, keep_k) lookup: output is garbage[0 : m + mu].
            qrom_kernel(index, garbage[l0:l0 + m], garbage[0:m + mu])
            # flag <- NOT (ref < keep): subtract keep on the (mu+1)-bit
            # extension of ref, copy the borrow (MSB), add keep back.
            subtract_register(garbage[k0:k0 + mu + 1], garbage[r0:r0 + mu + 1],
                              garbage[c0:c0 + 1])
            cx(garbage[r0 + mu], garbage[flag])
            x(garbage[flag])
            add_register(garbage[k0:k0 + mu + 1], garbage[r0:r0 + mu + 1],
                         garbage[c0:c0 + 1])
            # Swap the bin index with its alias on the flag.
            for b in range(m):
                swap.ctrl(garbage[flag], index[b], garbage[b])

        @cudaq.kernel
        def sparse_alias_prepare_adj(index: cudaq.qview, garbage: cudaq.qview):
            """Hand-written gate-reversal of ``sparse_alias_prepare``."""
            for j in range(m):
                b = m - 1 - j
                swap.ctrl(garbage[flag], index[b], garbage[b])
            subtract_register(garbage[k0:k0 + mu + 1], garbage[r0:r0 + mu + 1],
                              garbage[c0:c0 + 1])
            x(garbage[flag])
            cx(garbage[r0 + mu], garbage[flag])
            add_register(garbage[k0:k0 + mu + 1], garbage[r0:r0 + mu + 1],
                         garbage[c0:c0 + 1])
            # The X-only QROM lookup is its own inverse.
            qrom_kernel(index, garbage[l0:l0 + m], garbage[0:m + mu])
            for j in range(mu):
                b = mu - 1 - j
                h(garbage[r0 + b])
            for j in range(m):
                b = m - 1 - j
                h(index[b])

        _retain(sparse_alias_prepare, sparse_alias_prepare_adj)
        self._prepare = sparse_alias_prepare
        self._prepare_adj = sparse_alias_prepare_adj

    # ------------------------------------------------------------------
    # Inspection
    # ------------------------------------------------------------------

    @property
    def lam(self) -> float:
        """The weight one-norm ``lambda = sum_k w_k``."""
        return self._lam

    @property
    def mu(self) -> int:
        return self._mu

    @property
    def num_index(self) -> int:
        """Index register width (``log2`` of the padded bin count)."""
        return self._num_index

    @property
    def num_bins(self) -> int:
        """Padded bin count ``2^num_index`` (>= ``len(weights)``)."""
        return self._num_bins

    @property
    def num_garbage(self) -> int:
        """Garbage register width (layout in the module docstring)."""
        return 2 * self._num_index + 2 * self._mu + 4

    @property
    def ladder_offset(self) -> int:
        """Offset of the ``num_index``-wide QROM ladder inside garbage.

        The ladder qubits are clean (|0>) between ``kernel()`` and
        ``adjoint_kernel()`` — unlike the rest of the garbage — so a
        SELECT sandwiched between them may reuse
        ``garbage[ladder_offset : ladder_offset + num_index]`` as its own
        ladder.
        """
        return self._num_index + 2 * self._mu + 3

    @property
    def keep(self) -> tuple[int, ...]:
        """Per-bin mu-bit keep thresholds (full bins clamped, self-aliased)."""
        return self._keep

    @property
    def alias(self) -> tuple[int, ...]:
        return self._alias

    @property
    def probabilities(self) -> np.ndarray:
        """The ideal padded distribution ``w / lambda`` (zero-padded)."""
        return self._probabilities.copy()

    @property
    def table_probabilities(self) -> np.ndarray:
        """The exactly realized index marginal (integer-table exact)."""
        return self._table_probabilities.copy()

    @property
    def discretization_bound(self) -> float:
        """Per-bin bound on |table - ideal| probability (see module doc)."""
        return 1.5 / (self._num_bins * (1 << self._mu))

    def __repr__(self) -> str:
        return (f"AliasSamplingPrepare(weights={self._num_weights} "
                f"(padded to {self.num_bins} bins), mu={self.mu}, "
                f"lambda={self.lam:.6g}, index_qubits={self.num_index}, "
                f"garbage_qubits={self.num_garbage})")

    # ------------------------------------------------------------------
    # Kernels
    # ------------------------------------------------------------------

    def kernel(self):
        """PREPARE ``(index, garbage)`` — both |0...0> on entry."""
        return self._prepare

    def adjoint_kernel(self):
        """PREPARE^dagger ``(index, garbage)`` — the exact inverse."""
        return self._prepare_adj
