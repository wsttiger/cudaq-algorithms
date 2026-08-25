# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Fault-tolerant circuit primitives: unary iteration, QROM, arithmetic.

``unary_iteration_kernels`` mints the fused tree walk of Babbush et al.
(`arXiv:1805.03662`, Fig. 7) — apply a per-address body exactly when the
address register equals ``k`` — and ``QROM`` builds coherent
classical-table lookups on top of it: the plain unary-iteration SELECT,
the SELECT-SWAP / QROAM construction of Low, Kliuchnikov and Schaeffer
(`arXiv:1812.00954`), and the SELECT-COPY construction with sequential
bit packets of Motlagh and Pocrnic (`arXiv:2605.20334`). ``QROMChain``
chains lookups of several tables over one address register at
``m + 1`` (not ``2 m``) lookups via the same paper's difference-table
fusion — the surface THC-style multiplexed-rotation SELECTs consume.

The default primitives are strictly unitary: the papers' headline
Toffoli counts assume measurement-based ancilla uncomputation, and the
module docstrings state the coherent costs the minted kernels actually
have, pinned against the compiler by the resource tests.
``measured_unary_iteration_kernels`` is the measurement-assisted
variant: the same SELECT channel with every ladder ancilla disposed of
by Gidney's measure-and-fix-up gadget (`arXiv:1709.06648`), realizing
the papers' ``N - 1``-Toffoli accounting at the price of mid-circuit
measurement (composes as a sub-kernel, but is incompatible with
``cudaq.control`` and ``cudaq.sample`` — see its module docstring).

The reversible integer arithmetic device kernels (:mod:`._arithmetic`)
are the in-place little-endian adders and comparators the lookup-based
constructions compose with: the CDKM/Cuccaro ripple-carry family
(``add_register`` / ``subtract_register``, ``add_constant`` /
``subtract_constant``, ``cmp_ge_constant``) and the ancilla-free Draper
QFT family (``qft`` / ``iqft``, ``add_constant_qft`` /
``subtract_constant_qft``, the ``cmp_ge_constant_qft`` /
``cmp_ge_constant_qft_adj`` pair). Every inverse is hand-written and the
gate prices are compiler-pinned by the resource tests.

Import the subpackage directly (``from cudaq_algorithms.primitives
import QROM``); nothing here is re-exported from the package root.
"""

from ._arithmetic import (add_constant, add_constant_qft, add_register,
                          cmp_ge_constant, cmp_ge_constant_qft,
                          cmp_ge_constant_qft_adj, iqft, phase_add_constant,
                          qft, subtract_constant, subtract_constant_qft,
                          subtract_register)
from ._qrom import QROM
from ._qrom_chain import QROMChain
from ._unary_iteration import UnaryIterationKernels, unary_iteration_kernels
from ._unary_iteration_measured import (MeasuredUnaryIterationKernels,
                                        measured_unary_iteration_kernels)

__all__ = [
    "MeasuredUnaryIterationKernels",
    "QROM",
    "QROMChain",
    "UnaryIterationKernels",
    "add_constant",
    "add_constant_qft",
    "add_register",
    "cmp_ge_constant",
    "cmp_ge_constant_qft",
    "cmp_ge_constant_qft_adj",
    "iqft",
    "measured_unary_iteration_kernels",
    "phase_add_constant",
    "qft",
    "subtract_constant",
    "subtract_constant_qft",
    "subtract_register",
    "unary_iteration_kernels",
]
