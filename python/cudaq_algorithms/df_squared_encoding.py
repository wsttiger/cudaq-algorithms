# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                        #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""Reusable device primitives for a literature-faithful (von Burg / THC)
double-factorized *squared*-oracle block encoding.

This module is being built side-by-side with
:class:`cudaq_algorithms.df_encoding.DoubleFactorizedEncoding` (which is left
untouched). The target construction realises each rank-one X-DF leaf as a
*square* of an inner one-body operator,

    leaf_t = 1/2 sum_i s_{t,i} ( sum_p W^{t,i}_p . U^dag_{p,t} Z_0 U_{p,t} )^2

(von Burg, PRX Quantum 2, 030305; Lee et al., arXiv:2011.03494). Reaching
that form needs three reusable device primitives: a register-driven
("programmable") Givens rotation, a QROM angle-load, and block-encoding
squaring on disjoint ancilla copies.

Status
------
The **programmable Givens** primitive (primitive #1) is implemented and
dense-validated here (see ``tests/python/test_df_encoding2.py``). The
host-side rank-one factoring and one-column partial Givens sweep (the
classical Phase-0 pieces, correct independent of any circuit
normalisation) are also provided.

The full ``DoubleFactorizedSquaredEncoding`` class is **not yet assembled**:
assembling it is blocked on a normalisation question that the design plan
asserts but does not concretely resolve (whether the coherent-squaring
oracle achieves the ``burg`` one-norm ``1/4 |lambda| S^2`` per leaf, which
is *strictly smaller* than the existing encoding's ``lcu`` one-norm). A
plain disjoint-copy squaring provably yields ``1/2 |lambda| S^2`` per leaf,
which is *larger* than the existing encoding's leaf weight for every
rank-one leaf, so it is not worth shipping; the smaller ``burg`` weight
needs the specific coherent construction. Per the project's validation
culture (fail loudly, never guess a normalisation) the class is deferred
until that construction is pinned down. See the task report for the exact
algebraic argument.

Kernel-language constraints (CLAUDE.md) shape everything below: no early
``return`` (cuda-quantum#4845); no empty list across the boundary
(cuda-quantum#4847); no ``cudaq.adjoint`` -- every inverse is hand-written
and pinned by an inverse-property test (cuda-quantum#4897/#4898);
``exp_pauli`` needs runtime-contiguous slices and a classical angle, so the
only register-conditioned rotation is the full-control ``ry.ctrl``.
"""

from __future__ import annotations

import math

import numpy as np

import cudaq

__all__ = [
    "givens_frame",
    "givens_frame_adj",
    "register_driven_ry",
    "programmable_givens",
    "programmable_givens_adj",
    "one_column_sweep",
    "rank_one_leaf_slots",
    "quantize_angle",
]

# ============================================================================
# Primitive #1: register-driven ("programmable") Givens
#
# A single spatial Givens on a contiguous three-qubit slice is the fermionic
# hop ``exp(theta (XZY - YZX)/2)`` (``df_encoding.apply_basis_rotations``).
# Its generator ``A = (XZY - YZX)/2`` is Clifford-conjugate to a *controlled*
# single-qubit Y rotation:
#
#     F A F^dag = proj(q0 = 1) . Y_q2 ,   F = CX(q1->q2) . CX(q2->q0)
#
# where the slice qubits are (q0, q1, q2). Hence
#
#     givens3(theta) = F^dag . CRy(2 theta; ctrl=q0, tgt=q2) . F
#
# (verified densely to 1e-16 against the exp_pauli XZY/YZX pair). Replacing
# the fixed ``CRy(2 theta)`` with a bit-cascade of ``ry.ctrl`` over an angle
# register makes the rotation register-driven (Ry angles add, so the
# fixed-point angle is applied exactly in fp64).
# ============================================================================


@cudaq.kernel
def givens_frame(slice3: cudaq.qview):
    """The fixed Clifford frame ``F = CX(q1->q2) . CX(q2->q0)``.

    Applied to a contiguous three-qubit slice ``(q0, q1, q2)``. Folds the
    Jordan-Wigner string (middle-qubit Z) into the two end qubits so the
    hop generator becomes a single controlled-``Y`` rotation.
    """
    x.ctrl(slice3[2], slice3[0])
    x.ctrl(slice3[1], slice3[2])


@cudaq.kernel
def givens_frame_adj(slice3: cudaq.qview):
    """Inverse of :func:`givens_frame` (CNOTs are self-inverse, reversed)."""
    x.ctrl(slice3[1], slice3[2])
    x.ctrl(slice3[2], slice3[0])


@cudaq.kernel
def register_driven_ry(slice3: cudaq.qview, angle_reg: cudaq.qview,
                       full_turn: float):
    """Register-driven ``CRy(2 theta; ctrl=q0, tgt=q2)`` inside the frame.

    ``theta = full_turn * sum_j bit_j 2^{-(j+1)}`` is the fixed-point angle
    held (little-fraction-first) in ``angle_reg``. Each bit ``j`` contributes
    ``-full_turn * 2^{-j}`` to the total ``Ry`` angle ``-2 theta`` on ``q2``,
    doubly controlled by the frame qubit ``q0`` and the angle bit -- so the
    rotation is applied only where the frame control is set. (The sign matches
    CUDA-Q's ``exp_pauli`` hop, which realises ``givens(-theta)`` relative to
    the plain ``exp(-i t P)`` convention.)
    """
    nbits = angle_reg.size()
    for j in range(nbits):
        coeff = -full_turn * (2.0**(-j))
        ry.ctrl(coeff, slice3[0], angle_reg[j], slice3[2])


@cudaq.kernel
def programmable_givens(slice3: cudaq.qview, angle_reg: cudaq.qview,
                        full_turn: float):
    """Spatial Givens on ``slice3`` with the angle read from ``angle_reg``.

    Equal to ``df_encoding.apply_basis_rotations``'s single ``XZY``/``YZX``
    hop at the fixed-point angle encoded in ``angle_reg`` -- but the angle
    lives in a quantum register, so the rotation can be selected per QROM
    address.
    """
    givens_frame(slice3)
    register_driven_ry(slice3, angle_reg, full_turn)
    givens_frame_adj(slice3)


@cudaq.kernel
def programmable_givens_adj(slice3: cudaq.qview, angle_reg: cudaq.qview,
                            full_turn: float):
    """Hand-written inverse of :func:`programmable_givens`.

    The frame is self-conjugate around the rotation; the inverse negates
    every ``Ry`` angle (``ry.ctrl(-coeff, ...)``). Pinned by an
    inverse-property test rather than ``cudaq.adjoint``
    (cuda-quantum#4897/#4898).
    """
    nbits = angle_reg.size()
    givens_frame(slice3)
    for j in range(nbits):
        coeff = full_turn * (2.0**(-j))
        ry.ctrl(coeff, slice3[0], angle_reg[j], slice3[2])
    givens_frame_adj(slice3)


# ============================================================================
# Phase-0 host helpers (pure NumPy; correct independent of circuit
# normalisation). These feed the eventual QROM/angle tables.
# ============================================================================


def one_column_sweep(column: np.ndarray) -> list[float]:
    """Adjacent-Givens angles reducing a unit column ``v`` to ``+- e_0``.

    An ``O(n)`` specialisation of ``df_encoding._givens_sweep``'s inner loop:
    ``n - 1`` adjacent ``atan2`` plane rotations that zero ``v`` bottom-up
    onto ``e_0`` (the ``+-`` sign is irrelevant -- number operators are
    invariant under per-mode sign flips). All ``n - 1`` angles are returned,
    including exact zeros, so the count is fixed and address-regular for a
    QROM.
    """
    work = np.array(column, dtype=float, copy=True).ravel()
    n = work.shape[0]
    angles: list[float] = []
    for row in range(n - 1, 0, -1):
        upper, lower = work[row - 1], work[row]
        theta = math.atan2(lower, upper)
        c, s = math.cos(theta), math.sin(theta)
        work[row - 1] = c * upper + s * lower
        work[row] = -s * upper + c * lower
        angles.append(theta)
    # angles are emitted top-adjacent-first (rows n-1..1 -> plane (row-1,row));
    # reverse to circuit (application) order if needed by the caller.
    return angles


def rank_one_leaf_slots(rotation: np.ndarray,
                        core: np.ndarray,
                        rank_threshold: float = 1e-12):
    """Rank-one factor a (symmetric) leaf core into squared-oracle slots.

    Per the ``burg`` idiom (``_factorization.double_factorization_one_norm``):
    eigendecompose ``core = sum_i lambda_i v_i v_i^T`` and, for each kept
    eigenpair, emit a slot carrying ``W_i = sqrt(|lambda_i|) v_i``, the sign
    ``s_i = sign(lambda_i)``, the column-abs-sum ``S_i = sum_k |v_i[k]|``, and
    the burg weight ``1/4 |lambda_i| S_i^2``. X-DF leaves are exactly rank one
    (one slot); C-DF leaves emit up to ``rank`` slots that share ``rotation``.

    Returns a list of dicts (one per slot).
    """
    core = np.asarray(core, dtype=float)
    core = 0.5 * (core + core.T)
    eigenvalues, vectors = np.linalg.eigh(core)
    slots = []
    for i in range(len(eigenvalues)):
        lam = float(eigenvalues[i])
        v = vectors[:, i]
        weight = 0.25 * abs(lam) * float(np.sum(np.abs(v)))**2
        if weight < rank_threshold:
            continue
        slots.append({
            "rotation": np.asarray(rotation, dtype=float),
            "eigenvalue": lam,
            "vector": np.asarray(v, dtype=float),
            "sign": 1 if lam >= 0.0 else -1,
            "column_abs_sum": float(np.sum(np.abs(v))),
            "burg_weight": weight,
        })
    return slots


def quantize_angle(theta: float, bits: int, full_turn: float) -> float:
    """Round ``theta`` to the fixed-point grid of ``register_driven_ry``.

    Resolution ``full_turn / 2^bits`` over ``[0, full_turn)``, round-half-even.
    The *only* non-exact step of the eventual encoding; validated by an
    error-scaling test, not a magic tolerance. With ``bits is None`` the
    caller keeps exact fp64 angles instead.
    """
    if bits <= 0:
        raise ValueError("bits must be positive")
    frac = (theta / full_turn) % 1.0
    code = int(np.round(frac * (1 << bits)))
    code %= (1 << bits)
    return full_turn * code / (1 << bits)
