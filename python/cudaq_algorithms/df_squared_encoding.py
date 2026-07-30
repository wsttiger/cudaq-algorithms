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

(von Burg, PRX Quantum 2, 030305; Lee et al., arXiv:2011.03494).

Construction
------------
``DoubleFactorizedSquaredEncoding`` (below) is a *global* coherent LCU over
"slots": the ``1/4 |lambda| S^2`` ``burg`` weight is **not** a per-leaf
quantity (a standalone per-leaf squaring gives ``1/2 |lambda| S^2``, worse
than the existing encoding, and any per-leaf target below the leaf spectral
norm is impossible) -- it is the aggregate one-norm of the whole two-body
oracle built as one circuit, where the leaves' mutual cancellation lets the
total ``alpha`` clear ``||H_2||``. Each two-body slot realises its leaf
eigenpair as a Chebyshev sandwich ``M' = V R V`` (block ``2 Ahat^2 - I``,
one inner PREPARE counted once -> the ``1/4``). The class reuses
``DoubleFactorizedEncoding`` verbatim for the ``kappa`` / one-body / constant
machinery and swaps only the two-body realisation; the full block equals the
same dense ``H`` (dense-validated to 1e-11).

The **programmable Givens** primitive (primitive #1, used by the optional
QROM/angle-register path) is dense-validated in
``tests/python/test_df_encoding2.py``; the assembled encoding is validated in
``tests/python/test_df_squared_encoding.py``.

Constant bookkeeping (subtle)
-----------------------------
``const_burg = df_const + 1/4 sum_{t,i} lambda^t_i ((S^t_i)^2 - 1)``. The
squared slots carry identity content ``1/4 lambda S^2`` where the existing
ZZ-word two-body carried ``1/4 lambda`` (= ``1/4`` trace); the ``(S^2 - 1)``
term is that difference. This differs from a naive ``+ 1/4 lambda S^2`` by
``-1/4 sum lambda`` -- a pure identity offset a standalone reconstruction
otherwise double-counts (it is invisible to ``alpha - |const| == burg`` and
to ``alpha >= ||H||``, but the dense ``block == H/alpha`` gate pins it).

Kernel-language constraints (CLAUDE.md) shape everything below: no early
``return`` (cuda-quantum#4845); no empty list across the boundary
(cuda-quantum#4847); no ``cudaq.adjoint`` -- every inverse is hand-written
and pinned by an inverse-property test (cuda-quantum#4897/#4898);
``exp_pauli`` needs runtime-contiguous slices and a classical angle, so the
only register-conditioned rotation is the full-control ``ry.ctrl``.
"""

from __future__ import annotations

import math
from typing import Any, Union

import numpy as np
from numpy.typing import ArrayLike

import cudaq

from .block_encoding import Kernel
from .common_kernels import (_validate_power, controlled_reflect_about_zero,
                             reflect_about_zero)
from .df_encoding import DoubleFactorizedEncoding, _givens_sweep
from .double_factorization import DoubleFactorization
from .pauli_lcu import _prepare_angles, prepare, unprepare

__all__ = [
    "givens_frame",
    "givens_frame_adj",
    "register_driven_ry",
    "programmable_givens",
    "programmable_givens_adj",
    "one_column_sweep",
    "rank_one_leaf_slots",
    "quantize_angle",
    "DoubleFactorizedSquaredEncoding",
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


# ============================================================================
# Squared-oracle block encoding (von Burg / burg one-norm)
#
# A GLOBAL coherent LCU over "slots": one identity slot (the constant), one
# single-Z slot per (orbital, spin) of the corrected one-body matrix kappa,
# and one two-body slot per kept leaf eigenpair. Every slot body is a Hermitian
# involution on the system, so the outer PREPARE/SELECT/PREPARE-dagger
# block-encodes H / alpha. NOTE: the two-body sandwich V R V spreads the inner
# register (block-correct, but SELECT is a Hermitian involution only on the
# good ancilla subspace inner = flag = 0). Single-use block encoding post-
# selects inner = 0 so this is invisible; qubitization iterates, so the walk's
# reflection is about |Pi> over the WHOLE ancilla (outer + flag + inner) --
# ``_reflect_about_prepare_full`` -- not just the outer index. Walk/QSVT then
# stay in the good subspace step to step and reproduce T_p(-H/alpha) exactly.
#
# Each two-body slot realises its leaf eigenpair as a *square* via a Chebyshev
# sandwich M' = V . reflect_about_zero(inner) . V, an inner block encoding V of
# the normalised one-body operator Ahat (|Ahat| <= 1) built from a single inner
# PREPARE used twice. With reflect_about_zero = I - 2|0><0| the sandwich block
# is I - 2 Ahat^2; flipping the slot's outer sign turns it into the wanted
# sign * (2 Ahat^2 - I) = sign * T_2(Ahat). The 1/4 factor (vs the existing
# encoding's 1/2) is the product of the 1/2 from H's 1/2 sum (One_L)^2 and the
# 1/2 from Ahat^2 = 1/2 (I + T_2(Ahat)); a *single* inner PREPARE (one-norm
# counted once) is what makes it 1/4 rather than 1/2.
#
# Ancilla layout (good subspace = all ancilla |0>):
#     [ outer index : n_out ][ flag : 1 ][ inner index+spin : n_in ]
# The flag qubit (= AND(outer == slot), or AND(control, outer == slot) for the
# controlled variants) makes every per-slot controlled operation reference
# either two individual qubits ([flag, slice-q0]) or a contiguous view
# ([flag, inner...]); CUDA-Q rejects a control set that mixes a qview with a
# separate qubit, and this layout sidesteps that without an angle register.
#
# Kernel-language constraints (CLAUDE.md): positive ``if`` only, never early
# ``return`` (#4845); flattened lists padded so none is empty (#4847); no
# ``cudaq.adjoint`` -- ``_offset_unprepare`` is the hand-written inverse of
# ``_offset_prepare`` (pinned by an inverse-property test); only full-control
# ``ry.ctrl``; the Givens frames go through the Clifford-F + ``ry.ctrl`` route
# of the programmable-Givens primitive (no ``exp_pauli`` slice constraint).
# ============================================================================

_PI = 3.141592653589793


@cudaq.kernel
def _offset_prepare(reg: cudaq.qview, angles: list[float], base: int):
    """``pauli_lcu.prepare`` reading its angles from ``angles[base:]``.

    Lets one flat angle array feed the per-slot inner PREPAREs without
    slicing a list across the kernel boundary.
    """
    n = reg.size()
    if n > 0:
        ry(angles[base], reg[0])
    idx = 1
    for layer in range(1, n):
        branches = 1 << layer
        for branch in range(branches):
            for bit in range(layer):
                if ((branch >> bit) & 1) == 0:
                    x(reg[layer - 1 - bit])
            ry.ctrl(angles[base + idx], reg.front(layer), reg[layer])
            idx += 1
            for bit in range(layer):
                if ((branch >> bit) & 1) == 0:
                    x(reg[layer - 1 - bit])


@cudaq.kernel
def _offset_unprepare(reg: cudaq.qview, angles: list[float], base: int,
                      count: int):
    """Hand-written inverse of :func:`_offset_prepare` (no ``cudaq.adjoint``).

    ``count`` is the number of angles this register consumes
    (``2**reg.size() - 1``); the reverse walk starts at ``base + count - 1``.
    """
    n = reg.size()
    idx = base + count - 1
    for reverse_layer in range(n - 1):
        layer = n - 1 - reverse_layer
        branches = 1 << layer
        for reverse_branch in range(branches):
            branch = branches - 1 - reverse_branch
            for bit in range(layer):
                if ((branch >> bit) & 1) == 0:
                    x(reg[layer - 1 - bit])
            ry.ctrl(-angles[idx], reg.front(layer), reg[layer])
            idx -= 1
            for bit in range(layer):
                if ((branch >> bit) & 1) == 0:
                    x(reg[layer - 1 - bit])
    if n > 0:
        ry(-angles[base], reg[0])


@cudaq.kernel
def _inner_select(ancilla: cudaq.qview, system: cudaq.qview,
                  inner_signs: list[int], sign_ptr: int, flag_lo: int,
                  n_in: int, two_n: int):
    """SEL_in: signed ``Z`` on ``system[a]`` for each inner address ``a``.

    Controlled on ``flag == 1`` (outer index == this slot) and the inner
    register holding address ``a`` (both folded into the contiguous
    ``ancilla[flag_lo : flag_lo + 1 + n_in]`` control view via inner
    X-conjugation, the PauliLCU SELECT convention). Address ``a = 2k + sigma``
    maps to ``system[a]`` directly.
    """
    for a in range(two_n):
        for b in range(n_in):
            if ((a >> (n_in - 1 - b)) & 1) == 0:
                x(ancilla[flag_lo + 1 + b])
        z.ctrl(ancilla[flag_lo:flag_lo + 1 + n_in], system[a])
        if inner_signs[sign_ptr + a] < 0:
            z.ctrl(ancilla[flag_lo:flag_lo + n_in], ancilla[flag_lo + n_in])
        for b in range(n_in):
            if ((a >> (n_in - 1 - b)) & 1) == 0:
                x(ancilla[flag_lo + 1 + b])


@cudaq.kernel
def _reflect_about_prepare_full(ancilla: cudaq.qview,
                                outer_angles: list[float], n_out: int):
    """Reflect about ``|Pi> = PREP_out|0> (x) |0>_flag (x) |0>_inner``.

    The two-body sandwich ``V R V`` leaks the inner register, so the walk's
    reflection must be ``I - 2|Pi><Pi|`` over the *whole* ancilla (outer +
    flag + inner), not just the outer index. ``PREP_out`` acts on the outer
    slice; the zero-reflection covers the entire ancilla.
    """
    unprepare(ancilla[0:n_out], outer_angles)
    reflect_about_zero(ancilla)
    prepare(ancilla[0:n_out], outer_angles)


@cudaq.kernel
def _controlled_reflect_about_prepare_full(control_and_ancilla: cudaq.qview,
                                           outer_angles: list[float],
                                           n_out: int):
    """``_reflect_about_prepare_full`` controlled by qubit 0.

    The uncontrolled ``PREP_out`` pair wraps the controlled full-ancilla
    zero-reflection, so the whole reflection is the identity at control |0>.
    """
    unprepare(control_and_ancilla[1:1 + n_out], outer_angles)
    controlled_reflect_about_zero(control_and_ancilla)
    prepare(control_and_ancilla[1:1 + n_out], outer_angles)


def _sign(value: float) -> int:
    return -1 if value < 0.0 else 1


class DoubleFactorizedSquaredEncoding:
    """Squared-oracle double-factorized block encoding (von Burg ``burg`` norm).

    Block-encodes the *same* dense electronic Hamiltonian as
    :class:`cudaq_algorithms.df_encoding.DoubleFactorizedEncoding` (shared
    ``kappa``/one-body/constant machinery), but realises the two-body part as a
    coherent sum of squared one-body operators, giving the smaller ``burg``
    one-norm ``sum_k |F_k| + 1/4 sum_{t,i} |lambda^t_i| (sum_k |v^t_ki|)^2``.
    Both encodings therefore plug interchangeably into ``Walk`` / ``QSVT``.

    Parameters mirror ``DoubleFactorizedEncoding``, plus ``encode_constant``.
    With ``encode_constant=True`` (default) the identity term is block-encoded
    like every other slot, so ``<0|U|0> = H / alpha`` exactly. The squared
    construction, however, parks the Chebyshev ``-I`` offsets into that identity
    slot, and an LCU one-norm cannot cancel them against the slots' own ``-I``
    content, so ``alpha`` is inflated by ``|constant_term|`` and only ties the
    ZZ-word encoding on the *total* one-norm.

    With ``encode_constant=False`` the identity slot is dropped: the block
    encodes ``H - constant_term * I`` and ``alpha`` collapses to the published
    von Burg one-norm ``sum_k |F_k| + 1/4 sum_{t,i} |lambda^t_i| (sum_k
    |v^t_ki|)^2`` -- strictly below the ZZ-word encoding's one-norm. Add
    ``constant_term`` back to recovered energies (the standard nuclear-offset
    convention). This is the query-cost-relevant mode.

    Satisfies the ``BlockEncoding`` protocol except ``select_observable``
    (LCU-specific).
    """

    def __init__(self,
                 one_body: ArrayLike,
                 two_body: Union[DoubleFactorization, ArrayLike],
                 *,
                 scalar_offset: float = 0.0,
                 encode_constant: bool = True,
                 coefficient_threshold: float = 1e-12) -> None:
        self._encode_constant = bool(encode_constant)
        # Reuse the existing encoding VERBATIM for input validation and the
        # kappa / one-body / constant / singles handling. ``base.constant_term``
        # is the existing ``_constant`` (df_const); ``base.factorization`` is
        # the (possibly truncated) factorization actually encoded.
        base = DoubleFactorizedEncoding(
            one_body,
            two_body,
            scalar_offset=scalar_offset,
            coefficient_threshold=coefficient_threshold)
        factorization = base.factorization
        n = base.num_spatial_orbitals
        self._num_spatial = n
        self._num_system = 2 * n
        self._factorization = factorization

        # Recompute kappa exactly as df_encoding does, to expose the eigenvalues
        # F and the kappa eigenframe G_0 (the existing class keeps them private).
        kappa = np.asarray(one_body, dtype=float).copy()
        for rotation, core in zip(factorization.leaf_rotations,
                                  factorization.leaf_cores):
            absorbed = (0.5 * (core.sum(axis=1) + core.sum(axis=0)) -
                        0.5 * np.diag(core))
            kappa += (rotation * absorbed) @ rotation.T
        eigenvalues, eigenvectors = np.linalg.eigh(kappa)

        n_in = max(1, (2 * n - 1).bit_length())
        self._num_inner = n_in

        # const_burg = df_const + 1/4 sum_{t,i} lambda^t_i ((S^t_i)^2 - 1):
        # the squared slots carry identity content 1/4 lambda S^2 where the
        # existing ZZ two-body carried 1/4 lambda (= 1/4 trace); the (S^2 - 1)
        # correction makes the reconstructed block equal H exactly (dense-
        # validated to 1e-11). NOTE: this differs from a naive "+ 1/4 lambda
        # S^2" by the -1/4 sum lambda term, a pure identity offset a standalone
        # reconstruction otherwise double-counts.
        const_correction = 0.0
        g0_sweep = _givens_sweep(eigenvectors)

        slot_signs: list = []
        weights: list = []
        frame_sweeps: list = []
        inner_prep_angles: list = []
        inner_signs: list = []

        for rotation, core in zip(factorization.leaf_rotations,
                                  factorization.leaf_cores):
            lam, vecs = np.linalg.eigh(np.asarray(core, dtype=float))
            gt_sweep = _givens_sweep(rotation)
            for i in range(n):
                v = vecs[:, i]
                column_abs_sum = float(np.sum(np.abs(v)))
                weight = 0.25 * abs(float(lam[i])) * column_abs_sum**2
                const_correction += 0.25 * float(
                    lam[i]) * (column_abs_sum**2 - 1.0)
                if weight < coefficient_threshold:
                    continue
                slot_signs.append(-_sign(float(lam[i])))
                frame_sweeps.append(gt_sweep)
                weights.append(weight)
                probs = [0.0] * (1 << n_in)
                signs = [1] * (1 << n_in)
                for k in range(n):
                    p = 0.5 * abs(float(v[k])) / column_abs_sum
                    s = -_sign(float(v[k]))
                    for sigma in range(2):
                        probs[2 * k + sigma] = p
                        signs[2 * k + sigma] = s
                inner_prep_angles.extend(_prepare_angles(probs))
                inner_signs.extend(signs)

        const_burg = base.constant_term + const_correction
        self._constant = const_burg

        one_body_specs: list = []
        for k in range(n):
            f_k = float(eigenvalues[k])
            if abs(0.5 * f_k) >= coefficient_threshold:
                for sigma in range(2):
                    one_body_specs.append((2 * k + sigma, f_k))

        ordered_signs: list = []
        ordered_kinds: list = []
        ordered_targets: list = []
        ordered_frames: list = []
        ordered_weights: list = []

        # The identity slot is only block-encoded when requested. Dropping it
        # (``encode_constant=False``) makes the block encode ``H - const_burg I``
        # and collapses ``alpha`` to the published von Burg one-norm; the caller
        # adds ``constant_term`` back to recovered energies.
        if self._encode_constant and abs(const_burg) >= coefficient_threshold:
            ordered_signs.append(_sign(const_burg))
            ordered_kinds.append(0)
            ordered_targets.append(0)
            ordered_frames.append([])
            ordered_weights.append(abs(const_burg))

        for target, f_k in one_body_specs:
            ordered_signs.append(-_sign(f_k))
            ordered_kinds.append(1)
            ordered_targets.append(target)
            ordered_frames.append(g0_sweep)
            ordered_weights.append(0.5 * abs(f_k))

        for j in range(len(weights)):
            ordered_signs.append(slot_signs[j])
            ordered_kinds.append(2)
            ordered_targets.append(0)
            ordered_frames.append(frame_sweeps[j])
            ordered_weights.append(weights[j])

        if not ordered_weights:
            raise ValueError("hamiltonian has no retained terms")

        num_slots = len(ordered_weights)
        self._num_slots = num_slots
        n_out = max(1, (num_slots - 1).bit_length())
        self._num_outer = n_out
        self._num_ancilla = n_out + 1 + n_in

        self._alpha = float(sum(ordered_weights))

        probabilities = [w / self._alpha for w in ordered_weights]
        probabilities += [0.0] * ((1 << n_out) - num_slots)
        self._outer_angles = _prepare_angles(probabilities)

        slot_outer_controls: list = []
        for index in range(num_slots):
            for b in range(n_out):
                slot_outer_controls.append((index >> (n_out - 1 - b)) & 1)

        frame_counts: list = []
        frame_orbitals: list = []
        frame_angles: list = []
        for sweep in ordered_frames:
            frame_counts.append(len(sweep))
            for p, theta in sweep:
                frame_orbitals.append(int(p))
                frame_angles.append(float(theta))
        if not frame_orbitals:
            frame_orbitals = [0]
            frame_angles = [0.0]

        # Pad to a full slot's worth even when there are no two-body slots:
        # the (never-taken) two-body branch is still compiled, and CUDA-Q's
        # static bounds check rejects indexing a shorter literal (#4847-adjacent).
        if len(inner_prep_angles) < ((1 << n_in) - 1):
            inner_prep_angles = inner_prep_angles + [0.0] * ((
                (1 << n_in) - 1) - len(inner_prep_angles))
        if len(inner_signs) < (1 << n_in):
            inner_signs = inner_signs + [1] * ((1 << n_in) - len(inner_signs))

        self._slot_signs = ordered_signs
        self._slot_kinds = ordered_kinds
        self._slot_targets = ordered_targets
        self._slot_outer_controls = slot_outer_controls
        self._frame_counts = frame_counts
        self._frame_orbitals = frame_orbitals
        self._frame_angles = frame_angles
        self._inner_prep_angles = inner_prep_angles
        self._inner_signs = inner_signs

        self._select_kernel = self._build_select(0)
        self._controlled_select_kernel = self._build_select(1)

    def _build_select(self, off: int) -> Kernel:
        n_out = self._num_outer
        n_in = self._num_inner
        two_n = self._num_system
        num_slots = self._num_slots
        prep_count = (1 << n_in) - 1
        pow2_in = 1 << n_in
        flag_lo = off + n_out

        signs = self._slot_signs
        kinds = self._slot_kinds
        targets = self._slot_targets
        outer_controls = self._slot_outer_controls
        frame_counts = self._frame_counts
        frame_orbitals = self._frame_orbitals
        frame_angles = self._frame_angles
        inner_prep_angles = self._inner_prep_angles
        inner_signs = self._inner_signs

        @cudaq.kernel
        def _select(ancilla: cudaq.qview, system: cudaq.qview):
            frame_ptr = 0
            prep_ptr = 0
            sign_ptr = 0
            for s in range(num_slots):
                for b in range(n_out):
                    if outer_controls[s * n_out + b] == 0:
                        x(ancilla[off + b])
                x.ctrl(ancilla[0:flag_lo], ancilla[flag_lo])

                fc = frame_counts[s]
                for r in range(fc):
                    p = frame_orbitals[frame_ptr + r]
                    th = frame_angles[frame_ptr + r]
                    givens_frame(system[2 * p:2 * p + 3])
                    ry.ctrl(-2.0 * th, ancilla[flag_lo], system[2 * p],
                            system[2 * p + 2])
                    givens_frame_adj(system[2 * p:2 * p + 3])
                    givens_frame(system[2 * p + 1:2 * p + 4])
                    ry.ctrl(-2.0 * th, ancilla[flag_lo], system[2 * p + 1],
                            system[2 * p + 3])
                    givens_frame_adj(system[2 * p + 1:2 * p + 4])

                if kinds[s] == 2:
                    inner = ancilla[flag_lo + 1:flag_lo + 1 + n_in]
                    _offset_prepare(inner, inner_prep_angles, prep_ptr)
                    _inner_select(ancilla, system, inner_signs, sign_ptr,
                                  flag_lo, n_in, two_n)
                    _offset_unprepare(inner, inner_prep_angles, prep_ptr,
                                      prep_count)
                    for q in range(n_in):
                        x(ancilla[flag_lo + 1 + q])
                    r1.ctrl(_PI, ancilla[flag_lo:flag_lo + n_in],
                            ancilla[flag_lo + n_in])
                    for q in range(n_in):
                        x(ancilla[flag_lo + 1 + q])
                    _offset_prepare(inner, inner_prep_angles, prep_ptr)
                    _inner_select(ancilla, system, inner_signs, sign_ptr,
                                  flag_lo, n_in, two_n)
                    _offset_unprepare(inner, inner_prep_angles, prep_ptr,
                                      prep_count)
                    prep_ptr += prep_count
                    sign_ptr += pow2_in

                if kinds[s] == 1:
                    z.ctrl(ancilla[flag_lo], system[targets[s]])

                for r in range(fc):
                    idx = fc - 1 - r
                    p = frame_orbitals[frame_ptr + idx]
                    th = frame_angles[frame_ptr + idx]
                    givens_frame(system[2 * p:2 * p + 3])
                    ry.ctrl(2.0 * th, ancilla[flag_lo], system[2 * p],
                            system[2 * p + 2])
                    givens_frame_adj(system[2 * p:2 * p + 3])
                    givens_frame(system[2 * p + 1:2 * p + 4])
                    ry.ctrl(2.0 * th, ancilla[flag_lo], system[2 * p + 1],
                            system[2 * p + 3])
                    givens_frame_adj(system[2 * p + 1:2 * p + 4])
                frame_ptr += fc

                if signs[s] < 0:
                    z(ancilla[flag_lo])

                x.ctrl(ancilla[0:flag_lo], ancilla[flag_lo])
                for b in range(n_out):
                    if outer_controls[s * n_out + b] == 0:
                        x(ancilla[off + b])

        return _select

    @property
    def num_system(self) -> int:
        return self._num_system

    @property
    def num_ancilla(self) -> int:
        return self._num_ancilla

    @property
    def num_spatial_orbitals(self) -> int:
        return self._num_spatial

    @property
    def num_slots(self) -> int:
        return self._num_slots

    @property
    def alpha(self) -> float:
        """Block-encoding one-norm (``<0|U|0> = (H - offset) / alpha``).

        With ``encode_constant=False`` this is the published von Burg one-norm
        and ``offset = constant_term``; with ``encode_constant=True`` it is
        inflated by ``|constant_term|`` and ``offset = 0``.
        """
        return self._alpha

    @property
    def constant_term(self) -> float:
        """The identity offset ``c`` of the encoding.

        When ``encode_constant=False`` the block encodes ``H - c * I``; add
        ``c`` back to recovered energies. When ``encode_constant=True`` the
        identity is inside the block (``c`` is carried as its own slot).
        """
        return self._constant

    @property
    def encode_constant(self) -> bool:
        return self._encode_constant

    @property
    def factorization(self) -> DoubleFactorization:
        return self._factorization

    def __repr__(self) -> str:
        return (f"DoubleFactorizedSquaredEncoding(slots={self._num_slots}, "
                f"system_qubits={self._num_system}, "
                f"ancilla_qubits={self._num_ancilla}, "
                f"encode_constant={self._encode_constant}, "
                f"alpha={self._alpha:.6g})")

    def encode_kernel(self, state_prep: Kernel = None) -> Kernel:
        outer_angles = self._outer_angles
        select = self._select_kernel
        n_out = self._num_outer
        n_anc = self._num_ancilla
        n_sys = self._num_system

        if state_prep is not None:

            @cudaq.kernel
            def prep_encoded():
                system = cudaq.qvector(n_sys)
                state_prep(system)
                ancilla = cudaq.qvector(n_anc)
                prepare(ancilla[0:n_out], outer_angles)
                select(ancilla, system)
                unprepare(ancilla[0:n_out], outer_angles)

            return prep_encoded

        @cudaq.kernel
        def encoded(state: cudaq.State):
            system = cudaq.qvector(state)
            ancilla = cudaq.qvector(n_anc)
            prepare(ancilla[0:n_out], outer_angles)
            select(ancilla, system)
            unprepare(ancilla[0:n_out], outer_angles)

        return encoded

    def walk_kernel(self, power: int = 1, state_prep: Kernel = None) -> Kernel:
        outer_angles = self._outer_angles
        select = self._select_kernel
        n_out = self._num_outer
        n_anc = self._num_ancilla
        n_sys = self._num_system
        steps = _validate_power(power)

        if state_prep is not None:

            @cudaq.kernel
            def prep_walked():
                system = cudaq.qvector(n_sys)
                state_prep(system)
                ancilla = cudaq.qvector(n_anc)
                prepare(ancilla[0:n_out], outer_angles)
                for _ in range(steps):
                    select(ancilla, system)
                    _reflect_about_prepare_full(ancilla, outer_angles, n_out)
                unprepare(ancilla[0:n_out], outer_angles)

            return prep_walked

        @cudaq.kernel
        def walked(state: cudaq.State):
            system = cudaq.qvector(state)
            ancilla = cudaq.qvector(n_anc)
            prepare(ancilla[0:n_out], outer_angles)
            for _ in range(steps):
                select(ancilla, system)
                _reflect_about_prepare_full(ancilla, outer_angles, n_out)
            unprepare(ancilla[0:n_out], outer_angles)

        return walked

    def prepare_kernel(self) -> Kernel:
        outer_angles = self._outer_angles
        n_out = self._num_outer

        @cudaq.kernel
        def prepare_ancilla(ancilla: cudaq.qview):
            prepare(ancilla[0:n_out], outer_angles)

        return prepare_ancilla

    def unprepare_kernel(self) -> Kernel:
        outer_angles = self._outer_angles
        n_out = self._num_outer

        @cudaq.kernel
        def unprepare_ancilla(ancilla: cudaq.qview):
            unprepare(ancilla[0:n_out], outer_angles)

        return unprepare_ancilla

    def apply_kernel(self) -> Kernel:
        outer_angles = self._outer_angles
        select = self._select_kernel
        n_out = self._num_outer

        @cudaq.kernel
        def apply_encoding(ancilla: cudaq.qview, system: cudaq.qview):
            prepare(ancilla[0:n_out], outer_angles)
            select(ancilla, system)
            unprepare(ancilla[0:n_out], outer_angles)

        return apply_encoding

    def controlled_apply_kernel(self) -> Kernel:
        outer_angles = self._outer_angles
        controlled_select = self._controlled_select_kernel
        n_out = self._num_outer

        @cudaq.kernel
        def apply_controlled(control_and_ancilla: cudaq.qview,
                             system: cudaq.qview):
            prepare(control_and_ancilla[1:1 + n_out], outer_angles)
            controlled_select(control_and_ancilla, system)
            unprepare(control_and_ancilla[1:1 + n_out], outer_angles)

        return apply_controlled

    def walk_step_kernel(self) -> Kernel:
        outer_angles = self._outer_angles
        select = self._select_kernel
        n_out = self._num_outer

        @cudaq.kernel
        def walk_step(ancilla: cudaq.qview, system: cudaq.qview):
            select(ancilla, system)
            _reflect_about_prepare_full(ancilla, outer_angles, n_out)

        return walk_step

    def adjoint_walk_step_kernel(self) -> Kernel:
        outer_angles = self._outer_angles
        select = self._select_kernel
        n_out = self._num_outer

        @cudaq.kernel
        def adjoint_walk_step(ancilla: cudaq.qview, system: cudaq.qview):
            _reflect_about_prepare_full(ancilla, outer_angles, n_out)
            select(ancilla, system)

        return adjoint_walk_step

    def controlled_walk_step_kernel(self) -> Kernel:
        outer_angles = self._outer_angles
        controlled_select = self._controlled_select_kernel
        n_out = self._num_outer

        @cudaq.kernel
        def controlled_walk_step(control_and_ancilla: cudaq.qview,
                                 system: cudaq.qview):
            controlled_select(control_and_ancilla, system)
            _controlled_reflect_about_prepare_full(control_and_ancilla,
                                                   outer_angles, n_out)

        return controlled_walk_step

    def controlled_adjoint_walk_step_kernel(self) -> Kernel:
        outer_angles = self._outer_angles
        controlled_select = self._controlled_select_kernel
        n_out = self._num_outer

        @cudaq.kernel
        def controlled_adjoint_walk_step(control_and_ancilla: cudaq.qview,
                                         system: cudaq.qview):
            _controlled_reflect_about_prepare_full(control_and_ancilla,
                                                   outer_angles, n_out)
            controlled_select(control_and_ancilla, system)

        return controlled_adjoint_walk_step

    def select_observable(self) -> Any:
        raise NotImplementedError(
            "DoubleFactorizedSquaredEncoding does not provide "
            "select_observable: its SELECT terms are frame-rotated squared "
            "operators, not computational-frame Pauli words; odd Chebyshev "
            "moments via the observable trick are unavailable for this "
            "encoding")
