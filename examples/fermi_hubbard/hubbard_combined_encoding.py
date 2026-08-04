# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                        #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""Fermi-Hubbard via a combined block encoding: T in momentum space + U in real space.

The 2D Fermi-Hubbard Hamiltonian splits into two pieces that are each
diagonal in the *right* basis::

    H = T + U,   T = -t sum_<ij>,s (a+_is a_js + h.c.),   U = u sum_i n_iu n_id

* ``T`` is diagonal in momentum space: conjugating by the fermionic
  Fourier transform -- here the real orthogonal (standing-wave) eigenbasis
  of the one-particle hopping matrix, compiled to the same adjacent
  Givens-rotation networks the library's encodings use -- turns it into
  ``sum_k eps(k) n_k``, a sum of single-qubit Z words.
* ``U`` is diagonal in real space already: ``n_iu n_id`` is Z words in the
  computational basis.

Each piece is block-encoded where it is diagonal (``PauliLCU`` under the
hood; ``T`` gets the uncontrolled basis-change conjugation), and a **sum
combinator** aggregates the two encodings with one extra combining qubit:
PREPARE it with amplitudes weighted by the two subnormalizations, apply
each encoding controlled on its value, and unprepare. The block is then
``(T + U) / (alpha_T + alpha_U)``, and the combined object satisfies the
``BlockEncoding`` protocol -- ``Walk`` consumes it unchanged.

This T/U split with the Fourier basis change is the structure exploited in
Earl Campbell's Hubbard resource analysis (Early fault-tolerant simulations
of the Hubbard model, arXiv:2012.09238), there in split-operator Trotter
form; this example realizes the same decomposition at the block-encoding
level. Everything here is user-level code against the public API -- the
``RotatedEncoding`` and ``SumEncoding`` combinators below are the "bring
your own encoding" pattern applied to *composition*.

    python3 hubbard_combined_encoding.py [--lx 2] [--ly 2] [--t 1.0] [--u 4.0]

Defaults to the 2x2 lattice (8 system qubits); every claim is verified
against an independent sparse Jordan-Wigner reference. Runs on the CPU
statevector simulator; no compiled extension needed.
"""
from __future__ import annotations

import argparse
import math
import sys

import numpy as np
import scipy.sparse as sp

import cudaq

from cudaq_algorithms import BlockEncoding, PauliLCU, Walk, state_from
from cudaq_algorithms.common_kernels import (controlled_reflect_about_zero,
                                             reflect_about_zero)

cudaq.set_target("qpp-cpu")  # fp64: the checks below assert to ~1e-10

# ----------------------------------------------------------------------
# The lattice: one-particle hopping matrix and the interaction words.
# Spin orbitals are interleaved (2p up, 2p + 1 down), the library's
# Jordan-Wigner convention.
# ----------------------------------------------------------------------


def hopping_bonds(lx: int, ly: int) -> list[tuple[int, int]]:
    """Unique nearest-neighbor bonds of the lx x ly periodic lattice.

    Wrap-around bonds that coincide with an existing bond (length-2
    dimensions) are deduplicated, so the 2x2 lattice is the 4-site ring.
    """

    def site(ix, iy):
        return ix % lx + lx * (iy % ly)

    bonds = set()
    for iy in range(ly):
        for ix in range(lx):
            if lx > 1:
                bonds.add(tuple(sorted((site(ix, iy), site(ix + 1, iy)))))
            if ly > 1:
                bonds.add(tuple(sorted((site(ix, iy), site(ix, iy + 1)))))
    return sorted(bonds)


def hopping_matrix(lx: int, ly: int, t: float) -> np.ndarray:
    n = lx * ly
    matrix = np.zeros((n, n))
    for p, q in hopping_bonds(lx, ly):
        matrix[p, q] = matrix[q, p] = -t
    return matrix


def givens_sweep(orthogonal: np.ndarray) -> list[tuple[int, float]]:
    """Adjacent-Givens decomposition of an orthogonal matrix.

    Same construction as the library's double-factorized encoding: plane
    rotations ``(p, theta)`` on rows ``(p, p + 1)`` reducing the matrix to
    a diagonal of signs (irrelevant here -- every encoded term is a number
    operator, invariant under per-mode sign flips).
    """
    work = np.array(orthogonal, dtype=float, copy=True)
    n = work.shape[0]
    rotations: list[tuple[int, float]] = []
    for column in range(n):
        for row in range(n - 1, column, -1):
            upper, lower = work[row - 1, column], work[row, column]
            if abs(lower) > 1e-14:
                theta = math.atan2(lower, upper)
                c, s = math.cos(theta), math.sin(theta)
                rotation = np.array([[c, s], [-s, c]])
                work[row - 1:row + 1, :] = rotation @ work[row - 1:row + 1, :]
                rotations.append((row - 1, theta))
    return rotations


def _z_word(num_qubits: int, *qubits: int) -> str:
    word = ["I"] * num_qubits
    for q in qubits:
        word[q] = "Z"
    return "".join(word)


def momentum_diagonal_words(eigenvalues: np.ndarray,
                            num_qubits: int) -> dict[str, float]:
    """T in its eigenbasis: sum_m eps_m n_m per spin, n = (I - Z)/2."""
    words: dict[str, float] = {}
    constant = float(np.sum(eigenvalues))  # eps_m/2 over both spins
    if abs(constant) > 1e-12:
        words["I" * num_qubits] = constant
    for m, eps in enumerate(eigenvalues):
        if abs(eps) > 1e-12:
            for spin in range(2):
                words[_z_word(num_qubits, 2 * m + spin)] = -0.5 * float(eps)
    return words


def interaction_words(num_sites: int, u: float,
                      num_qubits: int) -> dict[str, float]:
    """U = u sum_i n_iu n_id = u/4 sum_i (I - Z_2i - Z_2i+1 + Z_2i Z_2i+1)."""
    words: dict[str, float] = {"I" * num_qubits: 0.25 * u * num_sites}
    for i in range(num_sites):
        up, down = 2 * i, 2 * i + 1
        words[_z_word(num_qubits,
                      up)] = words.get(_z_word(num_qubits, up), 0.0) - 0.25 * u
        words[_z_word(num_qubits, down)] = -0.25 * u
        words[_z_word(num_qubits, up, down)] = 0.25 * u
    return words


# ----------------------------------------------------------------------
# Device kernel: adjacent spatial Givens rotations, lifted to both spin
# sectors (the library's frame-rotation pattern; zero-angle entries are
# identity, so padded lists are harmless).
# ----------------------------------------------------------------------


@cudaq.kernel
def apply_rotations(system: cudaq.qview, orbitals: list[int],
                    angles: list[float]):
    for i in range(len(orbitals)):
        p = orbitals[i]
        theta = angles[i]
        up = system[2 * p:2 * p + 3]
        exp_pauli(0.5 * theta, up, "XZY")
        exp_pauli(-0.5 * theta, up, "YZX")
        down = system[2 * p + 1:2 * p + 4]
        exp_pauli(0.5 * theta, down, "XZY")
        exp_pauli(-0.5 * theta, down, "YZX")


def _rotation_lists(sweep, reverse):
    if reverse:
        pairs = [(p, -theta) for p, theta in reversed(sweep)]
    else:
        pairs = list(sweep)
    if not pairs:
        pairs = [(0, 0.0)]  # empty lists cannot cross the kernel boundary
    return [p for p, _ in pairs], [theta for _, theta in pairs]


# ----------------------------------------------------------------------
# RotatedEncoding: conjugate an encoding by an uncontrolled basis change.
#
# The rotations act on the system register only, so they commute with
# every ancilla-side operation (PREPARE, reflections): each protocol
# kernel of the inner encoding is simply sandwiched between the entry and
# exit rotation programs. At control |0> the inner kernel is the
# identity, so the uncontrolled sandwich telescopes away -- the controlled
# variants come for free.
# ----------------------------------------------------------------------


class RotatedEncoding:
    """Block encoding of ``V H V^T`` given an encoding of ``H``.

    ``sweep`` is the adjacent-Givens program reducing the orthogonal
    one-particle matrix ``V`` (whose columns are the target modes) to a
    sign diagonal, as produced by :func:`givens_sweep`.
    """

    def __init__(self, inner, sweep):
        self._inner = inner
        self._enter = _rotation_lists(sweep, reverse=False)
        self._exit = _rotation_lists(sweep, reverse=True)

    @property
    def num_system(self) -> int:
        return self._inner.num_system

    @property
    def num_ancilla(self) -> int:
        return self._inner.num_ancilla

    @property
    def alpha(self) -> float:
        return self._inner.alpha

    def prepare_kernel(self):
        return self._inner.prepare_kernel()

    def unprepare_kernel(self):
        return self._inner.unprepare_kernel()

    def _sandwich(self, inner_kernel):
        enter_orbitals, enter_angles = self._enter
        exit_orbitals, exit_angles = self._exit

        @cudaq.kernel
        def rotated(ancilla: cudaq.qview, system: cudaq.qview):
            apply_rotations(system, enter_orbitals, enter_angles)
            inner_kernel(ancilla, system)
            apply_rotations(system, exit_orbitals, exit_angles)

        return rotated

    def apply_kernel(self):
        return self._sandwich(self._inner.apply_kernel())

    def controlled_apply_kernel(self):
        return self._sandwich(self._inner.controlled_apply_kernel())

    def walk_step_kernel(self):
        return self._sandwich(self._inner.walk_step_kernel())

    def adjoint_walk_step_kernel(self):
        return self._sandwich(self._inner.adjoint_walk_step_kernel())

    def controlled_walk_step_kernel(self):
        return self._sandwich(self._inner.controlled_walk_step_kernel())

    def controlled_adjoint_walk_step_kernel(self):
        return self._sandwich(
            self._inner.controlled_adjoint_walk_step_kernel())

    def select_observable(self):
        raise NotImplementedError(
            "rotated-frame terms are not computational-frame Pauli words")


# ----------------------------------------------------------------------
# SumEncoding: block-encode H_A + H_B from encodings of H_A and H_B.
#
# Ancilla layout: [combining qubit c][flag][shared pool]. PREPARE puts
# sqrt(alpha_A/alpha)|0> + sqrt(alpha_B/alpha)|1> on c; SELECT computes
# flag = (c == 0) (or (c == 1)) and lets the flag drive each
# sub-encoding's own controlled_apply on a contiguous [flag | pool]
# view; the pool is shared because each sub-encoding starts and returns
# its ancillas to |0>. The block is (H_A + H_B) / (alpha_A + alpha_B).
#
# SELECT is a product of two commuting Hermitian involutions (each a
# CX-frame conjugation of a controlled Hermitian involution), so the
# whole apply is Hermitian and self-inverse and the standard walk
# construction goes through -- with the reflection over the ENTIRE
# ancilla register (combining qubit + flag + pool).
# ----------------------------------------------------------------------


class SumEncoding:
    """The sum combinator over two ``BlockEncoding``s on one system."""

    def __init__(self, first, second):
        if first.num_system != second.num_system:
            raise ValueError(
                f"encodings act on different systems ({first.num_system} vs "
                f"{second.num_system} qubits)")
        self._a = first
        self._b = second
        self._alpha = float(first.alpha) + float(second.alpha)
        if not self._alpha > 0.0:
            raise ValueError("combined alpha must be positive")
        self._pool = max(first.num_ancilla, second.num_ancilla)
        # ry(theta)|0> = cos(theta/2)|0> + sin(theta/2)|1>: weight the
        # |1> branch (the second encoding) by alpha_B / alpha.
        self._theta = 2.0 * math.asin(math.sqrt(second.alpha / self._alpha))

    @property
    def num_system(self) -> int:
        return self._a.num_system

    @property
    def num_ancilla(self) -> int:
        return 2 + self._pool  # [c][flag][pool]

    @property
    def alpha(self) -> float:
        return self._alpha

    def prepare_kernel(self):
        theta = self._theta

        @cudaq.kernel
        def prepare(ancilla: cudaq.qview):
            ry(theta, ancilla[0])

        return prepare

    def unprepare_kernel(self):
        theta = self._theta

        @cudaq.kernel
        def unprepare(ancilla: cudaq.qview):
            ry(-theta, ancilla[0])

        return unprepare

    def _select_kernel(self):
        ctrl_a = self._a.controlled_apply_kernel()
        ctrl_b = self._b.controlled_apply_kernel()
        width_a = 1 + self._a.num_ancilla
        width_b = 1 + self._b.num_ancilla

        @cudaq.kernel
        def select(ancilla: cudaq.qview, system: cudaq.qview):
            # flag <- (c == 0); the flag drives encoding A.
            x(ancilla[0])
            x.ctrl(ancilla[0:1], ancilla[1])
            x(ancilla[0])
            ctrl_a(ancilla[1:1 + width_a], system)
            x(ancilla[0])
            x.ctrl(ancilla[0:1], ancilla[1])
            x(ancilla[0])
            # flag <- (c == 1); the flag drives encoding B.
            x.ctrl(ancilla[0:1], ancilla[1])
            ctrl_b(ancilla[1:1 + width_b], system)
            x.ctrl(ancilla[0:1], ancilla[1])

        return select

    def _controlled_select_kernel(self):
        ctrl_a = self._a.controlled_apply_kernel()
        ctrl_b = self._b.controlled_apply_kernel()
        width_a = 1 + self._a.num_ancilla
        width_b = 1 + self._b.num_ancilla

        @cudaq.kernel
        def controlled_select(control_and_ancilla: cudaq.qview,
                              system: cudaq.qview):
            # Layout [control][c][flag][pool]; flag <- control AND (c == 0).
            x(control_and_ancilla[1])
            x.ctrl(control_and_ancilla[0:2], control_and_ancilla[2])
            x(control_and_ancilla[1])
            ctrl_a(control_and_ancilla[2:2 + width_a], system)
            x(control_and_ancilla[1])
            x.ctrl(control_and_ancilla[0:2], control_and_ancilla[2])
            x(control_and_ancilla[1])
            # flag <- control AND (c == 1).
            x.ctrl(control_and_ancilla[0:2], control_and_ancilla[2])
            ctrl_b(control_and_ancilla[2:2 + width_b], system)
            x.ctrl(control_and_ancilla[0:2], control_and_ancilla[2])

        return controlled_select

    def apply_kernel(self):
        theta = self._theta
        select = self._select_kernel()

        @cudaq.kernel
        def apply(ancilla: cudaq.qview, system: cudaq.qview):
            ry(theta, ancilla[0])
            select(ancilla, system)
            ry(-theta, ancilla[0])

        return apply

    def controlled_apply_kernel(self):
        theta = self._theta
        controlled_select = self._controlled_select_kernel()

        @cudaq.kernel
        def controlled_apply(control_and_ancilla: cudaq.qview,
                             system: cudaq.qview):
            # The PREPARE pair on c stays uncontrolled: it cancels when
            # the control is |0> because the controlled SELECT is identity.
            ry(theta, control_and_ancilla[1])
            controlled_select(control_and_ancilla, system)
            ry(-theta, control_and_ancilla[1])

        return controlled_apply

    def walk_step_kernel(self):
        theta = self._theta
        select = self._select_kernel()

        @cudaq.kernel
        def walk(ancilla: cudaq.qview, system: cudaq.qview):
            select(ancilla, system)
            ry(-theta, ancilla[0])
            reflect_about_zero(ancilla)
            ry(theta, ancilla[0])

        return walk

    def adjoint_walk_step_kernel(self):
        theta = self._theta
        select = self._select_kernel()

        @cudaq.kernel
        def adjoint_walk(ancilla: cudaq.qview, system: cudaq.qview):
            ry(-theta, ancilla[0])
            reflect_about_zero(ancilla)
            ry(theta, ancilla[0])
            select(ancilla, system)

        return adjoint_walk

    def controlled_walk_step_kernel(self):
        theta = self._theta
        controlled_select = self._controlled_select_kernel()

        @cudaq.kernel
        def controlled_walk(control_and_ancilla: cudaq.qview,
                            system: cudaq.qview):
            controlled_select(control_and_ancilla, system)
            ry(-theta, control_and_ancilla[1])
            controlled_reflect_about_zero(control_and_ancilla)
            ry(theta, control_and_ancilla[1])

        return controlled_walk

    def controlled_adjoint_walk_step_kernel(self):
        theta = self._theta
        controlled_select = self._controlled_select_kernel()

        @cudaq.kernel
        def controlled_adjoint_walk(control_and_ancilla: cudaq.qview,
                                    system: cudaq.qview):
            ry(-theta, control_and_ancilla[1])
            controlled_reflect_about_zero(control_and_ancilla)
            ry(theta, control_and_ancilla[1])
            controlled_select(control_and_ancilla, system)

        return controlled_adjoint_walk

    def select_observable(self):
        raise NotImplementedError(
            "the summands' terms live in different frames; no single "
            "computational-frame Pauli observable exists")


# ----------------------------------------------------------------------
# Independent sparse Jordan-Wigner reference (interleaved spins, qubit 0
# least significant).
# ----------------------------------------------------------------------


def _sparse_annihilators(num_qubits: int) -> list[sp.csr_matrix]:
    z2 = sp.csr_matrix(np.diag([1.0, -1.0]))
    identity = sp.identity(2, format="csr")
    lowering = sp.csr_matrix(np.array([[0.0, 1.0], [0.0, 0.0]]))
    out = []
    for mode in range(num_qubits):
        ops = ([z2] * mode + [lowering] + [identity] *
               (num_qubits - mode - 1))[::-1]
        matrix = sp.identity(1, format="csr")
        for op in ops:
            matrix = sp.kron(matrix, op, format="csr")
        out.append(matrix)
    return out


def reference_hamiltonians(lx, ly, t, u):
    """Sparse (T, U) built directly from bonds and number operators."""
    n = lx * ly
    num_qubits = 2 * n
    lower = _sparse_annihilators(num_qubits)
    raise_ = [op.conj().T.tocsr() for op in lower]
    dim = 1 << num_qubits
    hopping = sp.csr_matrix((dim, dim), dtype=complex)
    for p, q in hopping_bonds(lx, ly):
        for spin in range(2):
            m1, m2 = 2 * p + spin, 2 * q + spin
            hopping = hopping - t * (raise_[m1] @ lower[m2] +
                                     raise_[m2] @ lower[m1])
    interaction = sp.csr_matrix((dim, dim), dtype=complex)
    for i in range(n):
        n_up = raise_[2 * i] @ lower[2 * i]
        n_down = raise_[2 * i + 1] @ lower[2 * i + 1]
        interaction = interaction + u * (n_up @ n_down)
    return hopping.tocsr(), interaction.tocsr()


def flat_hubbard_words(lx, ly, t, u) -> dict[str, float]:
    """The full Hamiltonian as computational-frame Pauli words (baseline).

    A hopping bond (p < q) per spin is (X Z..Z X + Y Z..Z Y)/2 on the
    Jordan-Wigner string between the two modes, coefficient -t/2 each.
    """
    n = lx * ly
    num_qubits = 2 * n
    words = dict(interaction_words(n, u, num_qubits))
    for p, q in hopping_bonds(lx, ly):
        for spin in range(2):
            m1, m2 = 2 * p + spin, 2 * q + spin
            for pauli in ("X", "Y"):
                word = ["I"] * num_qubits
                word[m1] = pauli
                word[m2] = pauli
                for z in range(m1 + 1, m2):
                    word[z] = "Z"
                key = "".join(word)
                words[key] = words.get(key, 0.0) - 0.5 * t
    return {w: c for w, c in words.items() if abs(c) > 1e-12}


def chebyshev_moment(h_scaled, ket, order):
    t_prev, t_cur = ket, h_scaled @ ket
    if order == 0:
        return float(np.real(ket.conj() @ t_prev))
    for _ in range(order - 1):
        t_prev, t_cur = t_cur, 2.0 * (h_scaled @ t_cur) - t_prev
    return float(np.real(ket.conj() @ t_cur))


def check(label: str, condition: bool):
    print(f"  [check] {label} ... {'OK' if condition else 'FAILED'}")
    if not condition:
        sys.exit(1)


def encoded_block(encoding, kernel, ket):
    out = np.array(cudaq.get_state(kernel, state_from(ket)))
    return out[:1 << encoding.num_system]


# ----------------------------------------------------------------------
# The story
# ----------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--lx", type=int, default=2)
    parser.add_argument("--ly", type=int, default=2)
    parser.add_argument("--t", type=float, default=1.0)
    parser.add_argument("--u", type=float, default=4.0)
    args = parser.parse_args()
    lx, ly, t, u = args.lx, args.ly, args.t, args.u

    n = lx * ly
    num_qubits = 2 * n
    print(f"=== {lx}x{ly} Fermi-Hubbard (t = {t}, u = {u}): "
          f"{n} sites -> {num_qubits} system qubits ===")

    # --- T in momentum space -------------------------------------------
    one_particle = hopping_matrix(lx, ly, t)
    eigenvalues, modes = np.linalg.eigh(one_particle)
    reconstruction = modes @ np.diag(eigenvalues) @ modes.T
    check("standing-wave eigenbasis reconstructs the hopping matrix",
          float(np.max(np.abs(reconstruction - one_particle))) < 1e-12)
    print("  dispersion eps(k):", np.round(eigenvalues, 6))

    sweep = givens_sweep(modes)
    diagonal_t = PauliLCU(momentum_diagonal_words(eigenvalues, num_qubits))
    hopping_encoding = RotatedEncoding(diagonal_t, sweep)

    interaction_encoding = PauliLCU(interaction_words(n, u, num_qubits))

    combined = SumEncoding(hopping_encoding, interaction_encoding)
    check("the combined encoding satisfies the BlockEncoding protocol",
          isinstance(combined, BlockEncoding))

    flat = PauliLCU(flat_hubbard_words(lx, ly, t, u))
    print(f"\n  alpha_T (momentum space) = {hopping_encoding.alpha:8.4f}  "
          f"({len(sweep)} Givens rotations, "
          f"{hopping_encoding.num_ancilla} ancillas)")
    print(f"  alpha_U (real space)     = {interaction_encoding.alpha:8.4f}  "
          f"({interaction_encoding.num_ancilla} ancillas)")
    print(f"  alpha (combined)         = {combined.alpha:8.4f}  "
          f"(= alpha_T + alpha_U; {combined.num_ancilla} ancillas: "
          f"combining qubit + flag + shared pool)")
    print(f"  alpha (flat PauliLCU)    = {flat.alpha:8.4f}  "
          f"({flat.num_terms} Pauli words, {flat.num_ancilla} ancillas)")

    total_qubits = num_qubits + combined.num_ancilla
    if total_qubits > 18:
        print(f"\n  Circuit checks skipped: {total_qubits} total qubits "
              f"(2^{total_qubits} amplitudes per state).")
        return
    print(f"\n  Circuits: {num_qubits} system + {combined.num_ancilla} "
          f"ancilla = {total_qubits} qubits")

    hopping_ref, interaction_ref = reference_hamiltonians(lx, ly, t, u)
    dim = 1 << num_qubits
    rng = np.random.default_rng(7)
    ket = rng.normal(size=dim) + 1.0j * rng.normal(size=dim)
    ket = (ket / np.linalg.norm(ket)).astype(np.complex128)

    # Each piece, in its own basis, against the sparse JW reference.
    block_t = encoded_block(hopping_encoding, _encode(hopping_encoding), ket)
    error_t = np.max(
        np.abs(block_t - (hopping_ref @ ket) / hopping_encoding.alpha))
    check(f"T block == T/alpha_T (max |diff| = {error_t:.2e})",
          float(error_t) < 1e-10)

    block_u = encoded_block(interaction_encoding,
                            _encode(interaction_encoding), ket)
    error_u = np.max(
        np.abs(block_u - (interaction_ref @ ket) / interaction_encoding.alpha))
    check(f"U block == U/alpha_U (max |diff| = {error_u:.2e})",
          float(error_u) < 1e-10)

    # The combinator: one extra PREPARE qubit glues them into (T + U)/alpha.
    hubbard = (hopping_ref + interaction_ref).tocsr()
    block_sum = encoded_block(combined, _encode(combined), ket)
    error_sum = np.max(np.abs(block_sum - (hubbard @ ket) / combined.alpha))
    check(f"combined block == (T + U)/alpha (max |diff| = {error_sum:.2e})",
          float(error_sum) < 1e-10)

    # Control conventions: identity at |0>, the block at |1>.
    at_zero, at_one = _controlled_blocks(combined, ket)
    check("controlled apply is identity at control |0>",
          float(np.max(np.abs(at_zero - ket))) < 1e-10)
    check(
        "controlled apply reproduces the block at control |1>",
        float(np.max(np.abs(at_one - (hubbard @ ket) / combined.alpha)))
        < 1e-10)

    # Controlled walk wiring: a step and its adjoint must round-trip to
    # the identity at either control value (the wiring bugs live here).
    for value, state in _controlled_roundtrips(combined, ket):
        check(f"controlled walk roundtrip is identity at control |{value}>",
              float(np.max(np.abs(state - ket))) < 1e-10)

    # The payoff: the combined object drops into Walk unchanged.
    print("\n  Even Chebyshev moments <T_k(H/alpha)> through Walk:")
    scaled = hubbard / combined.alpha
    for order in (0, 2, 4):
        measured = Walk(combined).moment(ket, order)
        exact = chebyshev_moment(scaled, ket, order)
        print(f"    T_{order}: measured {measured:+.8f}   "
              f"exact {exact:+.8f}")
        check(f"T_{order} moment matches the reference",
              abs(measured - exact) < 1e-8)


def _encode(encoding):
    apply = encoding.apply_kernel()
    n_anc = encoding.num_ancilla

    @cudaq.kernel
    def encoded(state: cudaq.State):
        system = cudaq.qvector(state)
        ancilla = cudaq.qvector(n_anc)
        apply(ancilla, system)

    return encoded


def _controlled_roundtrips(encoding, ket):
    step = encoding.controlled_walk_step_kernel()
    adjoint = encoding.controlled_adjoint_walk_step_kernel()
    n_anc = encoding.num_ancilla
    dim = 1 << encoding.num_system

    def run(flip):

        @cudaq.kernel
        def circuit(state: cudaq.State, value: bool):
            system = cudaq.qvector(state)
            control_and_ancilla = cudaq.qvector(1 + n_anc)
            if value:
                x(control_and_ancilla[0])
            step(control_and_ancilla, system)
            adjoint(control_and_ancilla, system)
            if value:
                x(control_and_ancilla[0])

        return np.array(cudaq.get_state(circuit, state_from(ket), flip))[:dim]

    return [(0, run(False)), (1, run(True))]


def _controlled_blocks(encoding, ket):
    controlled = encoding.controlled_apply_kernel()
    n_anc = encoding.num_ancilla
    dim = 1 << encoding.num_system

    def run(control_value):

        @cudaq.kernel
        def circuit(state: cudaq.State, flip: bool):
            system = cudaq.qvector(state)
            control_and_ancilla = cudaq.qvector(1 + n_anc)
            if flip:
                x(control_and_ancilla[0])
            controlled(control_and_ancilla, system)
            if flip:
                x(control_and_ancilla[0])

        return np.array(
            cudaq.get_state(circuit, state_from(ket), control_value))[:dim]

    return run(False), run(True)


if __name__ == "__main__":
    main()
