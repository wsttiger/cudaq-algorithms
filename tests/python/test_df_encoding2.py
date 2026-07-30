# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                        #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""Validation for the squared-oracle DF primitives (``df_squared_encoding``).

Phase A dense-validates the register-driven ("programmable") Givens against
the ``exp_pauli`` XZY/YZX hop it must reproduce (both spin-lift slices), plus
its hand-written inverse. Phase 0 checks the pure-NumPy host helpers
(rank-one factoring, one-column partial sweep). The full encoding is not yet
assembled (see the module docstring / task report), so there are no
end-to-end block==H/alpha tests here yet.
"""

import math

import numpy as np
import pytest

import cudaq

from cudaq_algorithms import df_squared_encoding as dsq
from cudaq_algorithms.common_kernels import state_from

FULL_TURN = 2.0 * math.pi


def _angle_bits(code: int, nbits: int) -> list[int]:
    """Bit list b_j (coefficient 2^{-(j+1)}) for theta = FULL_TURN*code/2^nbits."""
    return [(code >> (nbits - 1 - j)) & 1 for j in range(nbits)]


def _system_amps(state, code: int, nbits: int, n_sys: int) -> np.ndarray:
    """Extract the n_sys-qubit amplitudes with the angle register fixed to code.

    System qubits are allocated first (indices 0..n_sys-1); the angle
    register occupies the higher bits. Bit ``j`` (fraction weight
    ``2^{-(j+1)}``) sits on angle qubit ``j``, so the register's little-endian
    integer value is ``sum_j bit_j 2^j`` -- the bit-reversal of ``code``.
    """
    out = np.array(state)
    bits = _angle_bits(code, nbits)
    reg_value = sum(bits[j] << j for j in range(nbits))
    base = reg_value << n_sys
    return out[base:base + (1 << n_sys)]


def _run_programmable(input_ket: np.ndarray, code: int, nbits: int, n_sys: int,
                      slice_start: int) -> np.ndarray:
    bits = _angle_bits(code, nbits)

    @cudaq.kernel
    def circuit(state: cudaq.State):
        system = cudaq.qvector(state)
        angle = cudaq.qvector(nbits)
        for j in range(nbits):
            if bits[j] == 1:
                x(angle[j])
        dsq.programmable_givens(system[slice_start:slice_start + 3], angle,
                                FULL_TURN)

    return _system_amps(cudaq.get_state(circuit, state_from(input_ket)), code,
                        nbits, n_sys)


def _run_exp_pauli(input_ket: np.ndarray, theta: float, n_sys: int,
                   slice_start: int) -> np.ndarray:

    @cudaq.kernel
    def circuit(state: cudaq.State):
        system = cudaq.qvector(state)
        s = system[slice_start:slice_start + 3]
        exp_pauli(0.5 * theta, s, "XZY")
        exp_pauli(-0.5 * theta, s, "YZX")

    return np.array(cudaq.get_state(circuit, state_from(input_ket)))


def _random_ket(dim: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    ket = rng.normal(size=dim) + 1.0j * rng.normal(size=dim)
    return (ket / np.linalg.norm(ket)).astype(np.complex128)


# ----------------------------------------------------------------------
# Phase A: programmable Givens == exp_pauli XZY/YZX hop
# ----------------------------------------------------------------------


@pytest.mark.parametrize("code", [0, 1, 37, 128, 200])
@pytest.mark.parametrize("slice_start", [0, 1])
def test_programmable_givens_matches_exp_pauli_hop(code, slice_start):
    # A 4-qubit system exercises both spin-lift slices ([0:3] and [1:4]),
    # the two contiguous three-qubit rotations of one spatial Givens.
    nbits = 8
    n_sys = 4
    theta = FULL_TURN * code / (1 << nbits)
    ket = _random_ket(1 << n_sys, seed=100 + code + slice_start)
    got = _run_programmable(ket, code, nbits, n_sys, slice_start)
    ref = _run_exp_pauli(ket, theta, n_sys, slice_start)
    np.testing.assert_allclose(got, ref, atol=1e-12)


def test_programmable_givens_adjoint_inverts():
    nbits = 8
    n_sys = 3
    code = 173
    ket = _random_ket(1 << n_sys, seed=7)
    bits = _angle_bits(code, nbits)

    @cudaq.kernel
    def circuit(state: cudaq.State):
        system = cudaq.qvector(state)
        angle = cudaq.qvector(nbits)
        for j in range(nbits):
            if bits[j] == 1:
                x(angle[j])
        dsq.programmable_givens(system, angle, FULL_TURN)
        dsq.programmable_givens_adj(system, angle, FULL_TURN)

    out = _system_amps(cudaq.get_state(circuit, state_from(ket)), code, nbits,
                       n_sys)
    np.testing.assert_allclose(out, ket, atol=1e-12)


def test_programmable_givens_zero_angle_is_identity():
    nbits = 6
    n_sys = 3
    ket = _random_ket(1 << n_sys, seed=3)
    got = _run_programmable(ket, 0, nbits, n_sys, 0)
    np.testing.assert_allclose(got, ket, atol=1e-12)


# ----------------------------------------------------------------------
# Phase 0: host helpers
# ----------------------------------------------------------------------


@pytest.mark.parametrize("n", [2, 3, 4])
def test_one_column_sweep_maps_to_e0(n):
    rng = np.random.default_rng(n)
    v = rng.normal(size=n)
    v = v / np.linalg.norm(v)
    angles = dsq.one_column_sweep(v)
    assert len(angles) == n - 1
    # Apply the plane rotations (row n-1..1) and confirm v -> +- e_0.
    work = v.copy()
    for idx, row in enumerate(range(n - 1, 0, -1)):
        theta = angles[idx]
        c, s = math.cos(theta), math.sin(theta)
        u, l = work[row - 1], work[row]
        work[row - 1] = c * u + s * l
        work[row] = -s * u + c * l
    assert abs(abs(work[0]) - 1.0) < 1e-12
    assert np.linalg.norm(work[1:]) < 1e-12


def test_rank_one_leaf_slots_reconstructs_core():
    rng = np.random.default_rng(11)
    n = 3
    v = rng.normal(size=n)
    lam = -0.83
    core = lam * np.outer(v, v)  # a true rank-one X-DF leaf
    rotation, _ = np.linalg.qr(rng.normal(size=(n, n)))
    slots = dsq.rank_one_leaf_slots(rotation, core)
    assert len(slots) == 1  # rank one -> one slot
    slot = slots[0]
    # W_i = sqrt(|lambda|) v_i ; s_i lambda... reconstruct core = s |lambda| v v^T.
    w = math.sqrt(abs(slot["eigenvalue"])) * slot["vector"]
    recon = slot["sign"] * np.outer(w, w)
    np.testing.assert_allclose(recon, core, atol=1e-12)
    # burg weight matches the closed form 1/4 |eigenvalue| S^2 (the eigenvalue
    # absorbs ||v||^2 since the test's v is unnormalised; the eigenvector is
    # unit, so S is its column-abs-sum).
    S = slot["column_abs_sum"]
    assert slot["burg_weight"] == pytest.approx(0.25 *
                                                abs(slot["eigenvalue"]) * S**2,
                                                rel=1e-12)


def test_quantize_angle_resolution_scaling():
    rng = np.random.default_rng(5)
    thetas = rng.uniform(0.0, FULL_TURN, size=64)
    prev = None
    for bits in (6, 8, 10, 12):
        err = max(
            abs(((dsq.quantize_angle(t, bits, FULL_TURN) - t + math.pi) %
                 FULL_TURN) - math.pi) for t in thetas)
        assert err <= FULL_TURN / (1 << bits) / 2 + 1e-15
        if prev is not None:
            assert err <= prev  # finer grid never increases the worst error
        prev = err
