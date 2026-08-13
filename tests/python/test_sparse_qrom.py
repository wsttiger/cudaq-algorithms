# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for unary iteration and the unary-iteration QROM.

QROM correctness is pinned exhaustively: for random tables every address
is read out via basis-state extraction (`cudaq.get_state`), including the
out-of-range addresses of non-power-of-two tables (which must read zero).
The X-only lookup is exactly self-inverse — that property is tested
instead of a duplicate adjoint kernel. The unary-iteration ladder is
additionally pinned directly through a per-address phase kick, a Y-body
spot check (sign convention), the hand-written inverse, and the
externally controlled variant.
"""

import numpy as np
import pytest

import cudaq

from cudaq_algorithms.sparse import QROM, unary_iteration_kernels


def _basis(index: int, num_qubits: int) -> np.ndarray:
    ket = np.zeros(1 << num_qubits, dtype=np.complex128)
    ket[index] = 1.0
    return ket


# ----------------------------------------------------------------------
# QROM: exhaustive readout
# ----------------------------------------------------------------------
#
# Register layout in the harnesses: address at [0, L), ladder at [L, 2L),
# output at [2L, 2L + W) — so the expected state on reading address k is
# the basis vector k + (data[k] << 2L), the ladder having returned to 0.


def _qrom_readout_state(qrom: QROM, address: int) -> np.ndarray:
    lookup = qrom.kernel()
    num_address = qrom.num_address
    num_output = qrom.num_output

    @cudaq.kernel
    def run():
        address_reg = cudaq.qvector(num_address)
        ladder = cudaq.qvector(num_address)
        output = cudaq.qvector(num_output)
        for b in range(num_address):
            if ((address >> b) & 1) == 1:
                x(address_reg[b])
        lookup(address_reg, ladder, output)

    return np.array(cudaq.get_state(run))


@pytest.mark.parametrize("address_bits,num_entries", [(1, 2), (2, 4), (2, 3),
                                                      (3, 8), (3, 5), (4, 16),
                                                      (4, 11)])
@pytest.mark.parametrize("output_bits", [1, 3, 6])
def test_qrom_exhaustive_readout(address_bits, num_entries, output_bits):
    rng = np.random.default_rng(7 * address_bits + num_entries + output_bits)
    data = [int(v) for v in rng.integers(0, 1 << output_bits, num_entries)]
    qrom = QROM(data, address_bits, output_bits)
    total = 2 * address_bits + output_bits
    for address in range(1 << address_bits):
        # Out-of-range addresses of a short table must read zero.
        expected_word = data[address] if address < num_entries else 0
        state = _qrom_readout_state(qrom, address)
        expected = _basis(address + (expected_word << (2 * address_bits)),
                          total)
        np.testing.assert_allclose(state, expected, atol=1e-12)


def test_qrom_is_self_inverse():
    # X-only write phase: applying the lookup twice XORs the table twice.
    # Run on all addresses at once (H on the address) with a non-zero
    # initial output word to pin |y XOR d XOR d> = |y>.
    data = [5, 3, 0, 6, 7]
    qrom = QROM(data, address_bits=3, output_bits=3)
    lookup = qrom.kernel()
    initial = 0b101

    @cudaq.kernel
    def run():
        address_reg = cudaq.qvector(3)
        ladder = cudaq.qvector(3)
        output = cudaq.qvector(3)
        for b in range(3):
            h(address_reg[b])
            if ((initial >> b) & 1) == 1:
                x(output[b])
        lookup(address_reg, ladder, output)
        lookup(address_reg, ladder, output)

    state = np.array(cudaq.get_state(run))
    expected = np.zeros(1 << 9, dtype=np.complex128)
    for address in range(8):
        expected[address + (initial << 6)] = 1.0 / np.sqrt(8.0)
    np.testing.assert_allclose(state, expected, atol=1e-12)


def test_qrom_toffoli_count_bound():
    data = list(range(16))
    qrom = QROM(data, address_bits=4, output_bits=4)
    assert 0 < qrom.toffoli_count <= 2 * (len(data) - 1)
    short = QROM(data[:5], address_bits=4, output_bits=4)
    assert short.toffoli_count < qrom.toffoli_count


def test_qrom_validation_raises():
    with pytest.raises(ValueError, match="data must be non-empty"):
        QROM([], address_bits=2, output_bits=2)
    with pytest.raises(ValueError, match="addresses only 4"):
        QROM([1] * 5, address_bits=2, output_bits=2)
    with pytest.raises(ValueError, match="non-negative"):
        QROM([1, -2], address_bits=1, output_bits=2)
    with pytest.raises(ValueError, match="does not fit in output_bits"):
        QROM([1, 4], address_bits=1, output_bits=2)
    with pytest.raises(ValueError, match="address_bits must be a positive"):
        QROM([1], address_bits=0, output_bits=1)
    with pytest.raises(ValueError, match="output_bits must be a positive"):
        QROM([1], address_bits=1, output_bits=0)
    with pytest.raises(ValueError, match="entries must be integers"):
        QROM([1.5], address_bits=1, output_bits=2)


# ----------------------------------------------------------------------
# Unary iteration: direct ladder tests
# ----------------------------------------------------------------------


@pytest.mark.parametrize("num_items,marked", [(8, (1, 3, 4)), (5, (0, 4)),
                                              (1, (0, ))])
def test_unary_iteration_phase_kick_per_address(num_items, marked):
    # body(k) = Z on a |1> target for marked addresses: a uniform address
    # superposition picks up a -1 phase exactly on those addresses.
    walk = unary_iteration_kernels(3, num_items, lambda k: [("z", 0)]
                                   if k in marked else [])
    kernel = walk.kernel

    @cudaq.kernel
    def run():
        address_reg = cudaq.qvector(3)
        ladder = cudaq.qvector(3)
        target = cudaq.qvector(1)
        x(target[0])
        for b in range(3):
            h(address_reg[b])
        kernel(address_reg, ladder, target)

    state = np.array(cudaq.get_state(run))
    expected = np.zeros(1 << 7, dtype=np.complex128)
    for address in range(8):
        sign = -1.0 if address in marked else 1.0
        expected[address + (1 << 6)] = sign / np.sqrt(8.0)
    np.testing.assert_allclose(state, expected, atol=1e-12)


def test_unary_iteration_y_body_sign_convention():
    # body(2) = Y on target 0: |2>|0> -> i |2>|1>; other addresses idle.
    walk = unary_iteration_kernels(2, 4, lambda k: [("y", 0)]
                                   if k == 2 else [])
    kernel = walk.kernel

    def run_state(address: int) -> np.ndarray:

        @cudaq.kernel
        def run():
            address_reg = cudaq.qvector(2)
            ladder = cudaq.qvector(2)
            target = cudaq.qvector(1)
            for b in range(2):
                if ((address >> b) & 1) == 1:
                    x(address_reg[b])
            kernel(address_reg, ladder, target)

        return np.array(cudaq.get_state(run))

    expected = np.zeros(1 << 5, dtype=np.complex128)
    expected[2 + (1 << 4)] = 1.0j
    np.testing.assert_allclose(run_state(2), expected, atol=1e-12)
    np.testing.assert_allclose(run_state(1), _basis(1, 5), atol=1e-12)


def test_unary_iteration_adjoint_composes_to_identity():
    # Non-commuting multi-gate bodies (X, Z, Y mixed) so the walk is NOT
    # an involution; only the hand-written reversed-instruction inverse
    # restores the state.
    def body(k):
        if k % 2 == 1:
            return [("x", 0), ("z", 0), ("y", 1)]
        return [("x", 1)]

    walk = unary_iteration_kernels(3, 6, body)
    kernel = walk.kernel
    kernel_adj = walk.kernel_adj

    @cudaq.kernel
    def run():
        address_reg = cudaq.qvector(3)
        ladder = cudaq.qvector(3)
        target = cudaq.qvector(2)
        x(target[0])
        for b in range(3):
            h(address_reg[b])
        kernel(address_reg, ladder, target)
        kernel_adj(address_reg, ladder, target)

    state = np.array(cudaq.get_state(run))
    expected = np.zeros(1 << 8, dtype=np.complex128)
    for address in range(8):
        expected[address + (1 << 6)] = 1.0 / np.sqrt(8.0)
    np.testing.assert_allclose(state, expected, atol=1e-12)


@pytest.mark.parametrize("control_value", [0, 1])
def test_unary_iteration_controlled_variant(control_value):
    marked = (1, 2)
    walk = unary_iteration_kernels(2,
                                   4,
                                   lambda k: [("z", 0)] if k in marked else [],
                                   controlled=True)
    kernel = walk.kernel
    assert walk.controlled

    @cudaq.kernel
    def run():
        control = cudaq.qvector(1)
        address_reg = cudaq.qvector(2)
        ladder = cudaq.qvector(2)
        target = cudaq.qvector(1)
        if control_value == 1:
            x(control[0])
        x(target[0])
        for b in range(2):
            h(address_reg[b])
        kernel(control, address_reg, ladder, target)

    state = np.array(cudaq.get_state(run))
    expected = np.zeros(1 << 6, dtype=np.complex128)
    for address in range(4):
        kicked = control_value == 1 and address in marked
        sign = -1.0 if kicked else 1.0
        # Layout: control at 0, address at [1, 3), ladder [3, 5), target 5.
        expected[control_value + (address << 1) + (1 << 5)] = \
            sign / 2.0
    np.testing.assert_allclose(state, expected, atol=1e-12)


def test_unary_iteration_validation_raises():
    body = lambda k: []
    with pytest.raises(ValueError, match="num_address_bits must be a"):
        unary_iteration_kernels(0, 1, body)
    with pytest.raises(ValueError, match=r"num_items must be an integer in"):
        unary_iteration_kernels(2, 0, body)
    with pytest.raises(ValueError, match=r"num_items must be an integer in"):
        unary_iteration_kernels(2, 5, body)
    with pytest.raises(ValueError, match="unsupported gate 'q'"):
        unary_iteration_kernels(2, 4, lambda k: [("q", 0)])
    with pytest.raises(ValueError, match="invalid target qubit index"):
        unary_iteration_kernels(2, 4, lambda k: [("x", -1)])
