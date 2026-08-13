# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Exhaustive tests for the sparse-encoding arithmetic device kernels.

Every operation is checked against classical integer arithmetic over *all*
inputs for register widths up to 5, and every inverse is pinned by an
op-then-inverse == identity test. Exhaustiveness comes from a superposition
harness: the input registers are prepared in a uniform superposition, so a
single statevector comparison validates the operation's action on every
computational-basis input at once (the operations are classical
permutations, so distinct inputs cannot interfere).
"""

import numpy as np
import pytest

import cudaq

from cudaq_algorithms.sparse import _arithmetic as arith

WIDTHS = [1, 2, 3, 4, 5]


def _basis(index: int, num_qubits: int) -> np.ndarray:
    ket = np.zeros(1 << num_qubits, dtype=np.complex128)
    ket[index] = 1.0
    return ket


# ----------------------------------------------------------------------
# Register-register add / subtract (CDKM)
# ----------------------------------------------------------------------
#
# Layout: a at bits [0, n), b at [n, 2n), carry at bit 2n.


@cudaq.kernel
def _run_add_register(n: int, subtract: int):
    a = cudaq.qvector(n)
    b = cudaq.qvector(n)
    carry = cudaq.qvector(1)
    for k in range(n):
        h(a[k])
        h(b[k])
    if subtract == 0:
        arith.add_register(a, b, carry)
    else:
        arith.subtract_register(a, b, carry)


@cudaq.kernel
def _run_add_then_subtract(n: int):
    a = cudaq.qvector(n)
    b = cudaq.qvector(n)
    carry = cudaq.qvector(1)
    for k in range(n):
        h(a[k])
        h(b[k])
    arith.add_register(a, b, carry)
    arith.subtract_register(a, b, carry)


def _register_op_expected(n: int, op) -> np.ndarray:
    """Uniform superposition over (a, b) mapped through (a, op(a, b))."""
    expected = np.zeros(1 << (2 * n + 1), dtype=np.complex128)
    norm = 1.0 / (1 << n)
    for a in range(1 << n):
        for b in range(1 << n):
            expected[a + (op(a, b) % (1 << n)) * (1 << n)] += norm
    return expected


@pytest.mark.parametrize("n", WIDTHS)
def test_add_register_all_inputs(n):
    state = np.array(cudaq.get_state(_run_add_register, n, 0))
    np.testing.assert_allclose(state,
                               _register_op_expected(n, lambda a, b: a + b),
                               atol=1e-12)


@pytest.mark.parametrize("n", WIDTHS)
def test_subtract_register_all_inputs(n):
    state = np.array(cudaq.get_state(_run_add_register, n, 1))
    np.testing.assert_allclose(state,
                               _register_op_expected(n, lambda a, b: b - a),
                               atol=1e-12)


@pytest.mark.parametrize("n", WIDTHS)
def test_subtract_register_inverts_add_register(n):
    state = np.array(cudaq.get_state(_run_add_then_subtract, n))
    np.testing.assert_allclose(state,
                               _register_op_expected(n, lambda a, b: b),
                               atol=1e-12)


# ----------------------------------------------------------------------
# Constant add / subtract (CDKM and Draper QFT)
# ----------------------------------------------------------------------
#
# CDKM layout: target at [0, n), work at [n, 2n), carry at 2n.
# QFT layout: target only.


@cudaq.kernel
def _run_add_constant(n: int, bits: list[int], subtract: int, roundtrip: int):
    target = cudaq.qvector(n)
    work = cudaq.qvector(n)
    carry = cudaq.qvector(1)
    for k in range(n):
        h(target[k])
    if subtract == 0:
        arith.add_constant(target, bits, work, carry)
    else:
        arith.subtract_constant(target, bits, work, carry)
    if roundtrip == 1:
        if subtract == 0:
            arith.subtract_constant(target, bits, work, carry)
        else:
            arith.add_constant(target, bits, work, carry)


@cudaq.kernel
def _run_add_constant_qft(n: int, constant: int, subtract: int,
                          roundtrip: int):
    target = cudaq.qvector(n)
    for k in range(n):
        h(target[k])
    if subtract == 0:
        arith.add_constant_qft(target, constant)
    else:
        arith.subtract_constant_qft(target, constant)
    if roundtrip == 1:
        if subtract == 0:
            arith.subtract_constant_qft(target, constant)
        else:
            arith.add_constant_qft(target, constant)


def _constant_op_expected(n: int, shift: int, extra_qubits: int) -> np.ndarray:
    expected = np.zeros(1 << (n + extra_qubits), dtype=np.complex128)
    norm = 1.0 / np.sqrt(1 << n)
    for value in range(1 << n):
        expected[(value + shift) % (1 << n)] += norm
    return expected


def _bits(value: int, n: int) -> list[int]:
    return [(value >> k) & 1 for k in range(n)]


@pytest.mark.parametrize("n", WIDTHS)
def test_add_and_subtract_constant_all_inputs(n):
    for constant in range(1 << n):
        bits = _bits(constant, n)
        added = np.array(cudaq.get_state(_run_add_constant, n, bits, 0, 0))
        np.testing.assert_allclose(added,
                                   _constant_op_expected(n, constant, n + 1),
                                   atol=1e-12)
        subtracted = np.array(cudaq.get_state(_run_add_constant, n, bits, 1,
                                              0))
        np.testing.assert_allclose(subtracted,
                                   _constant_op_expected(n, -constant, n + 1),
                                   atol=1e-12)
        roundtrip = np.array(cudaq.get_state(_run_add_constant, n, bits, 0, 1))
        np.testing.assert_allclose(roundtrip,
                                   _constant_op_expected(n, 0, n + 1),
                                   atol=1e-12)


@pytest.mark.parametrize("n", WIDTHS)
def test_add_and_subtract_constant_qft_all_inputs(n):
    for constant in range(1 << n):
        added = np.array(
            cudaq.get_state(_run_add_constant_qft, n, constant, 0, 0))
        np.testing.assert_allclose(added,
                                   _constant_op_expected(n, constant, 0),
                                   atol=1e-10)
        subtracted = np.array(
            cudaq.get_state(_run_add_constant_qft, n, constant, 1, 0))
        np.testing.assert_allclose(subtracted,
                                   _constant_op_expected(n, -constant, 0),
                                   atol=1e-10)
        roundtrip = np.array(
            cudaq.get_state(_run_add_constant_qft, n, constant, 0, 1))
        np.testing.assert_allclose(roundtrip,
                                   _constant_op_expected(n, 0, 0),
                                   atol=1e-10)


# ----------------------------------------------------------------------
# >= comparators (CDKM and Draper QFT)
# ----------------------------------------------------------------------


@cudaq.kernel
def _run_cmp_ge_constant(n: int, bits: list[int], k_is_zero: int):
    x_reg = cudaq.qvector(n)
    work = cudaq.qvector(n)
    carry = cudaq.qvector(1)
    out = cudaq.qvector(1)
    for k in range(n):
        h(x_reg[k])
    arith.cmp_ge_constant(x_reg, bits, k_is_zero, work, carry, out)


@cudaq.kernel
def _run_cmp_ge_constant_qft(n: int, constant: int, invert: int,
                             uncompute: int):
    x_reg = cudaq.qvector(n)
    out = cudaq.qvector(1)
    for k in range(n):
        h(x_reg[k])
    arith.cmp_ge_constant_qft(x_reg, out, constant, invert)
    if uncompute == 1:
        arith.cmp_ge_constant_qft_adj(x_reg, out, constant, invert)


def _cmp_expected(n: int, predicate, extra_before_out: int) -> np.ndarray:
    """|x> (work/carry |0>) |out = predicate(x)> over superposed x."""
    total = n + extra_before_out + 1
    expected = np.zeros(1 << total, dtype=np.complex128)
    norm = 1.0 / np.sqrt(1 << n)
    for value in range(1 << n):
        expected[value +
                 (1 << (n + extra_before_out)) * int(predicate(value))] += norm
    return expected


@pytest.mark.parametrize("n", WIDTHS)
def test_cmp_ge_constant_all_inputs(n):
    for constant in range(1 << n):
        complement = _bits((1 << n) - constant, n) if constant else [0] * n
        state = np.array(
            cudaq.get_state(_run_cmp_ge_constant, n, complement,
                            int(constant == 0)))
        np.testing.assert_allclose(state,
                                   _cmp_expected(n, lambda v: v >= constant,
                                                 n + 1),
                                   atol=1e-12)


@pytest.mark.parametrize("n", WIDTHS)
def test_cmp_ge_constant_qft_all_inputs(n):
    # Between the compute/adjoint pair the register is shifted by -K (a
    # documented contract), so the compute-only check reads out through the
    # shifted basis; the roundtrip check pins full restoration.
    for constant in range(1 << n):
        for invert in (0, 1):
            state = np.array(
                cudaq.get_state(_run_cmp_ge_constant_qft, n, constant, invert,
                                0))
            expected = np.zeros(1 << (n + 1), dtype=np.complex128)
            norm = 1.0 / np.sqrt(1 << n)
            for value in range(1 << n):
                flag = (value >= constant) if invert == 0 else (value
                                                                < constant)
                shifted = (value - constant) % (1 << n)
                expected[shifted + (int(flag) << n)] += norm
            np.testing.assert_allclose(state, expected, atol=1e-10)
            roundtrip = np.array(
                cudaq.get_state(_run_cmp_ge_constant_qft, n, constant, invert,
                                1))
            np.testing.assert_allclose(roundtrip,
                                       _constant_op_expected(n, 0, 1),
                                       atol=1e-10)


# ----------------------------------------------------------------------
# Direct basis-state spot checks (readable, non-superposed)
# ----------------------------------------------------------------------


@cudaq.kernel
def _spot_add(n: int, aval: int, bval: int):
    a = cudaq.qvector(n)
    b = cudaq.qvector(n)
    carry = cudaq.qvector(1)
    for k in range(n):
        if ((aval >> k) & 1) == 1:
            x(a[k])
        if ((bval >> k) & 1) == 1:
            x(b[k])
    arith.add_register(a, b, carry)


def test_add_register_basis_spot_checks():
    for n, aval, bval in [(2, 1, 2), (3, 5, 6), (5, 21, 27)]:
        state = np.array(cudaq.get_state(_spot_add, n, aval, bval))
        index = aval + (((aval + bval) % (1 << n)) << n)
        np.testing.assert_allclose(state, _basis(index, 2 * n + 1), atol=1e-12)
