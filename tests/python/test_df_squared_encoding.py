# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                        #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""DoubleFactorizedSquaredEncoding: correctness against dense references.

The squared-oracle (von Burg ``burg``) encoding block-encodes the *same* dense
electronic Hamiltonian as ``DoubleFactorizedEncoding`` but realises the
two-body part as a coherent sum of squared one-body operators, reaching the
smaller ``burg`` one-norm. Tests mirror ``test_df_encoding.py`` (the dense
harness is imported from it) plus a Phase-1 alpha/feasibility gate, the
inner ``_offset_prepare`` inverse property, and a C-DF check.
"""

import numpy as np
import pytest

import cudaq

import cudaq_algorithms as algorithms
from cudaq_algorithms import (BlockEncoding, DoubleFactorizedEncoding,
                              DoubleFactorizedSquaredEncoding, PhaseSequence,
                              QSVT, Walk)
from cudaq_algorithms.common_kernels import state_from
from cudaq_algorithms import df_squared_encoding as dsq

# Reuse the dense fermionic harness and random systems from the existing suite.
from test_df_encoding import (dense_hamiltonian, encoded_block, random_ket,
                              random_system)

df = algorithms.double_factorization


def _kappa_eigenvalues(one_body, factorization):
    kappa = np.asarray(one_body, dtype=float).copy()
    for rotation, core in zip(factorization.leaf_rotations,
                              factorization.leaf_cores):
        absorbed = (0.5 * (core.sum(axis=1) + core.sum(axis=0)) -
                    0.5 * np.diag(core))
        kappa += (rotation * absorbed) @ rotation.T
    return np.linalg.eigvalsh(kappa)


# ----------------------------------------------------------------------
# Protocol / construction
# ----------------------------------------------------------------------


def test_satisfies_block_encoding_protocol():
    one_body, eri = random_system(7)
    encoding = DoubleFactorizedSquaredEncoding(one_body, eri)
    assert isinstance(encoding, BlockEncoding)
    assert encoding.num_system == 4
    assert encoding.num_ancilla >= 1


def test_input_validation_delegates_to_base():
    one_body, eri = random_system(7)
    with pytest.raises(ValueError, match="square"):
        DoubleFactorizedSquaredEncoding(np.zeros((2, 3)), eri)
    with pytest.raises(ValueError, match="symmetric"):
        DoubleFactorizedSquaredEncoding(np.array([[0.0, 1.0], [0.0, 0.0]]),
                                        eri)
    with pytest.raises(ValueError, match="chemist-notation"):
        DoubleFactorizedSquaredEncoding(one_body, np.zeros((3, 3, 3, 3)))


# ----------------------------------------------------------------------
# Phase 1: alpha / feasibility gate
# ----------------------------------------------------------------------


@pytest.mark.parametrize("seed,n", [(7, 2), (21, 2), (5, 3)])
def test_alpha_minus_constant_is_burg_one_norm(seed, n):
    one_body, eri = random_system(seed, n=n)
    encoding = DoubleFactorizedSquaredEncoding(one_body, eri)
    burg = df.double_factorization_one_norm(
        encoding.factorization,
        _kappa_eigenvalues(one_body, encoding.factorization), "burg")
    assert encoding.alpha - abs(encoding.constant_term) == pytest.approx(
        burg, rel=1e-10)


@pytest.mark.parametrize("seed,n", [(7, 2), (21, 2), (5, 3)])
def test_alpha_is_feasible(seed, n):
    one_body, eri = random_system(seed, n=n)
    encoding = DoubleFactorizedSquaredEncoding(one_body, eri)
    spectral_norm = np.max(
        np.abs(np.linalg.eigvalsh(dense_hamiltonian(one_body, eri))))
    assert encoding.alpha + 1e-9 >= spectral_norm


def test_burg_alpha_beats_lcu_alpha():
    # The whole point of the squared oracle: a smaller (or equal) one-norm than
    # the ZZ-word LCU encoding for the two-body part.
    one_body, eri = random_system(5, n=3)
    factorization = df.explicit_double_factorization(eri, threshold=0.0)
    eig = _kappa_eigenvalues(one_body, factorization)
    burg = df.double_factorization_one_norm(factorization, eig, "burg")
    lcu = df.double_factorization_one_norm(factorization, eig, "lcu")
    assert burg <= lcu + 1e-12


# ----------------------------------------------------------------------
# Phase 5: encode block == H / alpha
# ----------------------------------------------------------------------


@pytest.mark.parametrize("seed,n", [(7, 2), (21, 2), (5, 3)])
def test_encode_block_is_h_over_alpha(seed, n):
    one_body, eri = random_system(seed, n=n)
    encoding = DoubleFactorizedSquaredEncoding(one_body, eri)
    h = dense_hamiltonian(one_body, eri)
    ket = random_ket(1 << (2 * n))
    block = encoded_block(encoding, encoding.encode_kernel(), ket)
    np.testing.assert_allclose(block, (h @ ket) / encoding.alpha, atol=1e-11)


# ----------------------------------------------------------------------
# encode_constant=False: the query-cost-relevant mode (block encodes the
# constant-shifted Hamiltonian at the published von Burg one-norm)
# ----------------------------------------------------------------------


@pytest.mark.parametrize("seed,n", [(7, 2), (21, 2), (5, 3)])
def test_encode_constant_false_alpha_is_dfnorm_and_beats_existing(seed, n):
    one_body, eri = random_system(seed, n=n)
    full = DoubleFactorizedSquaredEncoding(one_body, eri)
    shifted = DoubleFactorizedSquaredEncoding(one_body,
                                              eri,
                                              encode_constant=False)
    # Dropping the identity slot removes exactly |constant_term| from alpha,
    # leaving the published von Burg one-norm sum_k|F_k| + 1/4 sum|lambda|S^2.
    burg = df.double_factorization_one_norm(
        full.factorization, _kappa_eigenvalues(one_body, full.factorization),
        "burg")
    assert shifted.alpha == pytest.approx(full.alpha - abs(full.constant_term),
                                          rel=1e-12)
    assert shifted.alpha == pytest.approx(burg, rel=1e-10)
    # ... and it is strictly below the ZZ-word encoding's total one-norm.
    existing = DoubleFactorizedEncoding(one_body, eri)
    assert shifted.alpha < existing.alpha
    assert shifted.constant_term == pytest.approx(full.constant_term)


@pytest.mark.parametrize("seed,n", [(7, 2), (21, 2), (5, 3)])
def test_encode_constant_false_block_is_shifted_hamiltonian(seed, n):
    one_body, eri = random_system(seed, n=n)
    encoding = DoubleFactorizedSquaredEncoding(one_body,
                                               eri,
                                               encode_constant=False)
    h = dense_hamiltonian(one_body, eri)
    shifted_h = h - encoding.constant_term * np.eye(h.shape[0])
    ket = random_ket(1 << (2 * n))
    block = encoded_block(encoding, encoding.encode_kernel(), ket)
    np.testing.assert_allclose(block, (shifted_h @ ket) / encoding.alpha,
                               atol=1e-11)


def test_encode_constant_true_is_default_and_encodes_full_h():
    one_body, eri = random_system(21)
    default = DoubleFactorizedSquaredEncoding(one_body, eri)
    assert default.encode_constant is True
    h = dense_hamiltonian(one_body, eri)
    ket = random_ket(1 << default.num_system)
    block = encoded_block(default, default.encode_kernel(), ket)
    np.testing.assert_allclose(block, (h @ ket) / default.alpha, atol=1e-11)


def test_scalar_offset_shifts_the_encoded_operator():
    one_body, eri = random_system(7)
    offset = 0.7137
    encoding = DoubleFactorizedSquaredEncoding(one_body,
                                               eri,
                                               scalar_offset=offset)
    h = dense_hamiltonian(one_body, eri) + offset * np.eye(16)
    ket = random_ket(16)
    block = encoded_block(encoding, encoding.encode_kernel(), ket)
    np.testing.assert_allclose(block, (h @ ket) / encoding.alpha, atol=1e-11)


def test_truncated_factorization_encodes_truncated_hamiltonian():
    one_body, eri = random_system(7)
    truncated = df.explicit_double_factorization(eri, max_num_leaves=1)
    encoding = DoubleFactorizedSquaredEncoding(one_body, truncated)
    h = dense_hamiltonian(one_body, df.reconstruct_eri(truncated))
    ket = random_ket(16)
    block = encoded_block(encoding, encoding.encode_kernel(), ket)
    np.testing.assert_allclose(block, (h @ ket) / encoding.alpha, atol=1e-11)


def test_identity_only_encoding():
    encoding = DoubleFactorizedSquaredEncoding(np.zeros((2, 2)),
                                               np.zeros((2, 2, 2, 2)),
                                               scalar_offset=1.5)
    assert encoding.alpha == pytest.approx(1.5)
    assert encoding.constant_term == pytest.approx(1.5)
    ket = random_ket(16)
    block = encoded_block(encoding, encoding.encode_kernel(), ket)
    np.testing.assert_allclose(block, ket, atol=1e-11)


def test_negative_scalar_offset():
    encoding = DoubleFactorizedSquaredEncoding(np.zeros((2, 2)),
                                               np.zeros((2, 2, 2, 2)),
                                               scalar_offset=-1.5)
    assert encoding.alpha == pytest.approx(1.5)
    ket = random_ket(16)
    block = encoded_block(encoding, encoding.encode_kernel(), ket)
    np.testing.assert_allclose(block, -ket, atol=1e-10)


def test_diagonal_system():
    eri = np.einsum('pq,rs->pqrs', np.diag([0.4, 0.9]), np.diag([0.4, 0.9]))
    encoding = DoubleFactorizedSquaredEncoding(np.diag([0.7, -0.3]), eri)
    h = dense_hamiltonian(np.diag([0.7, -0.3]), eri)
    ket = random_ket(16)
    block = encoded_block(encoding, encoding.encode_kernel(), ket)
    np.testing.assert_allclose(block, (h @ ket) / encoding.alpha, atol=1e-11)


# ----------------------------------------------------------------------
# Phase 7: C-DF (rank > 1, mixed-sign leaf cores)
# ----------------------------------------------------------------------


def test_compressed_factorization_encodes_reconstructed_hamiltonian():
    one_body, eri = random_system(5, n=3)
    compressed = df.compressed_double_factorization(eri, num_leaves=3)
    encoding = DoubleFactorizedSquaredEncoding(one_body, compressed)
    h = dense_hamiltonian(one_body, df.reconstruct_eri(compressed))
    ket = random_ket(1 << 6)
    block = encoded_block(encoding, encoding.encode_kernel(), ket)
    np.testing.assert_allclose(block, (h @ ket) / encoding.alpha, atol=1e-11)


# ----------------------------------------------------------------------
# Phase 6: Walk / QSVT consumers
# ----------------------------------------------------------------------


def _chebyshev(dense_scaled: np.ndarray, order: int) -> np.ndarray:
    t_prev = np.eye(dense_scaled.shape[0], dtype=complex)
    t_cur = dense_scaled.copy()
    if order == 0:
        return t_prev
    for _ in range(order - 1):
        t_prev, t_cur = t_cur, 2.0 * dense_scaled @ t_cur - t_prev
    return t_cur


def test_walk_kernel_applies_chebyshev_of_minus_h():
    one_body, eri = random_system(7)
    encoding = DoubleFactorizedSquaredEncoding(one_body, eri)
    scaled = -dense_hamiltonian(one_body, eri) / encoding.alpha
    ket = random_ket(16)
    for power in (1, 2, 3):
        block = encoded_block(encoding, encoding.walk_kernel(power), ket)
        np.testing.assert_allclose(block,
                                   _chebyshev(scaled, power) @ ket,
                                   atol=1e-11)


def test_walk_even_moments_match_dense():
    one_body, eri = random_system(7)
    encoding = DoubleFactorizedSquaredEncoding(one_body, eri)
    walk = Walk(encoding)
    scaled = dense_hamiltonian(one_body, eri) / encoding.alpha
    ket = random_ket(16)
    for order in (0, 2, 4):
        expected = float(np.real(ket.conj() @ _chebyshev(scaled, order) @ ket))
        assert walk.moment(ket, order) == pytest.approx(expected, abs=1e-10)


def test_odd_moments_are_unavailable():
    one_body, eri = random_system(7)
    encoding = DoubleFactorizedSquaredEncoding(one_body, eri)
    with pytest.raises(NotImplementedError, match="select_observable"):
        encoding.select_observable()


def test_qsvt_sequence_matches_signal_model():
    from test_qsvt import reference_response

    one_body, eri = random_system(7)
    encoding = DoubleFactorizedSquaredEncoding(one_body, eri)
    h_scaled = dense_hamiltonian(one_body, eri) / encoding.alpha
    ket = random_ket(16)

    sequence = PhaseSequence([0.23, -0.41, 0.11])
    kernel = QSVT(encoding).kernel(sequence)
    block = encoded_block(encoding, kernel, ket)

    eigenvalues, vectors = np.linalg.eigh(h_scaled)
    coefficients = vectors.conj().T @ ket
    expected = (vectors * np.array(
        [reference_response(sequence, float(ev))
         for ev in eigenvalues])) @ coefficients
    np.testing.assert_allclose(block, expected, atol=1e-10)


# ----------------------------------------------------------------------
# Controlled variants
# ----------------------------------------------------------------------


def _controlled_circuit(encoding, control_value: int):
    controlled = encoding.controlled_apply_kernel()
    n_anc = encoding.num_ancilla
    flip = control_value

    @cudaq.kernel
    def circuit(state: cudaq.State):
        system = cudaq.qvector(state)
        control_and_ancilla = cudaq.qvector(n_anc + 1)
        if flip == 1:
            x(control_and_ancilla[0])
        controlled(control_and_ancilla, system)
        if flip == 1:
            x(control_and_ancilla[0])

    return circuit


def test_controlled_apply_control_conventions():
    one_body, eri = random_system(7)
    encoding = DoubleFactorizedSquaredEncoding(one_body, eri)
    h = dense_hamiltonian(one_body, eri)
    ket = random_ket(16)

    at_zero = np.array(
        cudaq.get_state(_controlled_circuit(encoding, 0), state_from(ket)))
    identity_expected = np.zeros_like(at_zero)
    identity_expected[:16] = ket
    np.testing.assert_allclose(at_zero, identity_expected, atol=1e-11)

    at_one = np.array(
        cudaq.get_state(_controlled_circuit(encoding, 1), state_from(ket)))
    np.testing.assert_allclose(at_one[:16], (h @ ket) / encoding.alpha,
                               atol=1e-11)


@pytest.mark.parametrize("control_value", [0, 1])
def test_controlled_walk_step_roundtrip_is_identity(control_value):
    one_body, eri = random_system(7)
    encoding = DoubleFactorizedSquaredEncoding(one_body, eri)
    step = encoding.controlled_walk_step_kernel()
    adjoint_step = encoding.controlled_adjoint_walk_step_kernel()
    n_anc = encoding.num_ancilla
    flip = control_value

    @cudaq.kernel
    def circuit(state: cudaq.State):
        system = cudaq.qvector(state)
        control_and_ancilla = cudaq.qvector(n_anc + 1)
        if flip == 1:
            x(control_and_ancilla[0])
        step(control_and_ancilla, system)
        adjoint_step(control_and_ancilla, system)
        if flip == 1:
            x(control_and_ancilla[0])

    ket = random_ket(16)
    out = np.array(cudaq.get_state(circuit, state_from(ket)))
    expected = np.zeros_like(out)
    expected[:16] = ket
    np.testing.assert_allclose(out, expected, atol=1e-10)


def test_walk_roundtrip_through_consumer_is_identity():
    one_body, eri = random_system(7)
    encoding = DoubleFactorizedSquaredEncoding(one_body, eri)
    walk = Walk(encoding)
    ket = random_ket(16)
    kernel = walk.roundtrip_kernel(power=2)
    out = np.array(cudaq.get_state(kernel, state_from(ket)))
    expected = np.zeros_like(out)
    expected[:16] = ket
    np.testing.assert_allclose(out, expected, atol=1e-10)


# ----------------------------------------------------------------------
# state_prep injection (twin-pinning)
# ----------------------------------------------------------------------

_T0, _T1, _T2, _T3 = 0.37, -0.52, 0.21, -0.83


@cudaq.kernel
def _product_prep(qubits: cudaq.qview):
    rx(_T0, qubits[0])
    ry(_T1, qubits[1])
    rx(_T2, qubits[2])
    ry(_T3, qubits[3])


def _injected_ket() -> np.ndarray:
    rx0 = np.array([np.cos(0.5 * _T0), -1.0j * np.sin(0.5 * _T0)])
    ry1 = np.array([np.cos(0.5 * _T1), np.sin(0.5 * _T1)])
    rx2 = np.array([np.cos(0.5 * _T2), -1.0j * np.sin(0.5 * _T2)])
    ry3 = np.array([np.cos(0.5 * _T3), np.sin(0.5 * _T3)])
    out = np.array([1.0])
    for factor in (rx0, ry1, rx2, ry3):
        out = np.kron(factor, out)
    return out.astype(np.complex128)


def test_encode_kernel_prep_injection():
    one_body, eri = random_system(7)
    encoding = DoubleFactorizedSquaredEncoding(one_body, eri)
    via_prep = np.array(
        cudaq.get_state(encoding.encode_kernel(state_prep=_product_prep)))
    via_state = np.array(
        cudaq.get_state(encoding.encode_kernel(), state_from(_injected_ket())))
    np.testing.assert_allclose(via_prep, via_state, atol=1e-11)


def test_walk_kernel_prep_injection():
    one_body, eri = random_system(7)
    encoding = DoubleFactorizedSquaredEncoding(one_body, eri)
    via_prep = np.array(
        cudaq.get_state(encoding.walk_kernel(power=2,
                                             state_prep=_product_prep)))
    via_state = np.array(
        cudaq.get_state(encoding.walk_kernel(power=2),
                        state_from(_injected_ket())))
    np.testing.assert_allclose(via_prep, via_state, atol=1e-11)


# ----------------------------------------------------------------------
# Phase 2: inner-primitive inverse property (no cudaq.adjoint)
# ----------------------------------------------------------------------


def test_offset_prepare_inverts():
    # _offset_unprepare is the hand-written inverse of _offset_prepare on the
    # angle window [base : base + count].
    n_in = 2
    pad = 5  # nonzero base to exercise the offset arithmetic
    probs = [0.1, 0.4, 0.2, 0.3]
    angles = dsq._prepare_angles(probs)
    padded = [0.0] * pad + list(angles)
    count = (1 << n_in) - 1

    ket = random_ket(1 << n_in, seed=4)

    @cudaq.kernel
    def circuit(state: cudaq.State):
        reg = cudaq.qvector(state)
        dsq._offset_prepare(reg, padded, pad)
        dsq._offset_unprepare(reg, padded, pad, count)

    out = np.array(cudaq.get_state(circuit, state_from(ket)))
    np.testing.assert_allclose(out, ket, atol=1e-12)
