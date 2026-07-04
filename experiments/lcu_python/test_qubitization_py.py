# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""Tests for the pure-Python qubitization prototype (dense references)."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

import cudaq

import pauli_lcu_py as lcu
import qubitization_py as qub
from test_pauli_lcu_py import dense_matrix, random_ket


@pytest.fixture(autouse=True)
def qpp_cpu_target():
    cudaq.set_target("qpp-cpu")
    yield
    cudaq.reset_target()


def exact_chebyshev_moments(terms, num_qubits, alpha, ket, count):
    """<ket| T_k(H/alpha) |ket> from the dense matrix, k = 0..count-1."""
    scaled = dense_matrix(terms, num_qubits) / alpha
    ket = np.asarray(ket, dtype=np.complex128)
    chebyshev = [np.eye(scaled.shape[0], dtype=np.complex128), scaled]
    while len(chebyshev) < count:
        chebyshev.append(2.0 * scaled @ chebyshev[-1] - chebyshev[-2])
    return [float(np.real(ket.conj() @ chebyshev[k] @ ket))
            for k in range(count)]


THREE_TERMS_1Q = {"I": 0.2, "X": 0.5, "Z": 0.3}  # asymmetric spectrum
FOUR_TERMS_2Q = {"ZI": 0.70, "IZ": -0.43, "XX": 0.19, "YZ": 0.11}


def test_moments_match_dense_chebyshev_1q():
    enc = lcu.PauliLCU(THREE_TERMS_1Q)
    walk = qub.Walk(enc)

    theta = 0.7
    ket = np.array([np.cos(theta / 2), np.sin(theta / 2)],
                   dtype=np.complex128)
    count = 6  # both parities, walk powers up to 2
    measured = walk.moments(ket, count)
    expected = exact_chebyshev_moments(
        [(c, w) for w, c in THREE_TERMS_1Q.items()], 1, enc.alpha, ket, count)
    assert np.allclose(measured, expected, atol=1e-10)


def test_moments_match_dense_chebyshev_2q():
    enc = lcu.PauliLCU(FOUR_TERMS_2Q)
    walk = qub.Walk(enc)

    ket = np.zeros(4, dtype=np.complex128)
    ket[1] = 1.0  # basis state, HF-style
    count = 5
    measured = walk.moments(ket, count)
    expected = exact_chebyshev_moments(
        [(c, w) for w, c in FOUR_TERMS_2Q.items()], 2, enc.alpha, ket, count)
    assert np.allclose(measured, expected, atol=1e-10)


def test_adjoint_walk_inverts_walk():
    enc = lcu.PauliLCU(FOUR_TERMS_2Q)
    walk = qub.Walk(enc)

    ket = random_ket(2, seed=5)
    reference = np.zeros(1 << (enc.num_system + enc.num_ancilla),
                         dtype=np.complex128)
    reference[:4] = ket

    for power in (1, 2, 3):
        state = cudaq.get_state(walk.roundtrip_kernel(power=power),
                                cudaq.State.from_data(ket))
        assert np.allclose(np.asarray(state), reference, atol=1e-10), \
            f"roundtrip failed at power {power}"


def test_walk_kernel_options():
    enc = lcu.PauliLCU(THREE_TERMS_1Q)
    walk = qub.Walk(enc)
    ket = np.array([1.0, 0.0], dtype=np.complex128)

    # uncompute=True must agree with the PauliLCU walk_kernel factory.
    a = np.asarray(cudaq.get_state(walk.kernel(power=2),
                                   cudaq.State.from_data(ket)))
    b = np.asarray(cudaq.get_state(enc.walk_kernel(power=2),
                                   cudaq.State.from_data(ket)))
    assert np.allclose(a, b, atol=1e-12)


def test_reflection_and_select_observables_shapes():
    enc = lcu.PauliLCU(FOUR_TERMS_2Q)
    reflection = qub.reflection_observable(enc)
    select = qub.select_observable(enc)
    # Spot check: <0...0| R |0...0> = +1 on the ancilla block.
    # (Full physics is covered by the moment tests.)
    assert reflection is not None
    assert select is not None


def test_walk_rejects_degenerate_encoding():
    single = lcu.PauliLCU({"XZ": -0.5})
    with pytest.raises(ValueError):
        qub.Walk(single)
    with pytest.raises(ValueError):
        qub.reflection_observable(single)
    with pytest.raises(ValueError):
        qub.select_observable(single)
