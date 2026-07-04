# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""Tests for the pure-Python QSVT prototype (dense and host-model references)."""

import math
import os
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

import cudaq

import pauli_lcu_py as lcu
import qsvt_py as qsvt
from test_pauli_lcu_py import dense_matrix, random_ket


# Override with e.g. LCU_PY_TARGET=nvidia-fp64 to run on a GPU simulator.
SIMULATION_TARGET = os.environ.get("LCU_PY_TARGET", "qpp-cpu")


@pytest.fixture(autouse=True)
def simulation_target():
    cudaq.set_target(SIMULATION_TARGET)
    yield
    cudaq.reset_target()


TWO_TERMS_1Q = {"X": 0.5, "Z": 0.3}
FOUR_TERMS_2Q = {"ZI": 0.70, "IZ": -0.43, "XX": 0.19, "YZ": 0.11}


def test_phase_sequence_validation_and_conversion():
    seq = qsvt.PhaseSequence([0.1, -0.2, 0.3])
    assert seq.degree == 2
    assert seq.convention == "qsvt"
    assert seq.walk_directions == (qsvt.FORWARD, qsvt.FORWARD)
    assert seq.projector_phases == pytest.approx([0.1, -0.2, 0.3])

    qsp = qsvt.PhaseSequence([0.1, -0.2], convention="qsp")
    assert qsp.projector_phases == pytest.approx([0.2, -0.4])
    assert qsp.phases == pytest.approx((0.1, -0.2))  # raw stays raw

    mixed = qsvt.PhaseSequence([0.1, -0.2, 0.3],
                               walk_directions=["forward", "adjoint"])
    assert mixed.walk_directions == (qsvt.FORWARD, qsvt.ADJOINT)

    with pytest.raises(ValueError):
        qsvt.PhaseSequence([])
    with pytest.raises(ValueError):
        qsvt.PhaseSequence([0.1, 0.2], walk_directions=[0, 1])
    with pytest.raises(ValueError):
        qsvt.PhaseSequence([0.1, 0.2], walk_directions=["sideways"])
    with pytest.raises(ValueError):
        qsvt.PhaseSequence([0.1], convention="phaseish")


def test_response_matches_circuit_on_eigenstate():
    # H = 0.5 X + 0.3 Z: eigenvalues +/- lambda, ry(theta)|0> the +lambda vec.
    enc = lcu.PauliLCU(TWO_TERMS_1Q)
    transformer = qsvt.QSVT(enc)
    lam = math.sqrt(0.34)
    theta = math.atan2(0.5, 0.3)
    eigenvector = np.array([math.cos(theta / 2), math.sin(theta / 2)],
                           dtype=np.complex128)

    for phases in ([0.3, -0.4], [0.2, -0.5, 0.1, 0.4]):
        seq = qsvt.PhaseSequence(phases)
        good = transformer.transform(eigenvector, seq)
        # x is the plain scaled eigenvalue — no caller-side negation.
        response = qsvt.evaluate_response(seq, lam / enc.alpha)
        assert np.allclose(good, response * eigenvector, atol=1e-10)

        # Guard: the negated-eigenvalue prediction must NOT match for odd
        # degree (the sign is folded into the model's walk step).
        flipped = qsvt.evaluate_response(seq, -lam / enc.alpha)
        assert abs(response - flipped) > 1e-6


def test_full_block_matches_host_model_with_mixed_directions():
    enc = lcu.PauliLCU(FOUR_TERMS_2Q)
    transformer = qsvt.QSVT(enc)
    seq = qsvt.PhaseSequence([0.3, -0.4, 0.25],
                             walk_directions=["forward", "adjoint"])

    matrix = dense_matrix([(c, w) for w, c in FOUR_TERMS_2Q.items()], 2)
    eigenvalues, eigenvectors = np.linalg.eigh(matrix / enc.alpha)

    columns = []
    for basis in range(4):
        ket = np.zeros(4, dtype=np.complex128)
        ket[basis] = 1.0
        columns.append(transformer.transform(ket, seq))
    device_block = np.column_stack(columns)

    response = np.array(
        [qsvt.evaluate_response(seq, float(m)) for m in eigenvalues])
    host_block = eigenvectors @ np.diag(response) @ eigenvectors.conj().T
    assert np.allclose(device_block, host_block, atol=1e-9)


def test_qsp_sequence_executes_as_doubled_projector_phases():
    enc = lcu.PauliLCU(TWO_TERMS_1Q)
    transformer = qsvt.QSVT(enc)
    ket = random_ket(1, seed=9)

    qsp_seq = qsvt.PhaseSequence([0.15, -0.3, 0.45], convention="qsp")
    doubled = qsvt.PhaseSequence([0.3, -0.6, 0.9])
    a = transformer.transform(ket, qsp_seq)
    b = transformer.transform(ket, doubled)
    assert np.allclose(a, b, atol=1e-12)

    # And the two host conventions genuinely differ.
    r_qsp = qsvt.evaluate_response(qsp_seq, 0.5)
    r_qsvt = qsvt.evaluate_response(qsvt.PhaseSequence([0.15, -0.3, 0.45]),
                                    0.5)
    assert abs(r_qsp - r_qsvt) > 1e-6


def test_degree_zero_sequence_is_a_signal_phase():
    enc = lcu.PauliLCU(TWO_TERMS_1Q)
    transformer = qsvt.QSVT(enc)
    ket = random_ket(1, seed=4)
    good = transformer.transform(ket, [0.7])
    assert np.allclose(good, np.exp(0.7j) * ket, atol=1e-12)


def test_qsvt_rejects_degenerate_encoding():
    with pytest.raises(ValueError):
        qsvt.QSVT(lcu.PauliLCU({"XZ": -0.5}))


def _qsppack_hamiltonian_simulation_phases(tau, degree):
    qsppack = pytest.importorskip("qsppack")
    special = pytest.importorskip("scipy.special")

    cos_coefficients = np.array(
        [0.5 * special.jv(0, tau)] +
        [((-1)**k) * special.jv(2 * k, tau)
         for k in range(1, degree // 2 + 1)],
        dtype=np.float64)
    sin_coefficients = np.array(
        [((-1)**k) * special.jv(2 * k + 1, tau) for k in range(degree // 2)],
        dtype=np.float64)
    options = {
        "criteria": 1e-12,
        "method": "Newton",
        "typePhi": "full",
        "useReal": True,
    }
    cos_phases, _ = qsppack.solve(cos_coefficients, 0, {
        **options, "targetPre": True
    })
    sin_phases, _ = qsppack.solve(sin_coefficients, 1, {
        **options, "targetPre": False
    })
    return [float(p) for p in cos_phases], [float(p) for p in sin_phases]


def test_hamiltonian_simulation_with_qsppack_phases():
    terms = {"ZI": 0.6, "XX": 0.8}
    enc = lcu.PauliLCU(terms)
    transformer = qsvt.QSVT(enc)

    time = 0.5
    tau = enc.alpha * time
    cos_phases, sin_phases = _qsppack_hamiltonian_simulation_phases(tau, 12)

    rng = np.random.default_rng(21)
    ket = rng.normal(size=4).astype(np.complex128)
    ket /= np.linalg.norm(ket)

    cos_state = transformer.transform(
        ket, qsvt.PhaseSequence(cos_phases, convention="qsp"))
    sin_state = transformer.transform(
        ket, qsvt.PhaseSequence(sin_phases, convention="qsp"))
    evolved = qsvt.recover_real_time_evolution(cos_state, sin_state,
                                               cos_phases, sin_phases)

    matrix = dense_matrix([(c, w) for w, c in terms.items()], 2)
    eigenvalues, eigenvectors = np.linalg.eigh(matrix)
    exact = eigenvectors @ (np.exp(-1.0j * time * eigenvalues) *
                            (eigenvectors.conj().T @ ket))
    assert np.linalg.norm(evolved - exact) < 1e-8


def test_controlled_sequence_respects_control():
    enc = lcu.PauliLCU(FOUR_TERMS_2Q)
    transformer = qsvt.QSVT(enc)
    seq = qsvt.PhaseSequence([0.3, -0.4, 0.25],
                             walk_directions=["forward", "adjoint"])
    ket = random_ket(2, seed=23)
    ns, na = enc.num_system, enc.num_ancilla

    def con_index(sys, ctrl, anc):
        return sys + (ctrl << ns) + (anc << (ns + 1))

    reference = np.asarray(
        cudaq.get_state(transformer.kernel(seq), lcu.state_from(ket)))
    on_state = np.asarray(
        cudaq.get_state(transformer.controlled_kernel(seq, control_state=1),
                        lcu.state_from(ket)))
    off_state = np.asarray(
        cudaq.get_state(transformer.controlled_kernel(seq, control_state=0),
                        lcu.state_from(ket)))

    for anc in range(1 << na):
        for sys in range(1 << ns):
            unc = reference[sys + (anc << ns)]
            assert on_state[con_index(sys, 1, anc)] == pytest.approx(
                unc, abs=1e-10)
            assert abs(on_state[con_index(sys, 0, anc)]) < 1e-10
            expected = ket[sys] if anc == 0 else 0.0
            assert off_state[con_index(sys, 0, anc)] == pytest.approx(
                expected, abs=1e-10)
            assert abs(off_state[con_index(sys, 1, anc)]) < 1e-10
