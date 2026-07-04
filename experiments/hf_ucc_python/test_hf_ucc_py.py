# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                        #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""Tests for the pure-Python HF + fixed-parameter UCC prototype.

Mirrors tests/python/test_hf_fixed_parameter_ucc.py from the
add_hf_fixed_param_ucc_state_prep branch, plus coverage for the ported
operator pools and the prototype-only plan surface.
"""

import os
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

import cudaq

import hf_ucc_py as stateprep

# Override with e.g. LCU_PY_TARGET=nvidia-fp64 to run on a GPU simulator.
SIMULATION_TARGET = os.environ.get("LCU_PY_TARGET", "qpp-cpu")


@pytest.fixture(autouse=True)
def simulation_target():
    cudaq.set_target(SIMULATION_TARGET)
    yield
    cudaq.reset_target()


def _assert_basis_state(state, index):
    expected = np.zeros_like(state, dtype=complex)
    expected[index] = 1.0
    np.testing.assert_allclose(state, expected, atol=1.0e-12)


def _operator_exp_evolve(operator_pool, parameters, ket, num_qubits):
    """Independent dense reference: prod_g exp(+i theta_g G_g) |ket>."""
    full_identity = cudaq.spin.i(0)
    for qubit in range(1, num_qubits):
        full_identity = full_identity * cudaq.spin.i(qubit)

    state = np.array(ket, dtype=np.complex128)
    for theta, op in zip(parameters, operator_pool):
        generator = np.array((op * full_identity).to_matrix(),
                             dtype=np.complex128)
        eigenvalues, eigenvectors = np.linalg.eigh(generator)
        propagator = eigenvectors @ (np.exp(
            1.0j * theta * eigenvalues)[:, None] * eigenvectors.conj().T)
        state = propagator @ state
    return state


# ----------------------------------------------------------------------------
# Hartree-Fock
# ----------------------------------------------------------------------------


def test_hartree_fock_host_helpers():
    occupation = stateprep.make_hartree_fock_occupation(6, 4)
    assert occupation == [0, 1, 2, 3]

    resources = stateprep.estimate_hartree_fock_resources(6, 4)
    assert resources.num_qubits == 6
    assert resources.num_electrons == 4
    assert resources.num_x_gates == 4

    explicit = stateprep.estimate_hartree_fock_occupation_resources(
        6, [0, 2, 5])
    assert explicit.num_electrons == 3
    assert explicit.num_x_gates == 3

    with pytest.raises(ValueError, match="num_electrons"):
        stateprep.make_hartree_fock_occupation(2, 3)
    with pytest.raises(ValueError, match="exceeds"):
        stateprep.validate_hartree_fock_occupation(4, [0, 4])
    with pytest.raises(ValueError, match="unique"):
        stateprep.validate_hartree_fock_occupation(4, [0, 2, 2])


def test_hartree_fock_canonical_statevector():

    @cudaq.kernel
    def kernel():
        q = cudaq.qvector(4)
        stateprep.hartree_fock(q, 3)

    state = np.asarray(cudaq.get_state(kernel), dtype=complex)
    _assert_basis_state(state, 0b0111)


def test_hartree_fock_explicit_occupation_statevector():
    occupation = [0, 2]

    @cudaq.kernel
    def kernel(occupied_orbitals: list[int]):
        q = cudaq.qvector(4)
        stateprep.hartree_fock_occupation(q, occupied_orbitals)

    state = np.asarray(cudaq.get_state(kernel, occupation), dtype=complex)
    _assert_basis_state(state, 0b0101)


def test_hartree_fock_open_shell_occupation():
    # Closed shell stays contiguous.
    assert stateprep.make_hartree_fock_occupation(8, 4, 0) == [0, 1, 2, 3]

    # Open shell (4 electrons, spin=2) must use the interleaved alpha/beta
    # convention of get_uccsd_excitations: {0, 1, 2, 4}, NOT {0, 1, 2, 3}.
    occupation = stateprep.make_hartree_fock_occupation(8, 4, 2)
    assert occupation == [0, 1, 2, 4]

    # Cross-check against the determinant implied by the UCCSD excitations.
    singles_alpha, singles_beta, _, _, _ = stateprep.get_uccsd_excitations(
        8, 4, 2)
    implied = sorted({s[0] for s in singles_alpha}
                     | {s[0] for s in singles_beta})
    assert occupation == implied

    @cudaq.kernel
    def kernel(occupied_orbitals: list[int]):
        q = cudaq.qvector(8)
        stateprep.hartree_fock_occupation(q, occupied_orbitals)

    state = np.asarray(cudaq.get_state(kernel, occupation), dtype=complex)
    _assert_basis_state(state, 0b00010111)  # qubits {0,1,2,4} set

    with pytest.raises(ValueError, match="even"):
        stateprep.make_hartree_fock_occupation(7, 3, 1)
    with pytest.raises(ValueError, match="spin"):
        stateprep.make_hartree_fock_occupation(8, 2, 3)


# ----------------------------------------------------------------------------
# Plans and pools
# ----------------------------------------------------------------------------


def test_fixed_parameter_ucc_plan_helpers():
    words, coeffs = stateprep.get_uccgsd_pauli_lists(4)
    parameters = [0.05 * (i + 1) for i in range(len(words))]

    plan = stateprep.make_fixed_parameter_ucc_plan(words, coeffs, parameters,
                                                   4)
    assert plan.num_qubits == 4
    assert plan.parameters == pytest.approx(parameters)
    assert len(plan.pauli_words) == len(parameters)
    assert len(plan.coefficients) == len(parameters)

    resources = stateprep.estimate_fixed_parameter_ucc_resources(plan)
    assert resources.num_qubits == 4
    assert resources.num_excitations == len(parameters)
    assert resources.num_pauli_rotations == sum(len(g) for g in words)
    assert resources.max_pauli_rotations_per_excitation == 8


def test_fixed_parameter_uccsd_plan_helper():
    parameters = [0.1, -0.2, 0.3]
    plan = stateprep.make_fixed_parameter_uccsd_plan(4, 2, parameters)
    assert plan.num_qubits == 4
    assert plan.parameters == pytest.approx(parameters)
    assert len(plan.pauli_words) == 3
    assert len(plan.coefficients) == 3

    with pytest.raises(ValueError, match="same length"):
        stateprep.make_fixed_parameter_uccsd_plan(4, 2, [0.1, 0.2])


def test_fixed_parameter_ucc_kernel_accepts_plan_data():
    words, coeffs = stateprep.get_upccgsd_pauli_lists(4)
    parameters = [0.1, -0.2, 0.3]
    plan = stateprep.make_fixed_parameter_upccgsd_plan(4, parameters)

    assert len(plan.pauli_words) == len(words)
    assert [len(g) for g in plan.pauli_words] == [len(g) for g in words]
    assert plan.coefficients == coeffs

    @cudaq.kernel
    def kernel(thetas: list[float],
               pauli_words: list[list[cudaq.pauli_word]],
               coefficients: list[list[float]]):
        q = cudaq.qvector(4)
        stateprep.hartree_fock(q, 2)
        stateprep.fixed_parameter_ucc(q, thetas, pauli_words, coefficients)

    state = np.asarray(cudaq.get_state(kernel, plan.parameters,
                                       plan.pauli_words, plan.coefficients),
                       dtype=complex)
    assert np.isclose(np.linalg.norm(state), 1.0)


def test_fixed_parameter_ucc_matches_dense_operator_reference():
    # Independent check of the UCC kernel against dense matrix exponentials
    # of the operator pool (not against another cudaq kernel).
    num_qubits, num_electrons = 4, 2
    pool = stateprep.make_uccsd_operator_pool(num_qubits, num_electrons, 0)
    parameters = [0.13 * (i + 1) for i in range(len(pool))]
    plan = stateprep.make_fixed_parameter_uccsd_plan(num_qubits,
                                                     num_electrons,
                                                     parameters)

    @cudaq.kernel
    def hf_only():
        q = cudaq.qvector(4)
        stateprep.hartree_fock(q, 2)

    @cudaq.kernel
    def hf_then_ucc(thetas: list[float],
                    pauli_words: list[list[cudaq.pauli_word]],
                    coefficients: list[list[float]]):
        q = cudaq.qvector(4)
        stateprep.hartree_fock(q, 2)
        stateprep.fixed_parameter_ucc(q, thetas, pauli_words, coefficients)

    hf_ket = np.asarray(cudaq.get_state(hf_only), dtype=np.complex128)
    expected = _operator_exp_evolve(pool, parameters, hf_ket, num_qubits)
    actual = np.asarray(cudaq.get_state(hf_then_ucc, plan.parameters,
                                        plan.pauli_words, plan.coefficients),
                        dtype=np.complex128)
    np.testing.assert_allclose(actual, expected, atol=1.0e-9)


def test_fixed_parameter_ucc_plan_composes_with_validate_and_estimate():
    words, coeffs = stateprep.get_uccgsd_pauli_lists(4)
    parameters = [0.05 * (i + 1) for i in range(len(words))]
    plan = stateprep.make_fixed_parameter_ucc_plan(words, coeffs, parameters,
                                                   4)

    stateprep.validate_fixed_parameter_ucc_plan(plan)
    resources = stateprep.estimate_fixed_parameter_ucc_resources(plan)
    assert resources.num_excitations == len(parameters)

    uccsd_plan = stateprep.make_fixed_parameter_uccsd_plan(
        4, 2, [0.1, -0.2, 0.3])
    stateprep.validate_fixed_parameter_ucc_plan(uccsd_plan)


def test_uccsd_pool_matches_expected_counts():
    # 4 qubits / 2 electrons closed shell: 1 alpha single, 1 beta single,
    # 1 mixed double -> 3 excitation groups (matches the uccsd plan test).
    pool = stateprep.make_uccsd_operator_pool(4, 2, 0)
    assert len(pool) == 3

    singles_alpha, singles_beta, mixed, d_alpha, d_beta = (
        stateprep.get_uccsd_excitations(4, 2, 0))
    assert singles_alpha == [[0, 2]]
    assert singles_beta == [[1, 3]]
    assert mixed == [[0, 1, 3, 2]]
    assert d_alpha == [] and d_beta == []


def test_pool_generators_produce_anti_hermitian_generators():
    # Every pool operator G must satisfy G^dagger = -G (i G Hermitian), so
    # exp(theta G) is unitary. Checked densely on 4 qubits.
    full_identity = cudaq.spin.i(0)
    for qubit in range(1, 4):
        full_identity = full_identity * cudaq.spin.i(qubit)

    pools = (stateprep.make_uccsd_operator_pool(4, 2, 0),
             stateprep.make_uccgsd_operator_pool(4),
             stateprep.make_upccgsd_operator_pool(4))
    for pool in pools:
        assert len(pool) > 0
        for op in pool:
            matrix = np.array((op * full_identity).to_matrix(),
                              dtype=np.complex128)
            # cudaq spin ops store i*G effectively real-coefficient Pauli
            # sums; the generator used is exp(i theta M) with M Hermitian.
            np.testing.assert_allclose(matrix, matrix.conj().T, atol=1e-12)


# ----------------------------------------------------------------------------
# Prototype-only surface: plan.kernel() / plan.state()
# ----------------------------------------------------------------------------


def test_plan_factory_matches_nested_argument_kernel():
    parameters = [0.1, -0.2, 0.3]
    plan = stateprep.make_fixed_parameter_uccsd_plan(4, 2, parameters)

    @cudaq.kernel
    def manual(thetas: list[float],
               pauli_words: list[list[cudaq.pauli_word]],
               coefficients: list[list[float]]):
        q = cudaq.qvector(4)
        stateprep.hartree_fock(q, 2)
        stateprep.fixed_parameter_ucc(q, thetas, pauli_words, coefficients)

    manual_state = np.asarray(cudaq.get_state(manual, plan.parameters,
                                              plan.pauli_words,
                                              plan.coefficients),
                              dtype=complex)
    factory_state = plan.state(num_electrons=2)
    np.testing.assert_allclose(factory_state, manual_state, atol=1e-12)


def test_plan_factory_open_shell_occupation():
    pool_size = len(stateprep.make_uccsd_operator_pool(8, 4, 2))
    plan = stateprep.make_fixed_parameter_uccsd_plan(
        8, 4, [0.0] * pool_size, spin_number=2)
    occupation = stateprep.make_hartree_fock_occupation(8, 4, 2)
    state = plan.state(occupied_orbitals=occupation)
    _assert_basis_state(state, 0b00010111)


def test_plan_factory_requires_exactly_one_reference_spec():
    plan = stateprep.make_fixed_parameter_uccsd_plan(4, 2, [0.1, -0.2, 0.3])
    with pytest.raises(ValueError, match="exactly one"):
        plan.kernel()
    with pytest.raises(ValueError, match="exactly one"):
        plan.kernel(num_electrons=2, occupied_orbitals=[0, 1])
