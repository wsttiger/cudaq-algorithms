#!/usr/bin/env python3
# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                        #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""Hartree-Fock + fixed-parameter UCCSD state preparation (pure Python).

Prepares a 4-qubit / 2-electron UCCSD ansatz state at fixed amplitudes
(H2-style; in a real workflow the parameters would come from a classical
pre-optimization) and validates it against dense matrix exponentials of the
operator pool. Also demonstrates the open-shell Hartree-Fock occupation.

Run with:  PYTHONPATH=/path/to/cudaq python3 example_hf_ucc.py
"""

import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import cudaq

import hf_ucc_py as stateprep


def dense_reference(pool, parameters, ket, num_qubits):
    full_identity = cudaq.spin.i(0)
    for qubit in range(1, num_qubits):
        full_identity = full_identity * cudaq.spin.i(qubit)
    state = np.array(ket, dtype=np.complex128)
    for theta, op in zip(parameters, pool):
        generator = np.array((op * full_identity).to_matrix(),
                             dtype=np.complex128)
        eigenvalues, eigenvectors = np.linalg.eigh(generator)
        state = eigenvectors @ (np.exp(1.0j * theta * eigenvalues) *
                                (eigenvectors.conj().T @ state))
    return state


def main():
    cudaq.set_target(os.environ.get("LCU_PY_TARGET", "qpp-cpu"))

    num_qubits, num_electrons = 4, 2
    parameters = [0.1129, -0.0421, 0.2839]  # fixed UCCSD amplitudes

    plan = stateprep.make_fixed_parameter_uccsd_plan(num_qubits,
                                                     num_electrons,
                                                     parameters)
    resources = plan.resources()

    # The whole quantum workflow: one line.
    state = plan.state(num_electrons=num_electrons)

    pool = stateprep.make_uccsd_operator_pool(num_qubits, num_electrons)
    hf_ket = np.zeros(1 << num_qubits, dtype=np.complex128)
    hf_ket[0b0011] = 1.0
    expected = dense_reference(pool, parameters, hf_ket, num_qubits)
    error = float(np.linalg.norm(state - expected))

    print("Fixed-parameter UCCSD state preparation (pure-Python prototype)")
    print("=" * 64)
    print(f"qubits:                {plan.num_qubits}")
    print(f"electrons:             {num_electrons}")
    print(f"excitation groups:     {resources.num_excitations}")
    print(f"pauli rotations:       {resources.num_pauli_rotations}")
    print(f"max rotations/group:   {resources.max_pauli_rotations_per_excitation}")
    print(f"L2 error vs dense:     {error:.3e}")
    print("dominant amplitudes:")
    for index in np.argsort(np.abs(state))[::-1][:4]:
        print(f"  |{index:04b}> {state[index].real:+.6f}"
              f"{state[index].imag:+.6f}j "
              f"(P = {abs(state[index])**2:.4f})")

    print()
    print("Open-shell Hartree-Fock (8 qubits, 4 electrons, spin 2):")
    occupation = stateprep.make_hartree_fock_occupation(8, 4, 2)
    print(f"  occupied spin orbitals: {occupation}  (interleaved "
          "alpha/beta, NOT the contiguous {0,1,2,3})")

    if error > 1e-9:
        raise SystemExit("UCC state does not match the dense reference")
    print("PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
