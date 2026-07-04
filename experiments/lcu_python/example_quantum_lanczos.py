#!/usr/bin/env python3
# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""Quantum Exact Lanczos with the pure-Python PauliLCU + qubitization prototype.

Measures Chebyshev moments <T_k(H/alpha)> with the QEL even/odd observable
convention (Walk.moments), builds the Krylov overlap/Hamiltonian matrices
classically, solves the filtered generalized eigenproblem, and compares the
ground-state energy with dense diagonalization.

Run with:  PYTHONPATH=/path/to/cudaq python3 example_quantum_lanczos.py
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import cudaq

import pauli_lcu_py as lcu
import qubitization_py as qub

HAMILTONIAN = {
    "ZI": 0.70,
    "IZ": -0.43,
    "XX": 0.19,
    "YZ": 0.11,
}
CONSTANT = -0.35  # scalar shift handled outside the encoding
KRYLOV_DIMENSION = 4
OVERLAP_CUTOFF = 1e-10


def chebyshev_krylov_matrices(moments, dimension):
    """Overlap and Hamiltonian matrices of the Chebyshev Krylov basis.

    With mu_k = <psi|T_k(A)|psi> and the product-to-sum identity
    T_i T_j = (T_{i+j} + T_|i-j|)/2:
        S_ij = (mu_{i+j} + mu_|i-j|) / 2
        H_ij = (mu_{i+j+1} + mu_|i+j-1| + mu_{|i-j|+1} + mu_||i-j|-1|) / 4
    """
    mu = list(moments)
    overlap = np.zeros((dimension, dimension))
    hamiltonian = np.zeros((dimension, dimension))
    for i in range(dimension):
        for j in range(dimension):
            overlap[i, j] = 0.5 * (mu[i + j] + mu[abs(i - j)])
            hamiltonian[i, j] = 0.25 * (
                mu[i + j + 1] + mu[abs(i + j - 1)] + mu[abs(i - j) + 1] +
                mu[abs(abs(i - j) - 1)])
    return overlap, hamiltonian


def solve_filtered(hamiltonian, overlap, cutoff):
    """Project out near-null overlap directions, then solve H c = E S c."""
    eigenvalues, eigenvectors = np.linalg.eigh(overlap)
    keep = eigenvalues > cutoff
    if not np.any(keep):
        raise RuntimeError("overlap matrix is numerically singular")
    transform = eigenvectors[:, keep] @ np.diag(eigenvalues[keep]**-0.5)
    return np.linalg.eigvalsh(transform.T @ hamiltonian @ transform)


def dense_ground_energy(terms, constant, num_qubits):
    from test_pauli_lcu_py import dense_matrix

    matrix = dense_matrix([(c, w) for w, c in terms.items()], num_qubits)
    return float(np.linalg.eigvalsh(matrix).min()) + constant


def main():
    cudaq.set_target("qpp-cpu")

    # -- the entire quantum workflow -------------------------------------
    encoding = lcu.PauliLCU(HAMILTONIAN)
    walk = qub.Walk(encoding)

    reference_state = np.zeros(1 << encoding.num_system, dtype=np.complex128)
    reference_state[1] = 1.0  # Hartree-Fock-style basis state

    num_moments = 2 * KRYLOV_DIMENSION
    moments = walk.moments(reference_state, num_moments)
    # ---------------------------------------------------------------------

    overlap, krylov_h = chebyshev_krylov_matrices(moments, KRYLOV_DIMENSION)
    scaled_eigenvalues = solve_filtered(krylov_h, overlap, OVERLAP_CUTOFF)
    qel_energy = float(scaled_eigenvalues.min()) * encoding.alpha + CONSTANT

    exact_energy = dense_ground_energy(HAMILTONIAN, CONSTANT,
                                       encoding.num_system)
    error = abs(qel_energy - exact_energy)

    print("Quantum Exact Lanczos (pure-Python prototype)")
    print("=" * 56)
    print(f"encoding:          {encoding}")
    print(f"Krylov dimension:  {KRYLOV_DIMENSION}")
    print(f"measured moments:  {np.array2string(np.asarray(moments), precision=8)}")
    print(f"QEL energy:        {qel_energy:.12f}")
    print(f"exact energy:      {exact_energy:.12f}")
    print(f"absolute error:    {error:.3e}")

    if error > 1e-6:
        raise SystemExit("QEL energy differs from dense exact by > 1e-6")
    print("PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
