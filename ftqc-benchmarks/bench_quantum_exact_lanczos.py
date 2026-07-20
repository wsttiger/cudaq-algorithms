#!/usr/bin/env python3
# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""FTQC benchmark: quantum exact Lanczos (QEL) ground-state energy.

Full-stack path exercised: frozen chemist integrals -> spin expansion +
Jordan-Wigner (``chemistry.qubit_hamiltonian``) -> ``PauliLCU`` block
encoding -> qubitization ``Walk`` Chebyshev moments <T_k(H/alpha)>
(even orders via the reflection observable, odd orders via the SELECT
observable — observables and sampling only, no statevector access) ->
classical Chebyshev-Krylov generalized eigenproblem -> ground-state
energy against the frozen FCI reference.

Reference: Kirby, Motta, Mezzacapo, "Exact and efficient Lanczos method
on a quantum computer", Quantum 7, 1018 (2023), arXiv:2208.00567.

The Krylov basis is |phi_i> = T_i(H/alpha)|HF>, i = 0..d-1. With
mu_k = <HF|T_k(H/alpha)|HF> and the Chebyshev product identities
``T_i T_j = (T_{i+j} + T_{|i-j|})/2`` and ``x T_j = (T_{j+1} +
T_{|j-1|})/2``, the overlap and Hamiltonian matrices are moment
combinations — 2d moments give a d-dimensional Krylov solve.

Usage: python3 bench_quantum_exact_lanczos.py [--molecule h2]
       [--krylov-dimension 4] [--overlap-cutoff 1e-8] [--tolerance 1e-9]
"""

from __future__ import annotations

import argparse
import os

import cudaq
import numpy as np

from cudaq_algorithms import PauliLCU, Walk

from _common import (MOLECULES, Report, dense_matrix, electronic_hamiltonian,
                     hartree_fock_ket, stopwatch)


def krylov_matrices(moments: np.ndarray, dimension: int):
    """Overlap S and scaled-Hamiltonian Htilde from Chebyshev moments.

    ``moments[k]`` must be <T_k(H/alpha)>, k = 0..2*dimension-1. The
    returned ``Htilde`` is in H/alpha units.
    """
    if len(moments) < 2 * dimension:
        raise ValueError("need 2*dimension moments")
    mu = moments
    overlap = np.empty((dimension, dimension))
    scaled = np.empty((dimension, dimension))
    for i in range(dimension):
        for j in range(dimension):
            overlap[i, j] = 0.5 * (mu[i + j] + mu[abs(i - j)])
            scaled[i,
                   j] = 0.25 * (mu[i + j + 1] + mu[abs(i - j - 1)] +
                                mu[i + abs(j - 1)] + mu[abs(i - abs(j - 1))])
    return overlap, scaled


def conditioned_ground_eigenvalue(scaled: np.ndarray, overlap: np.ndarray,
                                  cutoff: float):
    """Filter near-null overlap directions, solve the projected problem."""
    overlap_eigenvalues, overlap_eigenvectors = np.linalg.eigh(overlap)
    keep = overlap_eigenvalues > cutoff
    if not np.any(keep):
        raise RuntimeError("overlap matrix is numerically singular")
    transform = (overlap_eigenvectors[:, keep] /
                 np.sqrt(overlap_eigenvalues[keep]))
    conditioned = transform.T @ scaled @ transform
    eigenvalues = np.linalg.eigvalsh(conditioned)
    return float(eigenvalues.min()), int(np.count_nonzero(keep)), float(
        overlap_eigenvalues[keep].max() / overlap_eigenvalues[keep].min())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--molecule", default="h2", choices=sorted(MOLECULES))
    parser.add_argument("--krylov-dimension", type=int, default=4)
    parser.add_argument("--overlap-cutoff", type=float, default=1e-8)
    parser.add_argument("--tolerance", type=float, default=1e-9)
    args = parser.parse_args()

    cudaq.set_target(os.environ.get("CUDAQ_DEFAULT_SIMULATOR", "qpp-cpu"))
    molecule = MOLECULES[args.molecule]
    dimension = args.krylov_dimension
    report = Report(f"Quantum exact Lanczos — {molecule.description}, "
                    f"Krylov dimension {dimension}")

    # -- classical preprocessing ------------------------------------------
    with stopwatch() as t_build:
        hamiltonian = electronic_hamiltonian(molecule)
        encoding = PauliLCU(hamiltonian)
        walk = Walk(encoding)
    report.add("qubits (system + ancilla)",
               f"{encoding.num_system} + {encoding.num_ancilla}")
    report.add("LCU terms / alpha",
               f"{hamiltonian.term_count} / {encoding.alpha:.6f}")
    report.add("build time", f"{t_build.seconds:.3f} s")

    # -- quantum workflow: 2d Chebyshev moments ----------------------------
    psi = hartree_fock_ket(molecule)
    num_moments = 2 * dimension
    with stopwatch() as t_moments:
        moments = np.asarray(walk.moments(psi, num_moments))
    report.add(f"moments <T_0..T_{num_moments - 1}> collection",
               f"{t_moments.seconds:.3f} s")
    report.add("moments", "  ".join(f"{m:+.6f}" for m in moments))

    # -- classical postprocessing ------------------------------------------
    overlap, scaled = krylov_matrices(moments, dimension)
    ground_scaled, kept_rank, condition = conditioned_ground_eigenvalue(
        scaled, overlap, args.overlap_cutoff)
    qel_energy = ground_scaled * encoding.alpha + molecule.nuclear_repulsion
    report.add("kept Krylov rank / condition",
               f"{kept_rank} / {condition:.3e}")
    report.add("QEL energy", f"{qel_energy:.12f} Ha")
    report.add("frozen FCI reference", f"{molecule.fci_energy:.12f} Ha")

    # -- checks --------------------------------------------------------------
    matrix = dense_matrix(hamiltonian, molecule.num_qubits)
    dense_ground = float(
        np.min(np.linalg.eigvalsh(matrix)) + molecule.nuclear_repulsion)
    report.check("dense qubit-H ground vs frozen FCI (stack sanity)",
                 abs(dense_ground - molecule.fci_energy), 1e-10)
    report.check("|E_QEL - E_FCI|", abs(qel_energy - molecule.fci_energy),
                 args.tolerance)
    return report.render()


if __name__ == "__main__":
    raise SystemExit(main())
