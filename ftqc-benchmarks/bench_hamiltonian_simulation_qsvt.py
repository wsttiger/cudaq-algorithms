#!/usr/bin/env python3
# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""FTQC benchmark: Hamiltonian simulation of a molecule via QSVT.

Full-stack path exercised: frozen chemist integrals -> spin expansion +
Jordan-Wigner (``chemistry.qubit_hamiltonian``) -> ``PauliLCU`` block
encoding -> Jacobi-Anger phase sequences (QSPPACK, external) ->
``QSVT`` cos/sin circuits -> ``recover_real_time_evolution``.

Pass criterion: the evolved state matches the dense
``expm(-i H t)|HF>`` reference to ``--tolerance`` in max amplitude
error, and survival of the HF energy expectation is exact.

Usage: python3 bench_hamiltonian_simulation_qsvt.py [--molecule h2]
       [--time 0.8] [--degree 16] [--tolerance 1e-8]
"""

from __future__ import annotations

import argparse
import os

import cudaq
import numpy as np
from scipy.linalg import expm

from cudaq_algorithms import (PauliLCU, PhaseSequence, QSVT,
                              recover_real_time_evolution, sim_utils as sim)

from _common import (MOLECULES, Report, dense_matrix, electronic_hamiltonian,
                     hartree_fock_ket, stopwatch)


def qsppack_phases(tau, degree):
    """cos/sin QSP phases for exp(-i tau x) via Jacobi-Anger + QSPPACK."""
    import qsppack
    from scipy import special

    cos_coefficients = np.array([0.5 * special.jv(0, tau)] +
                                [((-1)**k) * special.jv(2 * k, tau)
                                 for k in range(1, degree // 2 + 1)])
    sin_coefficients = np.array([((-1)**k) * special.jv(2 * k + 1, tau)
                                 for k in range(degree // 2)])
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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--molecule", default="h2", choices=sorted(MOLECULES))
    parser.add_argument("--time", type=float, default=0.8)
    parser.add_argument("--degree", type=int, default=16)
    parser.add_argument("--tolerance", type=float, default=1e-8)
    args = parser.parse_args()

    cudaq.set_target(os.environ.get("CUDAQ_DEFAULT_SIMULATOR", "qpp-cpu"))
    molecule = MOLECULES[args.molecule]
    report = Report(f"Hamiltonian simulation via QSVT — {molecule.description}"
                    f", t={args.time}, degree={args.degree}")

    # -- classical preprocessing ------------------------------------------
    with stopwatch() as t_build:
        hamiltonian = electronic_hamiltonian(molecule)
        encoding = PauliLCU(hamiltonian)
        transformer = QSVT(encoding)
    tau = encoding.alpha * args.time
    report.add("qubits (system + ancilla)",
               f"{encoding.num_system} + {encoding.num_ancilla}")
    report.add("LCU terms / alpha",
               f"{hamiltonian.term_count} / {encoding.alpha:.6f}")
    report.add("tau = alpha * t", f"{tau:.6f}")
    report.add("build time", f"{t_build.seconds:.3f} s")

    with stopwatch() as t_phases:
        cos_phases, sin_phases = qsppack_phases(tau, args.degree)
    report.add("phase generation (QSPPACK)", f"{t_phases.seconds:.3f} s")

    # -- quantum workflow ---------------------------------------------------
    psi = hartree_fock_ket(molecule)
    with stopwatch() as t_circuits:
        cos_state = sim.transform(transformer, psi,
                                  PhaseSequence(cos_phases, convention="qsp"))
        sin_state = sim.transform(transformer, psi,
                                  PhaseSequence(sin_phases, convention="qsp"))
        evolved = recover_real_time_evolution(cos_state, sin_state, cos_phases,
                                              sin_phases)
    report.add("QSVT circuit simulation", f"{t_circuits.seconds:.3f} s")

    # -- reference ----------------------------------------------------------
    matrix = dense_matrix(hamiltonian, molecule.num_qubits)
    exact = expm(-1j * matrix * args.time) @ psi

    state_error = float(np.max(np.abs(evolved - exact)))
    fidelity = float(np.abs(np.vdot(exact, evolved)))
    hf_energy = float(np.real(np.vdot(psi, matrix @ psi)))
    evolved_energy = float(
        np.real(
            np.vdot(evolved, matrix @ evolved) / np.vdot(evolved, evolved)))
    report.add("fidelity |<exact|evolved>|", f"{fidelity:.12f}")
    report.check("max |amplitude error|", state_error, args.tolerance)
    report.check("energy conservation |E(t) - E(0)|",
                 abs(evolved_energy - hf_energy), 1e-8)
    return report.render()


if __name__ == "__main__":
    raise SystemExit(main())
