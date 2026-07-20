#!/usr/bin/env python3
# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""FTQC benchmark: Hamiltonian simulation of a molecule via Trotter.

Full-stack path exercised: frozen chemist integrals -> spin expansion +
Jordan-Wigner (``chemistry.qubit_hamiltonian``) -> ``Trotter`` product
formulas (orders 1/2/4) -> evolved statevector, against the dense
``expm(-i H t)|HF>`` reference.

Two pass criteria:
- accuracy: the finest step count of the highest order reaches
  ``--tolerance`` in max amplitude error;
- convergence: each order's measured error slope (log2 error vs log2
  steps) is within 0.5 of its theoretical order.

Together with ``bench_hamiltonian_simulation_qsvt.py`` this cross-checks
two independent circuit constructions against the same target.

Usage: python3 bench_hamiltonian_simulation_trotter.py [--molecule h2]
       [--time 0.8] [--tolerance 1e-6]
"""

from __future__ import annotations

import argparse
import os

import cudaq
import numpy as np
from scipy.linalg import expm

from cudaq_algorithms import Trotter, sim_utils as sim

from _common import (MOLECULES, Report, dense_matrix, electronic_hamiltonian,
                     hamiltonian_dict, hartree_fock_ket, stopwatch)

STEP_SWEEP = (1, 2, 4, 8, 16, 32)
ORDERS = (1, 2, 4)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--molecule", default="h2", choices=sorted(MOLECULES))
    parser.add_argument("--time", type=float, default=0.8)
    parser.add_argument("--tolerance", type=float, default=1e-6)
    args = parser.parse_args()

    cudaq.set_target(os.environ.get("CUDAQ_DEFAULT_SIMULATOR", "qpp-cpu"))
    molecule = MOLECULES[args.molecule]
    report = Report(f"Hamiltonian simulation via Trotter — "
                    f"{molecule.description}, t={args.time}")

    with stopwatch() as t_build:
        hamiltonian = electronic_hamiltonian(molecule)
        terms = hamiltonian_dict(hamiltonian, molecule.num_qubits)
        evolution = Trotter(terms)
    report.add("qubits / terms", f"{molecule.num_qubits} / {len(terms)}")
    report.add("build time", f"{t_build.seconds:.3f} s")

    psi = hartree_fock_ket(molecule)
    matrix = dense_matrix(hamiltonian, molecule.num_qubits)
    exact = expm(-1j * matrix * args.time) @ psi

    errors = {}
    with stopwatch() as t_sweep:
        for order in ORDERS:
            for steps in STEP_SWEEP:
                evolved = sim.evolve(evolution,
                                     psi,
                                     args.time,
                                     steps=steps,
                                     order=order)
                errors[order, steps] = float(np.max(np.abs(evolved - exact)))
    report.add("sweep time (18 evolutions)", f"{t_sweep.seconds:.3f} s")

    for order in ORDERS:
        resources = evolution.resources(steps=STEP_SWEEP[-1], order=order)
        row = "  ".join(f"{errors[order, steps]:.2e}" for steps in STEP_SWEEP)
        report.add(f"order {order} errors (steps {STEP_SWEEP})", row)
        report.add(f"order {order} pauli rotations at steps={STEP_SWEEP[-1]}",
                   f"{resources.pauli_rotations}")

        # Fit the convergence slope on the asymptotic (large-steps) side,
        # skipping saturated points near machine precision.
        points = [(np.log2(steps), np.log2(errors[order, steps]))
                  for steps in STEP_SWEEP if errors[order, steps] > 1e-12]
        if len(points) >= 3:
            xs, ys = zip(*points[-4:])
            slope = -np.polyfit(xs, ys, 1)[0]
            report.check(f"order {order} measured convergence slope",
                         abs(slope - order), 0.5)

    report.check(f"order {ORDERS[-1]} error at steps={STEP_SWEEP[-1]}",
                 errors[ORDERS[-1], STEP_SWEEP[-1]], args.tolerance)
    return report.render()


if __name__ == "__main__":
    raise SystemExit(main())
