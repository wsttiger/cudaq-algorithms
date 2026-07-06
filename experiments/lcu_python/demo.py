#!/usr/bin/env python3
# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""Walkthrough of the pure-Python PauliLCU API surface.

Run with:  PYTHONPATH=/path/to/cudaq python3 demo.py
"""

import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import cudaq
from cudaq import spin

import pauli_lcu_py as lcu
import sim_utils as sim


def banner(title):
    print(f"\n=== {title} " + "=" * max(0, 60 - len(title)))


def main():
    cudaq.set_target(os.environ.get("LCU_PY_TARGET", "qpp-cpu"))

    banner("1. Construct from a plain dict (word -> coefficient)")
    enc = lcu.PauliLCU({"ZI": 0.70, "IZ": -0.43, "XX": 0.19, "YZ": 0.11})
    print(enc)
    print(f"terms:         {enc.terms}")
    print(f"alpha:         {enc.alpha}")
    print(f"constant term: {enc.constant_term}")

    banner("2. Or from a cudaq.SpinOperator — same encoding")
    h = (0.70 * spin.z(0) - 0.43 * spin.z(1) + 0.19 * spin.x(0) * spin.x(1) +
         0.11 * spin.y(0) * spin.z(1))
    print(lcu.PauliLCU(h, num_qubits=2))

    banner("3. One call: (H/alpha) |psi> via encode + postselect")
    rng = np.random.default_rng(7)
    psi = rng.normal(size=4).astype(np.complex128)
    psi /= np.linalg.norm(psi)
    good = sim.action(enc, psi)
    print(f"|| action(psi) ||           = {np.linalg.norm(good):.6f}")
    print(f"success probability         = {np.linalg.norm(good)**2:.6f}")

    banner("4. Kernel factories for sampling/observing workflows")
    kernel = enc.encode_kernel()
    state = cudaq.get_state(kernel, lcu.state_from(psi))
    print(f"full statevector dimension  = {len(np.asarray(state))}")
    print(f"good-subspace dimension     = {len(sim.good_subspace(enc, state))}")

    banner("5. Qubitization walks (Chebyshev moments)")
    moment_enc = lcu.PauliLCU({"I": 0.2, "X": 0.5, "Z": 0.3})
    ket = np.array([np.cos(0.35), np.sin(0.35)], dtype=np.complex128)
    for k in (1, 2, 3):
        walked = cudaq.get_state(moment_enc.walk_kernel(power=k),
                                 lcu.state_from(ket))
        p0 = float(np.sum(np.abs(sim.good_subspace(moment_enc, walked))**2))
        print(f"<T_{2*k}(H/alpha)> from the circuit = {2 * p0 - 1:+.10f}")

    banner("6. Escape hatch: compose inside your own kernel")
    angles, controls, ops, lengths, signs = enc.kernel_args
    n_anc = enc.num_ancilla

    @cudaq.kernel
    def custom(state: cudaq.State):
        system = cudaq.qvector(state)
        ancilla = cudaq.qvector(n_anc)
        lcu.prepare(ancilla, angles)
        lcu.select(ancilla, system, controls, ops, lengths, signs)
        lcu.unprepare(ancilla, angles)

    manual = sim.good_subspace(enc, cudaq.get_state(custom,
                                               lcu.state_from(psi)))
    print(f"manual composition matches action(): "
          f"{np.allclose(manual, good, atol=1e-12)}")

    banner("7. The zero-ancilla edge case just works")
    negative_single = lcu.PauliLCU({"XZ": -0.5})
    positive_single = lcu.PauliLCU({"XZ": +0.5})
    print(negative_single)
    opposite = np.allclose(sim.action(negative_single, psi),
                           -sim.action(positive_single, psi), atol=1e-12)
    print(f"-0.5*XZ encodes the OPPOSITE state of +0.5*XZ: {opposite}")
    print(f"action norm = {np.linalg.norm(sim.action(negative_single, psi)):.6f} "
          f"(single unitary Pauli word: exactly 1)")

    print()


if __name__ == "__main__":
    main()
