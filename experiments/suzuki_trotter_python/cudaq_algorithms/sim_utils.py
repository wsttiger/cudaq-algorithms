# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                        #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""Simulation-only helpers.

Everything here depends on statevector access (``cudaq.get_state``), which
only exists on simulators. The module ships with the package as a
clearly-labeled companion, but it is not part of the hardware-shaped API:
the library classes (plans, kernel factories, resource estimators) never
execute ``get_state``.
"""

from __future__ import annotations

import cudaq

from .trotter import TrotterPlan, state_from

__all__ = ["evolve", "state_from"]


def evolve(plan: TrotterPlan, ket, include_identity_phase: bool = True):
    """Simulate a Trotter plan on ``ket`` and return the evolved statevector.

    Unlike the circuit primitive, this can reintroduce the identity phase
    ``exp(-i * identity_coefficient * time)`` (on by default), so the
    result approximates the full ``exp(-i H t)|ket>``.

    The plan's kernel prepares |0...0> and evolves; to evolve an arbitrary
    ``ket``, the input state is loaded through ``cudaq.get_state``'s
    initial-state support via a state-taking wrapper kernel.
    """
    import numpy as np

    num_qubits = int(plan.num_qubits)
    coefficients = [float(c) for c in plan.coefficients]
    words = [cudaq.pauli_word(str(w)) for w in plan.words]
    time = float(plan.time)
    steps = int(plan.steps)
    order = int(plan.order)

    if words:
        from .trotter import apply_trotter

        @cudaq.kernel
        def evolve_state(state: cudaq.State):
            qubits = cudaq.qvector(state)
            apply_trotter(coefficients, words, time, steps, order, qubits)

        state = np.asarray(cudaq.get_state(evolve_state, state_from(ket)),
                           dtype=np.complex128)
    else:
        state = np.asarray(ket, dtype=np.complex128).copy()

    if include_identity_phase and plan.identity_coefficient != 0.0:
        state = state * np.exp(-1.0j * plan.identity_coefficient * time)
    return state
