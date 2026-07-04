# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""Pure-Python qubitization on top of the PauliLCU prototype.

Walk kernels, reflection/SELECT observables, and a ``Walk`` object that
measures Chebyshev moments ``<T_k(H/alpha)>`` with the QEL even/odd
convention — everything a quantum Lanczos workflow needs, with no compiled
bindings.

Conventions match the C++ library: one walk step is SELECT followed by a
reflection about the PREPARE state, the walk block is ``-H/alpha``, and
moments are measured as

* even ``k = 2p``:  reflection observable ``2|0..0><0..0| - I`` on the
  ancillas after PREPARE, p walks, UNPREPARE;
* odd ``k = 2p+1``: the SELECT observable after PREPARE and p walks
  (no UNPREPARE).
"""

from __future__ import annotations

import cudaq
from cudaq import spin

from pauli_lcu_py import (PauliLCU, prepare, reflect_about_zero, select,
                          unprepare, walk)

FORWARD = 0
ADJOINT = 1


# ============================================================================
# Device kernels
# ============================================================================


@cudaq.kernel
def reflect_about_prepare(ancilla: cudaq.qview, angles: list[float]):
    """Reflect about the PREPARE state: PREPARE†, zero reflection, PREPARE."""
    unprepare(ancilla, angles)
    reflect_about_zero(ancilla)
    prepare(ancilla, angles)


@cudaq.kernel
def adjoint_walk(ancilla: cudaq.qview, system: cudaq.qview,
                 angles: list[float], term_controls: list[int],
                 term_ops: list[int], term_lengths: list[int],
                 term_signs: list[int]):
    """Adjoint walk step: reflection first, then SELECT (both self-adjoint)."""
    reflect_about_prepare(ancilla, angles)
    select(ancilla, system, term_controls, term_ops, term_lengths, term_signs)


# ============================================================================
# Observables (system register at qubits [0, ns), ancillas at [ns, ns+na) —
# matching the factory kernels, which allocate the system register first)
# ============================================================================


def _bit_projector(qubit, bit):
    """|bit><bit| on one qubit as a spin operator."""
    sign = 1.0 - 2.0 * float(bit)
    return 0.5 * (spin.i(qubit) + sign * spin.z(qubit))


def reflection_observable(encoding: PauliLCU):
    """R = 2|0...0><0...0| - I on the ancilla register."""
    if encoding.num_ancilla == 0:
        raise ValueError("reflection observable needs at least one ancilla")
    offset = encoding.num_system
    projector = _bit_projector(offset, 0)
    for b in range(1, encoding.num_ancilla):
        projector = projector * _bit_projector(offset + b, 0)
    return 2.0 * projector - spin.i(offset)


def select_observable(encoding: PauliLCU):
    """The SELECT operator sum_i sign_i |i><i|_anc x P_i as an observable."""
    if encoding.num_ancilla == 0:
        raise ValueError("select observable needs at least one ancilla")
    offset = encoding.num_system
    n_anc = encoding.num_ancilla

    observable = None
    for index, (coefficient, word) in enumerate(encoding.terms):
        term = 1.0 if coefficient >= 0.0 else -1.0
        for b in range(n_anc):
            bit = (index >> (n_anc - 1 - b)) & 1
            term = term * _bit_projector(offset + b, bit)
        for qubit, label in enumerate(word):
            if label == "X":
                term = term * spin.x(qubit)
            elif label == "Y":
                term = term * spin.y(qubit)
            elif label == "Z":
                term = term * spin.z(qubit)
        observable = term if observable is None else observable + term
    return observable


# ============================================================================
# The user-facing object
# ============================================================================


class Walk:
    """Qubitization walk for a PauliLCU encoding.

    Provides walk/adjoint-walk kernel factories and Chebyshev-moment
    measurement in the QEL even/odd convention. Requires a non-degenerate
    encoding (at least one ancilla, i.e. two or more LCU terms).
    """

    def __init__(self, encoding: PauliLCU):
        if encoding.num_ancilla == 0:
            raise ValueError(
                "Walk requires an encoding with at least one ancilla "
                "(two or more LCU terms); a single-term encoding is the "
                "signed Pauli word itself")
        self.encoding = encoding

    def __repr__(self):
        return f"Walk({self.encoding!r})"

    # ------------------------------------------------------------------
    # Kernel factories
    # ------------------------------------------------------------------

    def _factory(self, power, uncompute, forward):
        angles, controls, ops, lengths, signs = self.encoding.kernel_args
        n_anc = self.encoding.num_ancilla
        steps = int(power)

        if forward and uncompute:

            @cudaq.kernel
            def walk_and_uncompute(state: cudaq.State):
                system = cudaq.qvector(state)
                ancilla = cudaq.qvector(n_anc)
                prepare(ancilla, angles)
                for _ in range(steps):
                    walk(ancilla, system, angles, controls, ops, lengths,
                         signs)
                unprepare(ancilla, angles)

            return walk_and_uncompute

        if forward:

            @cudaq.kernel
            def walk_prepared(state: cudaq.State):
                system = cudaq.qvector(state)
                ancilla = cudaq.qvector(n_anc)
                prepare(ancilla, angles)
                for _ in range(steps):
                    walk(ancilla, system, angles, controls, ops, lengths,
                         signs)

            return walk_prepared

        if uncompute:

            @cudaq.kernel
            def adjoint_and_uncompute(state: cudaq.State):
                system = cudaq.qvector(state)
                ancilla = cudaq.qvector(n_anc)
                prepare(ancilla, angles)
                for _ in range(steps):
                    adjoint_walk(ancilla, system, angles, controls, ops,
                                 lengths, signs)
                unprepare(ancilla, angles)

            return adjoint_and_uncompute

        @cudaq.kernel
        def adjoint_prepared(state: cudaq.State):
            system = cudaq.qvector(state)
            ancilla = cudaq.qvector(n_anc)
            prepare(ancilla, angles)
            for _ in range(steps):
                adjoint_walk(ancilla, system, angles, controls, ops, lengths,
                             signs)

        return adjoint_prepared

    def kernel(self, power: int = 1, uncompute: bool = True):
        """``@cudaq.kernel(state)``: PREPARE, W^power, optionally UNPREPARE."""
        return self._factory(power, uncompute, forward=True)

    def adjoint_kernel(self, power: int = 1, uncompute: bool = True):
        """``@cudaq.kernel(state)``: PREPARE, (W†)^power, optionally UNPREPARE."""
        return self._factory(power, uncompute, forward=False)

    def roundtrip_kernel(self, power: int = 1):
        """PREPARE, W^power, (W†)^power, UNPREPARE — the identity, for tests."""
        angles, controls, ops, lengths, signs = self.encoding.kernel_args
        n_anc = self.encoding.num_ancilla
        steps = int(power)

        @cudaq.kernel
        def roundtrip(state: cudaq.State):
            system = cudaq.qvector(state)
            ancilla = cudaq.qvector(n_anc)
            prepare(ancilla, angles)
            for _ in range(steps):
                walk(ancilla, system, angles, controls, ops, lengths, signs)
            for _ in range(steps):
                adjoint_walk(ancilla, system, angles, controls, ops, lengths,
                             signs)
            unprepare(ancilla, angles)

        return roundtrip

    # ------------------------------------------------------------------
    # Moment measurement (simulation-friendly, but observable-based:
    # the same circuits and operators run on hardware)
    # ------------------------------------------------------------------

    def moment(self, ket, order: int) -> float:
        """Measure the Chebyshev moment <T_order(H/alpha)> for |ket>."""
        from pauli_lcu_py import state_from

        order = int(order)
        if order < 0:
            raise ValueError("order must be non-negative")
        power = order // 2
        if order % 2 == 0:
            kernel = self.kernel(power=power, uncompute=True)
            observable = reflection_observable(self.encoding)
        else:
            kernel = self.kernel(power=power, uncompute=False)
            observable = select_observable(self.encoding)
        state = state_from(ket)
        return float(cudaq.observe(kernel, observable, state).expectation())

    def moments(self, ket, count: int) -> list[float]:
        """Measure moments <T_0>, ..., <T_{count-1}> for |ket>."""
        return [self.moment(ket, order) for order in range(int(count))]
