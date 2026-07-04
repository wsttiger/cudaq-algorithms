# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""Pure-Python QSVT on top of the PauliLCU prototype.

A ``PhaseSequence`` value type (with the qsvt/qsp phase-convention handling
built in), the phase/walk sequence kernel, a ``QSVT`` object with a kernel
factory and a one-call ``transform``, and the host-side 2x2 response model
for validating phases.

Composition matches the verified library construction: each walk step is the
full block encoding (PREPARE, SELECT, PREPARE dagger) composed with a
reflection about the all-zero signal state, and projector phases
``diag(e^{i phi}, 1)`` act on the same |0...0> signal subspace. The signal
register starts at |0...0>.

Sign convention (deliberately simpler than the C++ host helper): the walk's
``-H/alpha`` sign is folded INTO the step of ``evaluate_response``, so ``x``
is the plain scaled eigenvalue ``eigenvalue / alpha`` — the circuit's good
subspace equals ``evaluate_response(sequence, eigenvalue / alpha)`` times the
eigenvector, with no caller-side negation.
"""

from __future__ import annotations

import math

import cudaq

from pauli_lcu_py import PauliLCU, reflect_about_zero
from pauli_lcu_py import apply as lcu_apply

FORWARD = 0
ADJOINT = 1

_DIRECTION_CODES = {
    "forward": FORWARD,
    "adjoint": ADJOINT,
    "backward": ADJOINT,
    "reverse": ADJOINT,
    FORWARD: FORWARD,
    ADJOINT: ADJOINT,
}


def _direction_code(direction):
    key = direction.lower() if isinstance(direction, str) else direction
    try:
        return _DIRECTION_CODES[key]
    except (KeyError, TypeError) as exc:
        raise ValueError(
            "walk direction must be 'forward', 'adjoint', 0, or 1") from exc


# ============================================================================
# Phase sequences
# ============================================================================


class PhaseSequence:
    """A validated QSVT/QSP phase sequence.

    Parameters
    ----------
    phases
        d + 1 phase angles for a degree-d polynomial.
    walk_directions
        Optional; one direction ('forward'/'adjoint' or 0/1) per walk,
        length d. Defaults to all forward.
    convention
        "qsvt" (projector phases ``diag(e^{i phi}, 1)``, the default) or
        "qsp" (Z-rotation phases ``diag(e^{i phi}, e^{-i phi})``, the
        QSPPACK convention). qsp-tagged phases are converted automatically
        wherever a circuit is built; ``phases`` always stays raw.
    """

    def __init__(self, phases, walk_directions=None, convention="qsvt"):
        self.phases = tuple(float(p) for p in phases)
        if not self.phases:
            raise ValueError("phases must contain at least one value")
        if not all(math.isfinite(p) for p in self.phases):
            raise ValueError("phases must be finite")

        convention = str(convention).lower()
        if convention not in ("qsvt", "qsp"):
            raise ValueError("convention must be 'qsvt' or 'qsp'")
        self.convention = convention

        if walk_directions is None:
            self.walk_directions = (FORWARD,) * self.degree
        else:
            self.walk_directions = tuple(
                _direction_code(d) for d in walk_directions)
            if len(self.walk_directions) != self.degree:
                raise ValueError(
                    "walk_directions must contain len(phases) - 1 entries")

    @property
    def degree(self) -> int:
        return len(self.phases) - 1

    @property
    def projector_phases(self) -> list[float]:
        """Phases in the projector convention the circuits implement.

        qsp phases are doubled (equivalent up to a global phase of
        ``exp(i * sum(phases))``; see ``recover_real_time_evolution``).
        """
        if self.convention == "qsp":
            return [2.0 * p for p in self.phases]
        return list(self.phases)

    def __repr__(self):
        return (f"PhaseSequence(degree={self.degree}, "
                f"convention={self.convention!r})")


def _as_sequence(sequence, convention=None) -> PhaseSequence:
    if isinstance(sequence, PhaseSequence):
        if convention is not None and convention != sequence.convention:
            return PhaseSequence(sequence.phases, sequence.walk_directions,
                                 convention)
        return sequence
    return PhaseSequence(sequence, convention=convention or "qsvt")


# ============================================================================
# Device kernels
# ============================================================================


@cudaq.kernel
def signal_phase(register: cudaq.qview, phase: float):
    """exp(i * phase * |0...0><0...0|) on the signal register."""
    n = register.size()
    if n == 0:
        return
    for i in range(n):
        x(register[i])
    if n == 1:
        r1(phase, register[0])
    else:
        r1.ctrl(phase, register.front(n - 1), register[n - 1])
    for i in range(n):
        x(register[i])


@cudaq.kernel
def apply_phase_sequence(signal: cudaq.qview, system: cudaq.qview,
                         phases: list[float], walk_directions: list[int],
                         angles: list[float], term_controls: list[int],
                         term_ops: list[int], term_lengths: list[int],
                         term_signs: list[int]):
    """Projector-phase QSVT sequence: phase, then (walk step, phase) repeats.

    The signal register must start in |0...0>. A forward step is the full
    block encoding followed by the zero-state reflection; an adjoint step is
    the reverse (both factors are self-adjoint).
    """
    signal_phase(signal, phases[0])
    for i in range(1, len(phases)):
        if walk_directions[i - 1] == 1:
            reflect_about_zero(signal)
            lcu_apply(signal, system, angles, term_controls, term_ops,
                      term_lengths, term_signs)
        else:
            lcu_apply(signal, system, angles, term_controls, term_ops,
                      term_lengths, term_signs)
            reflect_about_zero(signal)
        signal_phase(signal, phases[i])


# ============================================================================
# The user-facing object
# ============================================================================


class QSVT:
    """Quantum singular value transformation for a PauliLCU encoding."""

    def __init__(self, encoding: PauliLCU):
        if encoding.num_ancilla == 0:
            raise ValueError(
                "QSVT requires an encoding with at least one ancilla "
                "(two or more LCU terms); the 0-ancilla case is degenerate")
        self.encoding = encoding

    def __repr__(self):
        return f"QSVT({self.encoding!r})"

    def kernel(self, sequence, convention=None):
        """A ``@cudaq.kernel(state)`` applying the phase/walk sequence.

        ``sequence`` may be a PhaseSequence or a plain list of phases
        (optionally with ``convention="qsp"``). The signal register is
        allocated in |0...0> after the system register.
        """
        seq = _as_sequence(sequence, convention)
        phases = seq.projector_phases
        # A degree-0 sequence has no walks; pad with one unused entry because
        # empty list kernel arguments cannot be marshaled.
        directions = list(seq.walk_directions) or [FORWARD]
        angles, controls, ops, lengths, signs = self.encoding.kernel_args
        n_anc = self.encoding.num_ancilla

        @cudaq.kernel
        def qsvt_kernel(state: cudaq.State):
            system = cudaq.qvector(state)
            signal = cudaq.qvector(n_anc)
            apply_phase_sequence(signal, system, phases, directions, angles,
                                 controls, ops, lengths, signs)

        return qsvt_kernel

    def transform(self, ket, sequence, convention=None):
        """Return the good-subspace state after the sequence (simulation).

        For an eigenstate of H with eigenvalue lambda this equals
        ``evaluate_response(sequence, lambda / alpha)`` times the input.
        """
        import numpy as np

        ket = np.asarray(ket, dtype=np.complex128)
        state = cudaq.get_state(self.kernel(sequence, convention),
                                cudaq.State.from_data(ket))
        return self.encoding.good_subspace(state)


# ============================================================================
# Host-side response model
# ============================================================================


def evaluate_response(sequence, x, convention=None) -> complex:
    """Scalar response of the sequence at scaled eigenvalue ``x`` in [-1, 1].

    This is the upper-left element of the 2x2 signal model of the circuit
    built by ``QSVT.kernel``: the good-subspace block of the device circuit
    acting on an eigenstate with eigenvalue ``lambda`` equals
    ``evaluate_response(sequence, lambda / alpha)`` times that eigenstate.
    The walk's -H/alpha sign is folded into the step, so no caller-side
    negation is needed.

    For qsp-convention sequences the device circuit (which runs doubled
    projector phases) differs from this model by the global phase
    ``exp(i * sum(phases))`` per sequence; ``recover_real_time_evolution``
    accounts for it.
    """
    import numpy as np

    seq = _as_sequence(sequence, convention)
    x = float(x)
    if abs(x) > 1.0:
        raise ValueError("x must lie in [-1, 1]")
    s = math.sqrt(max(0.0, 1.0 - x * x))

    # One forward step of the circuit on the 2D invariant subspace:
    # reflect_about_zero * block_encoding = diag(-1, 1) @ [[x, s], [s, -x]].
    step_forward = np.array([[-x, -s], [s, -x]], dtype=np.complex128)
    step_adjoint = step_forward.T.copy()

    def phase_matrix(phi):
        if seq.convention == "qsp":
            return np.diag(
                [np.exp(1.0j * phi), np.exp(-1.0j * phi)]).astype(complex)
        return np.diag([np.exp(1.0j * phi), 1.0]).astype(complex)

    matrix = phase_matrix(seq.phases[0])
    for i in range(1, len(seq.phases)):
        step = (step_adjoint
                if seq.walk_directions[i - 1] == ADJOINT else step_forward)
        matrix = phase_matrix(seq.phases[i]) @ step @ matrix
    return complex(matrix[0, 0])


def recover_real_time_evolution(cos_state, sin_state, cos_phases, sin_phases):
    """Combine cosine/sine QSP components into exp(-i H t)|psi>.

    ``cos_state`` and ``sin_state`` are good-subspace statevectors produced by
    running qsp-convention sequences through ``QSVT.transform`` (which
    executes doubled projector phases); the per-sequence global phase
    ``exp(i * sum(phases))`` is removed here. Valid for real Hamiltonians and
    real input states, where the cosine/sine parts live in the real/imaginary
    components. Simulation-validation helper.
    """
    import numpy as np

    cos_state = np.asarray(cos_state, dtype=np.complex128)
    sin_state = np.asarray(sin_state, dtype=np.complex128)
    cos_state = cos_state * np.exp(-1.0j * np.sum(cos_phases))
    sin_state = sin_state * np.exp(-1.0j * np.sum(sin_phases))
    return 2.0 * (cos_state.real + 1.0j * sin_state.imag)
