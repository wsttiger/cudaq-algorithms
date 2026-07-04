# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""Pure-Python Givens-rotation Slater determinant preparation.

A self-contained replication of the `given_rotation_state_prep_phase2`
branch implemented entirely in Python: the Givens elimination schedules
(real and complex), plan construction and validation, resource estimation,
and the state-preparation kernels — no compiled cudaq-algorithms bindings.

Functionality parity with the C++/bindings branch:

* ``apply_givens_rotation`` / ``apply_phase_givens_rotation`` /
  ``prepare_slater_determinant`` / ``prepare_complex_slater_determinant``
  device kernels with the upstream signatures and semantics (adjacent
  rotations only; invalid flattened inputs are no-ops).
* ``make_givens_rotation_schedule`` / ``make_slater_determinant_plan`` with
  automatic real/complex dispatch, the same orthonormality validation, and
  the same sign/phase conventions.
* ``validate_slater_determinant_plan`` and both resource estimators.

Prototype-style ergonomics on top (same philosophy as the LCU and Trotter
experiments): ``plan.kernel()`` returns a ready ``@cudaq.kernel()`` and
``plan.state()`` simulates it, so no caller has to thread the flattened
arrays by hand; ``plan.resources()`` wraps the estimator.

Kernel-language notes (see README): Python ``exp_pauli`` does not accept
individual qubit operands, but runtime-contiguous slices
(``qubits[a:a + 2]``) do work — and the library only ever emits adjacent
rotations, so the two-qubit C++ form maps onto slices exactly. Early
``return`` in Python kernels is silently ignored, so the no-op guards are
positive if-blocks.
"""

from __future__ import annotations

import cmath
import math
from dataclasses import dataclass, field

import cudaq

# ============================================================================
# Device kernels (module level, composable from user kernels)
# ============================================================================


@cudaq.kernel
def apply_givens_rotation(qubits: cudaq.qview, theta: float,
                          first_orbital: int, second_orbital: int):
    """Apply an adjacent real fermionic Givens rotation.

    CUDA-Q's built-in Givens convention maps |10> to
    cos(theta)|10> - sin(theta)|01>; the state-preparation convention here
    uses the opposite sign, so this inlines givens_rotation(-theta).
    Non-adjacent orbital pairs are a no-op, matching the C++ kernel.
    """
    if first_orbital + 1 == second_orbital:
        pair = qubits[first_orbital:first_orbital + 2]
        exp_pauli(0.5 * theta, pair, "YX")
        exp_pauli(-0.5 * theta, pair, "XY")
    elif second_orbital + 1 == first_orbital:
        pair = qubits[second_orbital:second_orbital + 2]
        exp_pauli(-0.5 * theta, pair, "YX")
        exp_pauli(0.5 * theta, pair, "XY")


@cudaq.kernel
def apply_phase_givens_rotation(qubits: cudaq.qview, theta: float,
                                phase: float, first_orbital: int,
                                second_orbital: int):
    """Apply an adjacent phase-aware fermionic Givens rotation."""
    apply_givens_rotation(qubits, theta, first_orbital, second_orbital)
    # rz(phase) is equivalent to exp(i * phase * n) up to global phase.
    rz(phase, qubits[second_orbital])


@cudaq.kernel
def prepare_slater_determinant(qubits: cudaq.qview,
                               orbital_indices: list[int],
                               angles: list[float], num_electrons: int):
    """Prepare a real Slater determinant from a flattened Givens schedule.

    Mismatched flattened inputs are a no-op. (Positive guard rather than an
    early return: `return` in Python kernels is silently ignored.)
    """
    if len(orbital_indices) == 2 * len(angles):
        for i in range(num_electrons):
            x(qubits[i])
        for i in range(len(angles)):
            apply_givens_rotation(qubits, angles[i], orbital_indices[2 * i],
                                  orbital_indices[2 * i + 1])


@cudaq.kernel
def prepare_complex_slater_determinant(qubits: cudaq.qview,
                                       orbital_indices: list[int],
                                       angles: list[float],
                                       phases: list[float],
                                       final_phases: list[float],
                                       num_electrons: int):
    """Prepare a complex Slater determinant from a flattened Givens schedule."""
    if (len(orbital_indices) == 2 * len(angles)
            and len(phases) == len(angles)
            and len(final_phases) >= num_electrons):
        for i in range(num_electrons):
            x(qubits[i])
        for i in range(num_electrons):
            rz(final_phases[i], qubits[i])
        for i in range(len(angles)):
            apply_phase_givens_rotation(qubits, angles[i], phases[i],
                                        orbital_indices[2 * i],
                                        orbital_indices[2 * i + 1])


def state_from(ket):
    """Build a cudaq.State at the current target's precision."""
    import numpy as np

    return cudaq.State.from_data(np.asarray(ket, dtype=cudaq.complex()))


# ============================================================================
# Host-side data types
# ============================================================================


@dataclass
class GivensRotation:
    first_orbital: int = 0
    second_orbital: int = 0
    theta: float = 0.0
    phase: float = 0.0


@dataclass
class GivensRotationSchedule:
    num_orbitals: int = 0
    num_electrons: int = 0
    rotations: list = field(default_factory=list)
    final_phases: list = field(default_factory=list)


@dataclass
class GivensStatePrepResources:
    num_givens_rotations: int = 0
    num_exp_pauli_calls: int = 0
    num_phase_rotations: int = 0
    two_qubit_gate_count_proxy: int = 0
    depth_proxy: int = 0


@dataclass
class SlaterDeterminantPlan:
    num_orbitals: int = 0
    num_electrons: int = 0
    is_complex: bool = False
    orbital_indices: list = field(default_factory=list)
    angles: list = field(default_factory=list)
    phases: list = field(default_factory=list)
    final_phases: list = field(default_factory=list)

    # ------------------------------------------------------------------
    # Prototype ergonomics
    # ------------------------------------------------------------------

    def kernel(self):
        """A ready ``@cudaq.kernel()`` preparing this plan's determinant.

        Allocates ``num_orbitals`` qubits and applies the real or complex
        preparation as appropriate; no argument threading needed.
        """
        validate_slater_determinant_plan(self)
        num_orbitals = int(self.num_orbitals)
        num_electrons = int(self.num_electrons)

        if not self.angles:
            # Computational-basis determinant: no rotations. Special-cased
            # because captured empty lists cannot be marshaled.
            @cudaq.kernel
            def prepare_basis_determinant():
                qubits = cudaq.qvector(num_orbitals)
                for i in range(num_electrons):
                    x(qubits[i])

            return prepare_basis_determinant

        indices = [int(i) for i in self.orbital_indices]
        angles = [float(a) for a in self.angles]

        if self.is_complex:
            phases = [float(p) for p in self.phases]
            final_phases = [float(p) for p in self.final_phases]

            @cudaq.kernel
            def prepare_complex():
                qubits = cudaq.qvector(num_orbitals)
                prepare_complex_slater_determinant(qubits, indices, angles,
                                                   phases, final_phases,
                                                   num_electrons)

            return prepare_complex

        @cudaq.kernel
        def prepare_real():
            qubits = cudaq.qvector(num_orbitals)
            prepare_slater_determinant(qubits, indices, angles, num_electrons)

        return prepare_real

    def state(self):
        """Simulate the preparation and return the statevector (numpy)."""
        import numpy as np

        return np.asarray(cudaq.get_state(self.kernel()),
                          dtype=np.complex128)

    def resources(self) -> GivensStatePrepResources:
        return estimate_givens_stateprep_resources(self)


# ============================================================================
# Schedule construction
# ============================================================================


def _as_matrix(occupied_orbitals):
    """Normalize input to a list of rows; detect complex entries."""
    if hasattr(occupied_orbitals, "tolist"):
        is_complex = getattr(getattr(occupied_orbitals, "dtype", None), "kind",
                             None) == "c"
        rows = occupied_orbitals.tolist()
    else:
        is_complex = False
        rows = [list(row) for row in occupied_orbitals]
    if not is_complex:
        is_complex = any(
            isinstance(value, complex) for row in rows for value in row)
    return rows, is_complex


def _validate_occupied_orbitals(rows, tolerance, is_complex):
    if not rows:
        raise ValueError("occupied_orbitals must not be empty")

    num_electrons = len(rows[0])
    if num_electrons == 0:
        raise ValueError(
            "occupied_orbitals must contain at least one occupied orbital")
    if num_electrons > len(rows):
        raise ValueError(
            "number of occupied orbitals cannot exceed number of spin "
            "orbitals")
    for row in rows:
        if len(row) != num_electrons:
            raise ValueError("occupied_orbitals must be a rectangular matrix")

    for col in range(num_electrons):
        norm = sum(abs(row[col])**2 for row in rows)
        if abs(norm - 1.0) > 100.0 * tolerance:
            raise ValueError("occupied_orbitals columns must be normalized")
        for other in range(col + 1, num_electrons):
            if is_complex:
                overlap = sum(
                    complex(row[col]).conjugate() * row[other]
                    for row in rows)
            else:
                overlap = sum(row[col] * row[other] for row in rows)
            if abs(overlap) > 100.0 * tolerance:
                raise ValueError("occupied_orbitals columns must be "
                                 "orthogonal")


def _argument_or_zero(value, tolerance):
    if abs(value) <= tolerance:
        return 0.0
    return cmath.phase(value)


def make_givens_rotation_schedule(occupied_orbitals,
                                  tolerance=1.0e-12) -> GivensRotationSchedule:
    """Build the Givens rotation schedule preparing a Slater determinant.

    ``occupied_orbitals`` is a (num_orbitals x num_electrons) matrix (numpy
    array or nested lists) whose orthonormal columns are the occupied
    orbitals. Real and complex inputs dispatch automatically, exactly like
    the upstream Python wrapper (a complex dtype routes complex even when
    all values are real).
    """
    rows, is_complex = _as_matrix(occupied_orbitals)
    _validate_occupied_orbitals(rows, tolerance, is_complex)

    num_orbitals = len(rows)
    num_electrons = len(rows[0])
    work = [[complex(value) for value in row] for row in rows]
    eliminations = []

    for col in range(num_electrons):
        for row in range(num_orbitals - 1, col, -1):
            upper_row = row - 1
            upper = work[upper_row][col]
            lower = work[row][col]

            if abs(lower) <= tolerance:
                continue

            upper_magnitude = abs(upper)
            lower_magnitude = abs(lower)
            radius = math.hypot(upper_magnitude, lower_magnitude)
            if radius <= tolerance:
                raise RuntimeError("failed to construct Givens rotation")

            cosine = upper_magnitude / radius
            sine = lower_magnitude / radius
            theta = math.atan2(sine, cosine)
            if is_complex:
                phase = (_argument_or_zero(lower, tolerance) -
                         _argument_or_zero(upper, tolerance))
            else:
                # The real path works on signed values directly.
                cosine = upper.real / radius
                sine = lower.real / radius
                theta = math.atan2(sine, cosine)
                phase = 0.0

            lower_phase = cmath.exp(-1.0j * phase)
            for k in range(num_electrons):
                upper_value = work[upper_row][k]
                lower_value = lower_phase * work[row][k]
                work[upper_row][k] = cosine * upper_value + sine * lower_value
                work[row][k] = -sine * upper_value + cosine * lower_value

            eliminations.append(
                GivensRotation(upper_row, row, theta, phase))

    schedule = GivensRotationSchedule(num_orbitals=num_orbitals,
                                      num_electrons=num_electrons)
    if is_complex:
        schedule.final_phases = [
            _argument_or_zero(work[col][col], tolerance)
            for col in range(num_electrons)
        ]
    else:
        schedule.final_phases = [0.0] * num_electrons

    # State preparation applies the inverse of the row rotations that reduce
    # the occupied-orbital matrix to the computational-basis determinant.
    schedule.rotations = list(reversed(eliminations))
    schedule._is_complex = is_complex
    return schedule


def get_givens_rotation_indices(schedule: GivensRotationSchedule) -> list:
    indices = []
    for rotation in schedule.rotations:
        indices.append(int(rotation.first_orbital))
        indices.append(int(rotation.second_orbital))
    return indices


def get_givens_rotation_angles(schedule: GivensRotationSchedule) -> list:
    return [float(rotation.theta) for rotation in schedule.rotations]


def get_givens_rotation_phases(schedule: GivensRotationSchedule) -> list:
    return [float(rotation.phase) for rotation in schedule.rotations]


def make_slater_determinant_plan(occupied_orbitals,
                                 tolerance=1.0e-12) -> SlaterDeterminantPlan:
    """Build and validate a flattened plan (real/complex auto-dispatch)."""
    schedule = make_givens_rotation_schedule(occupied_orbitals, tolerance)
    plan = SlaterDeterminantPlan(
        num_orbitals=schedule.num_orbitals,
        num_electrons=schedule.num_electrons,
        is_complex=getattr(schedule, "_is_complex", False),
        orbital_indices=get_givens_rotation_indices(schedule),
        angles=get_givens_rotation_angles(schedule),
        phases=get_givens_rotation_phases(schedule),
        final_phases=list(schedule.final_phases))
    validate_slater_determinant_plan(plan)
    return plan


def validate_slater_determinant_plan(plan: SlaterDeterminantPlan):
    if plan.num_orbitals == 0:
        raise ValueError("num_orbitals must be greater than zero")
    if plan.num_electrons == 0:
        raise ValueError("num_electrons must be greater than zero")
    if plan.num_electrons > plan.num_orbitals:
        raise ValueError("num_electrons cannot exceed num_orbitals")
    if len(plan.orbital_indices) != 2 * len(plan.angles):
        raise ValueError(
            "orbital_indices must contain two entries for each angle")

    for i in range(len(plan.angles)):
        first = plan.orbital_indices[2 * i]
        second = plan.orbital_indices[2 * i + 1]
        if first >= plan.num_orbitals or second >= plan.num_orbitals:
            raise ValueError("Givens rotation orbital index is out of range")
        if abs(first - second) != 1:
            raise ValueError(
                "Givens state-preparation kernels require adjacent rotations")

    if plan.is_complex:
        if len(plan.phases) != len(plan.angles):
            raise ValueError(
                "complex Slater determinant plans require one phase per angle")
        if len(plan.final_phases) != plan.num_electrons:
            raise ValueError(
                "complex Slater determinant plans require one final phase per "
                "electron")
        return

    if plan.phases and len(plan.phases) != len(plan.angles):
        raise ValueError(
            "real Slater determinant plan phases must be empty or match "
            "angles")
    if plan.final_phases and len(plan.final_phases) != plan.num_electrons:
        raise ValueError(
            "real Slater determinant plan final phases must be empty or "
            "match num_electrons")


def _resource_estimate(num_rotations, num_electrons,
                       is_complex) -> GivensStatePrepResources:
    resources = GivensStatePrepResources()
    resources.num_givens_rotations = num_rotations
    resources.num_exp_pauli_calls = 2 * num_rotations
    resources.num_phase_rotations = (num_rotations +
                                     num_electrons if is_complex else 0)
    resources.two_qubit_gate_count_proxy = resources.num_exp_pauli_calls
    resources.depth_proxy = (resources.num_exp_pauli_calls +
                             resources.num_phase_rotations)
    return resources


def estimate_givens_rotation_schedule_resources(
        schedule: GivensRotationSchedule,
        is_complex: bool = False) -> GivensStatePrepResources:
    return _resource_estimate(len(schedule.rotations),
                              schedule.num_electrons, is_complex)


def estimate_givens_stateprep_resources(plan_or_schedule,
                                        is_complex: bool = False
                                        ) -> GivensStatePrepResources:
    if isinstance(plan_or_schedule, SlaterDeterminantPlan):
        plan = plan_or_schedule
        validate_slater_determinant_plan(plan)
        return _resource_estimate(len(plan.angles), plan.num_electrons,
                                  plan.is_complex)
    return estimate_givens_rotation_schedule_resources(
        plan_or_schedule, is_complex)


__all__ = [
    "GivensRotation",
    "GivensRotationSchedule",
    "GivensStatePrepResources",
    "SlaterDeterminantPlan",
    "apply_givens_rotation",
    "apply_phase_givens_rotation",
    "estimate_givens_rotation_schedule_resources",
    "estimate_givens_stateprep_resources",
    "get_givens_rotation_angles",
    "get_givens_rotation_indices",
    "get_givens_rotation_phases",
    "make_givens_rotation_schedule",
    "make_slater_determinant_plan",
    "prepare_complex_slater_determinant",
    "prepare_slater_determinant",
    "state_from",
    "validate_slater_determinant_plan",
]
