# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""Pure-Python Suzuki-Trotter Hamiltonian simulation — API-surface experiment.

A self-contained replication of the `add_suzuki_trotter` branch implemented
entirely in Python: term extraction, planning/ordering, resource estimation,
and the product-formula circuit itself (`exp_pauli` inside a Python
`@cudaq.kernel`) — no compiled cudaq-algorithms bindings.

Functionality parity with the C++/bindings branch:

* ``apply_trotter`` device kernel — identical signature and semantics,
  orders 1, 2 (symmetric), and 4 (Forest-Ruth), invalid runtime inputs are
  no-ops (zero steps, length mismatch, unsupported order).
* ``make_trotter_terms`` / ``make_trotter_plan`` / ``TrotterPlan`` /
  ``TrotterOrdering`` / ``estimate_trotter_resources`` — same fields, same
  validation, same resource formulas.

Prototype-style ergonomics on top (same philosophy as the LCU experiment):

* ``PauliLCU``-style flexible Hamiltonian input: a ``cudaq.SpinOperator``,
  a single spin term, a ``{"XZI...": coeff}`` mapping, or
  ``[(coeff, word), ...]`` pairs.
* ``plan.kernel()`` — a ready ``@cudaq.kernel(state)``, no argument
  threading.
* ``plan.evolve(ket)`` — one-call simulation that can INCLUDE the identity
  phase ``exp(-i c t)``, which the circuit primitive necessarily omits.
* ``plan.resources()`` — the estimate as a method.

Identity terms: for ``H = c I + H'``, ``apply_trotter`` implements the
product formula for ``H'`` only. The omitted ``exp(-i c t)`` is an
unobservable global phase for one unconditioned evolution but a real
relative phase for controlled/interference-based algorithms;
``identity_coefficient`` is reported so callers can account for it, and
``plan.evolve`` reintroduces it host-side by default.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from enum import Enum

import cudaq

FIRST_ORDER_TROTTER = 1
SECOND_ORDER_TROTTER = 2
FOURTH_ORDER_TROTTER = 4

FOREST_RUTH_W1 = 1.3512071919596578
FOREST_RUTH_W0 = -1.7024143839193153


# ============================================================================
# Device kernel (module level, composable from user kernels)
# ============================================================================


@cudaq.kernel
def apply_trotter(coefficients: list[float], words: list[cudaq.pauli_word],
                  time: float, steps: int, order: int, qubits: cudaq.qview):
    """Apply Suzuki-Trotter evolution exp(-i H' t) to a live register.

    Identity terms are intentionally omitted (see the module docstring).
    Invalid runtime inputs are no-ops: zero steps, mismatched
    coefficient/word lengths, or an unsupported order leave the register
    unchanged — matching the C++ device kernel, which cannot throw.

    NOTE: the whole body is one guarded if-block rather than early-return
    guards because `return` inside a Python @cudaq.kernel is silently
    ignored by the AST bridge (gates after it still execute).
    """
    valid_order = order == 1 or order == 2 or order == 4
    if steps > 0 and len(coefficients) == len(words) and valid_order:
        n = len(words)
        dt = time / steps
        for _ in range(steps):
            if order == 1:
                for i in range(n):
                    exp_pauli(-dt * coefficients[i], qubits, words[i])
            elif order == 4:
                # Forest-Ruth: symmetric second-order steps with weights
                # w1, w0, w1 (w1 = 1.3512..., w0 = -1.7024...).
                for i in range(n):
                    exp_pauli(-0.5 * 1.3512071919596578 * dt * coefficients[i],
                              qubits, words[i])
                for j in range(n):
                    i = n - 1 - j
                    exp_pauli(-0.5 * 1.3512071919596578 * dt * coefficients[i],
                              qubits, words[i])
                for i in range(n):
                    exp_pauli(-0.5 * -1.7024143839193153 * dt * coefficients[i],
                              qubits, words[i])
                for j in range(n):
                    i = n - 1 - j
                    exp_pauli(-0.5 * -1.7024143839193153 * dt * coefficients[i],
                              qubits, words[i])
                for i in range(n):
                    exp_pauli(-0.5 * 1.3512071919596578 * dt * coefficients[i],
                              qubits, words[i])
                for j in range(n):
                    i = n - 1 - j
                    exp_pauli(-0.5 * 1.3512071919596578 * dt * coefficients[i],
                              qubits, words[i])
            else:
                for i in range(n):
                    exp_pauli(-0.5 * dt * coefficients[i], qubits, words[i])
                for j in range(n):
                    i = n - 1 - j
                    exp_pauli(-0.5 * dt * coefficients[i], qubits, words[i])


def state_from(ket):
    """Build a cudaq.State at the current target's precision.

    fp32 simulators reject complex128 initial-state data; cudaq.complex()
    reports the dtype the active target expects.
    """
    import numpy as np

    return cudaq.State.from_data(np.asarray(ket, dtype=cudaq.complex()))


# ============================================================================
# Host-side term extraction
# ============================================================================


def _maybe_call(value):
    return value() if callable(value) else value


def _is_spin_like(value):
    return (hasattr(value, "evaluate_coefficient")
            or hasattr(value, "term_count")
            or hasattr(value, "get_term_count"))


def _term_qubit_extent(term):
    max_degree = _maybe_call(getattr(term, "max_degree", -1))
    return max_degree + 1 if max_degree >= 0 else 0


def _real_coefficient(coefficient, tolerance):
    coefficient = complex(coefficient)
    if abs(coefficient.imag) > tolerance:
        raise ValueError(
            "trotter error - only real Hamiltonian coefficients are "
            "supported.")
    return float(coefficient.real)


def _word_pairs_from_input(hamiltonian, tolerance):
    """Normalize dict / pairs / spin-op / spin-term inputs to (coeff, str)."""
    if isinstance(hamiltonian, Mapping):
        pairs = [(_real_coefficient(c, tolerance), str(w))
                 for w, c in hamiltonian.items()]
        width = max((len(w) for _, w in pairs), default=0)
    elif _is_spin_like(hamiltonian):
        # Wrapping canonicalizes: elementary products (SpinOperatorTerm) and
        # full operators both become a SpinOperator whose iteration yields
        # terms with evaluate_coefficient() and get_pauli_word().
        terms = list(cudaq.SpinOperator(hamiltonian))
        # Fix the register width once so every word is padded to the same
        # length (mirrors the C++ path, which uses num_qubits()).
        width = 0
        for term in terms:
            width = max(width, _term_qubit_extent(term))
        pairs = [(_real_coefficient(term.evaluate_coefficient(), tolerance),
                  str(term.get_pauli_word(width))) for term in terms]
    elif isinstance(hamiltonian, Iterable):
        pairs = [(_real_coefficient(c, tolerance), str(w))
                 for c, w in hamiltonian]
        width = max((len(w) for _, w in pairs), default=0)
    else:
        raise TypeError(
            "hamiltonian must be a cudaq spin operator or term, a "
            "{word: coeff} mapping, or an iterable of (coeff, word) pairs")

    for _, word in pairs:
        if len(word) != width:
            raise ValueError("all Pauli words must have the same length")
        if any(ch not in "IXYZ" for ch in word):
            raise ValueError(f"unsupported Pauli word: {word!r}")
    return pairs, width


def make_trotter_terms(hamiltonian, coefficient_tolerance=1e-12):
    """Return flattened terms for the Suzuki-Trotter circuit primitive.

    Returns ``(coefficients, words, identity_coefficient, num_qubits)``
    where ``words`` are padded plain strings: readable, comparable, and
    accepted directly as ``list[cudaq.pauli_word]`` kernel arguments.
    (Only factory-captured words need explicit ``cudaq.pauli_word``
    conversion, which ``TrotterPlan.kernel`` does internally — captured
    plain strings cannot be lowered.)
    """
    if coefficient_tolerance < 0.0:
        raise ValueError(
            "trotter error - coefficient tolerance must be non-negative.")

    pairs, num_qubits = _word_pairs_from_input(hamiltonian,
                                               coefficient_tolerance)

    coefficients = []
    words = []
    identity_coefficient = 0.0
    for coefficient, word in pairs:
        if set(word) <= {"I"}:
            identity_coefficient += coefficient
            continue
        coefficients.append(coefficient)
        words.append(word)

    return coefficients, words, identity_coefficient, num_qubits


# ============================================================================
# Planning, ordering, resources
# ============================================================================


class TrotterOrdering(Enum):
    PRESERVE_INPUT = "preserve_input"
    COEFFICIENT_MAGNITUDE_DESCENDING = "coefficient_magnitude_descending"


def _validate_order(order):
    if order not in (FIRST_ORDER_TROTTER, SECOND_ORDER_TROTTER,
                     FOURTH_ORDER_TROTTER):
        raise ValueError("order must be one of {1, 2, 4}")
    return int(order)


def _validate_steps(steps):
    steps = int(steps)
    if steps < 1:
        raise ValueError("steps must be greater than zero")
    return steps


def _validate_time(time):
    time = float(time)
    if not math.isfinite(time):
        raise ValueError("time must be a finite number")
    return time


def _coerce_ordering(ordering):
    if isinstance(ordering, TrotterOrdering):
        return ordering
    try:
        return TrotterOrdering(str(ordering))
    except ValueError as exc:
        raise ValueError(f"unsupported Trotter ordering: {ordering}") from exc


def _ordered_terms(coefficients, words, ordering):
    coefficients = list(coefficients)
    words = list(words)
    if ordering == TrotterOrdering.PRESERVE_INPUT:
        return coefficients, words
    ordered = sorted(zip(coefficients, words),
                     key=lambda item: abs(item[0]),
                     reverse=True)
    if not ordered:
        return [], []
    ordered_coefficients, ordered_words = zip(*ordered)
    return list(ordered_coefficients), list(ordered_words)


def _pauli_weight(word):
    return sum(1 for op in str(word) if op != "I")


def _rotations_per_step(order):
    return {1: 1, 2: 2, 4: 6}[_validate_order(order)]


@dataclass(frozen=True)
class TrotterResourceEstimate:
    num_terms: int
    steps: int
    order: int
    pauli_rotations: int
    estimated_cx_count: int
    identity_coefficient: float


@dataclass(frozen=True)
class TrotterPlan:
    """Host-side plan for Suzuki-Trotter evolution.

    Carries the flattened data ``apply_trotter`` consumes, plus factory and
    simulation conveniences so no caller has to thread the arrays by hand.
    """

    coefficients: list[float]
    words: list = field(repr=False)
    identity_coefficient: float
    num_qubits: int
    time: float
    steps: int
    order: int
    ordering: TrotterOrdering

    @property
    def num_terms(self) -> int:
        return len(self.coefficients)

    def kernel(self):
        """A ``@cudaq.kernel(state)`` applying this plan's evolution.

        The kernel allocates the register from ``state`` and applies the
        product formula; the identity phase is NOT included (it cannot be,
        in a circuit) — use ``evolve`` when it matters.
        """
        coefficients = [float(c) for c in self.coefficients]
        words = [cudaq.pauli_word(str(w)) for w in self.words]
        time = float(self.time)
        steps = int(self.steps)
        order = int(self.order)

        if not words:
            # Identity-only Hamiltonian: the circuit is the identity. This is
            # special-cased because captured empty lists cannot be marshaled
            # across the kernel boundary.
            @cudaq.kernel
            def evolve_identity(state: cudaq.State):
                cudaq.qvector(state)

            return evolve_identity

        @cudaq.kernel
        def evolve(state: cudaq.State):
            qubits = cudaq.qvector(state)
            apply_trotter(coefficients, words, time, steps, order, qubits)

        return evolve

    def evolve(self, ket, include_identity_phase: bool = True):
        """Simulate this plan on ``ket`` and return the evolved statevector.

        Unlike the circuit primitive, this can reintroduce the identity
        phase exp(-i * identity_coefficient * time) (on by default), so the
        result approximates the full exp(-i H t)|ket>.
        """
        import numpy as np

        state = np.asarray(cudaq.get_state(self.kernel(), state_from(ket)),
                           dtype=np.complex128)
        if include_identity_phase and self.identity_coefficient != 0.0:
            state = state * np.exp(
                -1.0j * self.identity_coefficient * self.time)
        return state

    def resources(self) -> TrotterResourceEstimate:
        return estimate_trotter_resources(self)


def make_trotter_plan(hamiltonian,
                      time,
                      steps=1,
                      order=SECOND_ORDER_TROTTER,
                      ordering=TrotterOrdering.PRESERVE_INPUT,
                      coefficient_tolerance=1e-12) -> TrotterPlan:
    """Build a validated host-side plan for Suzuki-Trotter evolution."""
    time = _validate_time(time)
    steps = _validate_steps(steps)
    order = _validate_order(order)
    ordering = _coerce_ordering(ordering)
    coefficients, words, identity, num_qubits = make_trotter_terms(
        hamiltonian, coefficient_tolerance)
    coefficients, words = _ordered_terms(coefficients, words, ordering)
    return TrotterPlan(coefficients=coefficients,
                       words=words,
                       identity_coefficient=identity,
                       num_qubits=num_qubits,
                       time=time,
                       steps=steps,
                       order=order,
                       ordering=ordering)


def estimate_trotter_resources(plan_or_coefficients,
                               words=None,
                               steps=None,
                               order=None,
                               identity_coefficient=0.0
                               ) -> TrotterResourceEstimate:
    """Return a lightweight resource estimate for a Trotter sequence.

    The CNOT count is a decomposition proxy: two CNOTs per additional
    non-identity Pauli in each Pauli rotation.
    """
    if isinstance(plan_or_coefficients, TrotterPlan):
        coefficients = plan_or_coefficients.coefficients
        words = plan_or_coefficients.words
        steps = plan_or_coefficients.steps
        order = plan_or_coefficients.order
        identity_coefficient = plan_or_coefficients.identity_coefficient
    else:
        coefficients = list(plan_or_coefficients)
        words = list(words)
        steps = _validate_steps(steps)
        order = _validate_order(order)

    if len(coefficients) != len(words):
        raise ValueError("coefficients and words must have equal length")

    rotations = len(words) * steps * _rotations_per_step(order)
    cx_per_ordered_step = sum(
        max(0, 2 * (_pauli_weight(word) - 1)) for word in words)
    estimated_cx_count = cx_per_ordered_step * steps * _rotations_per_step(
        order)
    return TrotterResourceEstimate(
        num_terms=len(words),
        steps=steps,
        order=order,
        pauli_rotations=rotations,
        estimated_cx_count=estimated_cx_count,
        identity_coefficient=float(identity_coefficient))


__all__ = [
    "FIRST_ORDER_TROTTER",
    "SECOND_ORDER_TROTTER",
    "FOURTH_ORDER_TROTTER",
    "FOREST_RUTH_W1",
    "FOREST_RUTH_W0",
    "TrotterOrdering",
    "TrotterPlan",
    "TrotterResourceEstimate",
    "apply_trotter",
    "make_trotter_terms",
    "make_trotter_plan",
    "estimate_trotter_resources",
    "state_from",
]
