# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                        #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""Pure-Python Hartree-Fock and fixed-parameter UCC state preparation.

A self-contained replication of the `add_hf_fixed_param_ucc_state_prep`
branch implemented entirely in Python — no compiled cudaq-algorithms
bindings. Because the fixed-parameter UCC plan constructors depend on the
library's operator-pool generators, the needed subset of those is ported
too: `get_uccsd_excitations`, `make_uccsd_operator_pool`,
`make_uccgsd_operator_pool` / `get_uccgsd_pauli_lists`, and
`make_upccgsd_operator_pool` / `get_upccgsd_pauli_lists` (built with
`cudaq.spin` algebra, extracted through `SpinOperatorTerm.get_pauli_word`).

Functionality parity with the C++/bindings branch:

* ``hartree_fock`` / ``hartree_fock_occupation`` kernels; closed-shell and
  open-shell (interleaved alpha/beta) occupation builders; occupation
  validation; both resource estimators.
* ``fixed_parameter_ucc`` kernel with the upstream nested-list signature
  (one parameter per excitation group of Pauli words and coefficients).
* ``FixedParameterUccPlan`` + validation, the generic plan constructors
  (operator pool or word/coefficient lists), the uccsd/uccgsd/upccgsd
  convenience constructors, and the resource estimator.

Prototype-style ergonomics on top: ``plan.kernel(num_electrons=...)``
returns a ready ``@cudaq.kernel()`` preparing Hartree-Fock plus the UCC
product, and ``plan.state(...)`` simulates it. (The factory flattens the
grouped data internally: nested lists are accepted as kernel *arguments*
but cannot be *captured* — another AST-bridge finding.)
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cudaq
from cudaq import spin

# ============================================================================
# Device kernels (module level, composable from user kernels)
# ============================================================================


@cudaq.kernel
def hartree_fock(qubits: cudaq.qview, num_electrons: int):
    """Fill the first num_electrons spin orbitals (closed-shell reference).

    For open-shell systems build the occupation with
    make_hartree_fock_occupation(num_qubits, num_electrons, spin) and use
    hartree_fock_occupation instead.
    """
    for i in range(num_electrons):
        x(qubits[i])


@cudaq.kernel
def hartree_fock_occupation(qubits: cudaq.qview,
                            occupied_orbitals: list[int]):
    """Prepare a determinant from explicit occupied spin-orbital indices."""
    for i in range(len(occupied_orbitals)):
        x(qubits[occupied_orbitals[i]])


@cudaq.kernel
def fixed_parameter_ucc(qubits: cudaq.qview, parameters: list[float],
                        pauli_words: list[list[cudaq.pauli_word]],
                        coefficients: list[list[float]]):
    """Apply a fixed-parameter UCC-style product over grouped Pauli terms.

    One parameter per excitation group; each group's words/coefficients are
    generated on the host. The qubits must already hold a Hartree-Fock
    reference (prepare it with hartree_fock / hartree_fock_occupation
    first) — applied to |0...0> this yields a physically meaningless state.
    """
    for i in range(len(pauli_words)):
        theta = parameters[i]
        words = pauli_words[i]
        coeffs = coefficients[i]
        for j in range(len(words)):
            exp_pauli(theta * coeffs[j], qubits, words[j])


@cudaq.kernel
def _fixed_parameter_ucc_flat(qubits: cudaq.qview, angles: list[float],
                              words: list[cudaq.pauli_word]):
    """Flattened variant used by the plan factory (nested captures are
    unsupported); angles[j] = theta_group(j) * coefficient_j."""
    for j in range(len(words)):
        exp_pauli(angles[j], qubits, words[j])


def state_from(ket):
    """Build a cudaq.State at the current target's precision."""
    import numpy as np

    return cudaq.State.from_data(np.asarray(ket, dtype=cudaq.complex()))


# ============================================================================
# Hartree-Fock host helpers
# ============================================================================


@dataclass
class HartreeFockResources:
    num_qubits: int = 0
    num_electrons: int = 0
    num_x_gates: int = 0


def make_hartree_fock_occupation(num_qubits, num_electrons, spin_number=0):
    """Occupied spin-orbital indices of the Hartree-Fock reference.

    spin_number == 0 (closed shell): the contiguous set {0..num_electrons-1}.
    spin_number > 0 (open shell): alpha electrons on even spin orbitals and
    beta on odd, matching get_uccsd_excitations so the determinant lines up
    with a fixed-parameter UCCSD plan built at the same spin.
    """
    num_qubits = int(num_qubits)
    num_electrons = int(num_electrons)
    spin_number = int(spin_number)
    if num_electrons > num_qubits:
        raise ValueError(
            "hartree_fock error - num_electrons cannot exceed num_qubits.")

    if spin_number == 0:
        return list(range(num_electrons))

    if num_qubits % 2 != 0:
        raise ValueError(
            "hartree_fock error - num_qubits must be even for spin > 0.")
    if spin_number > num_electrons:
        raise ValueError(
            "hartree_fock error - spin cannot exceed num_electrons.")

    num_spatial_orbitals = num_qubits // 2
    num_occupied_beta = (num_electrons - spin_number) // 2
    num_occupied_alpha = num_electrons - num_occupied_beta
    if (num_occupied_alpha > num_spatial_orbitals
            or num_occupied_beta > num_spatial_orbitals):
        raise ValueError(
            "hartree_fock error - the requested (num_electrons, spin) does "
            "not fit in num_qubits spin orbitals.")

    occupied = [2 * i for i in range(num_occupied_alpha)]
    occupied += [2 * i + 1 for i in range(num_occupied_beta)]
    return sorted(occupied)


def validate_hartree_fock_occupation(num_qubits, occupied_orbitals):
    seen = set()
    for orbital in occupied_orbitals:
        if orbital >= num_qubits:
            raise ValueError("hartree_fock error - occupied orbital index "
                             "exceeds num_qubits.")
        if orbital in seen:
            raise ValueError("hartree_fock error - occupied orbital indices "
                             "must be unique.")
        seen.add(orbital)


def estimate_hartree_fock_resources(num_qubits,
                                    num_electrons) -> HartreeFockResources:
    occupation = make_hartree_fock_occupation(num_qubits, num_electrons)
    return estimate_hartree_fock_occupation_resources(num_qubits, occupation)


def estimate_hartree_fock_occupation_resources(
        num_qubits, occupied_orbitals) -> HartreeFockResources:
    validate_hartree_fock_occupation(num_qubits, occupied_orbitals)
    return HartreeFockResources(num_qubits=int(num_qubits),
                                num_electrons=len(occupied_orbitals),
                                num_x_gates=len(occupied_orbitals))


# ============================================================================
# Operator pools (the subset the fixed-parameter constructors need)
# ============================================================================


def get_uccsd_excitations(num_qubits, num_electrons, spin_number=0):
    """Enumerate UCCSD excitations (ported from the C++ implementation).

    Returns (singles_alpha, singles_beta, doubles_mixed, doubles_alpha,
    doubles_beta) as lists of index lists.
    """
    num_qubits = int(num_qubits)
    num_electrons = int(num_electrons)
    spin_number = int(spin_number)
    if num_qubits % 2 != 0:
        raise RuntimeError("The total number of qubits should be even.")

    num_spatial = num_qubits // 2
    if spin_number > 0:
        n_occ_beta = (num_electrons - spin_number) // 2
        n_occ_alpha = num_electrons - n_occ_beta
        n_virt_alpha = num_spatial - n_occ_alpha
        n_virt_beta = num_spatial - n_occ_beta
        occupied_alpha = [i * 2 for i in range(n_occ_alpha)]
        virtual_alpha = [i * 2 + 2 * n_occ_alpha for i in range(n_virt_alpha)]
        occupied_beta = [i * 2 + 1 for i in range(n_occ_beta)]
        virtual_beta = [
            i * 2 + 2 * n_occ_beta + 1 for i in range(n_virt_beta)
        ]
    elif num_electrons % 2 == 0 and spin_number == 0:
        n_occ = num_electrons // 2
        n_virt = num_spatial - n_occ
        occupied_alpha = [i * 2 for i in range(n_occ)]
        virtual_alpha = [i * 2 + num_electrons for i in range(n_virt)]
        occupied_beta = [i * 2 + 1 for i in range(n_occ)]
        virtual_beta = [i * 2 + num_electrons + 1 for i in range(n_virt)]
    else:
        raise RuntimeError(
            "Incorrect spin multiplicity. Number of electrons is odd but "
            f"spin is 0 {num_electrons}, {spin_number}")

    singles_alpha = [[p, q] for p in occupied_alpha for q in virtual_alpha]
    singles_beta = [[p, q] for p in occupied_beta for q in virtual_beta]
    doubles_mixed = [[p, q, r, s] for p in occupied_alpha
                     for q in occupied_beta for r in virtual_beta
                     for s in virtual_alpha]

    doubles_alpha = []
    for p in range(len(occupied_alpha) - 1):
        for q in range(p + 1, len(occupied_alpha)):
            for r in range(len(virtual_alpha) - 1):
                for s in range(r + 1, len(virtual_alpha)):
                    doubles_alpha.append([
                        occupied_alpha[p], occupied_alpha[q],
                        virtual_alpha[r], virtual_alpha[s]
                    ])
    doubles_beta = []
    for p in range(len(occupied_beta) - 1):
        for q in range(p + 1, len(occupied_beta)):
            for r in range(len(virtual_beta) - 1):
                for s in range(r + 1, len(virtual_beta)):
                    doubles_beta.append([
                        occupied_beta[p], occupied_beta[q], virtual_beta[r],
                        virtual_beta[s]
                    ])

    return (singles_alpha, singles_beta, doubles_mixed, doubles_alpha,
            doubles_beta)


def _z_parity(low, high):
    """Product of Z operators strictly between low and high (may be None)."""
    parity = None
    for i in range(low + 1, high):
        factor = spin.z(i)
        parity = factor if parity is None else parity * factor
    return parity


def _times(*factors):
    result = None
    for factor in factors:
        if factor is None:
            continue
        result = factor if result is None else result * factor
    return result


def _uccsd_single(p, q):
    """0.5 * (Y_p Z... X_q - X_p Z... Y_q) with p < q parity string."""
    parity = _z_parity(p, q)
    return 0.5 * _times(spin.y(p), parity, spin.x(q)) - 0.5 * _times(
        spin.x(p), parity, spin.y(q))


def _uccsd_double(p, q, r, s):
    """The 8-term UCCSD double-excitation generator (C++ index conventions)."""
    if p < q and r < s:
        i_occ, j_occ, a_virt, b_virt = p, q, r, s
    elif p > q and r > s:
        i_occ, j_occ, a_virt, b_virt = q, p, s, r
    elif p < q and r > s:
        i_occ, j_occ, a_virt, b_virt = p, q, s, r
    else:
        i_occ, j_occ, a_virt, b_virt = q, p, r, s

    parity_a = _z_parity(i_occ, j_occ)
    parity_b = _z_parity(a_virt, b_virt)

    def term(op_i, op_j, op_a, op_b):
        return _times(op_i(i_occ), parity_a, op_j(j_occ), op_a(a_virt),
                      parity_b, op_b(b_virt))

    op = term(spin.x, spin.x, spin.x, spin.y)
    op += term(spin.x, spin.x, spin.y, spin.x)
    op += term(spin.x, spin.y, spin.y, spin.y)
    op += term(spin.y, spin.x, spin.y, spin.y)
    op -= term(spin.x, spin.y, spin.x, spin.x)
    op -= term(spin.y, spin.x, spin.x, spin.x)
    op -= term(spin.y, spin.y, spin.x, spin.y)
    op -= term(spin.y, spin.y, spin.y, spin.x)
    return 0.125 * op


def make_uccsd_operator_pool(num_qubits, num_electrons, spin_number=0):
    (singles_alpha, singles_beta, doubles_mixed, doubles_alpha,
     doubles_beta) = get_uccsd_excitations(num_qubits, num_electrons,
                                           spin_number)
    ops = []
    for p, q in singles_alpha:
        ops.append(_uccsd_single(p, q))
    for p, q in singles_beta:
        ops.append(_uccsd_single(p, q))
    for p, q, r, s in doubles_mixed:
        ops.append(_uccsd_double(p, q, r, s))
    for p, q, r, s in doubles_alpha:
        ops.append(_uccsd_double(p, q, r, s))
    for p, q, r, s in doubles_beta:
        ops.append(_uccsd_double(p, q, r, s))
    return ops


def _uccgsd_single(p, q):
    """0.5 * (Y_q Z... X_p - X_q Z... Y_p) for p > q."""
    parity = _z_parity(q, p)
    return 0.5 * _times(spin.y(q), parity, spin.x(p)) - 0.5 * _times(
        spin.x(q), parity, spin.y(p))


def _uccgsd_double(p, q, r, s):
    """The 8-term generalized double for p > q, r > s."""
    parity_a = _z_parity(q, p)
    parity_b = _z_parity(s, r)

    def term(op_s, op_r, op_q, op_p):
        return _times(op_s(s), parity_b, op_r(r), op_q(q), parity_a, op_p(p))

    op = term(spin.y, spin.x, spin.x, spin.x)
    op += term(spin.x, spin.y, spin.x, spin.x)
    op += term(spin.y, spin.y, spin.y, spin.x)
    op += term(spin.y, spin.y, spin.x, spin.y)
    op -= term(spin.x, spin.x, spin.y, spin.x)
    op -= term(spin.x, spin.x, spin.x, spin.y)
    op -= term(spin.x, spin.y, spin.y, spin.y)
    op -= term(spin.y, spin.x, spin.y, spin.y)
    return 0.125 * op


def _generate_uccgsd_singles(num_qubits):
    return [(p, q) for p in range(1, num_qubits) for q in range(p)]


def _generate_uccgsd_doubles(num_qubits):
    doubles = set()
    for a in range(num_qubits):
        for b in range(a + 1, num_qubits):
            for c in range(b + 1, num_qubits):
                for d in range(c + 1, num_qubits):
                    for pairing in (((a, b), (c, d)), ((a, c), (b, d)),
                                    ((a, d), (b, c))):
                        p1, p2 = pairing
                        p1 = (max(p1), min(p1))
                        p2 = (max(p2), min(p2))
                        doubles.add((min(p1, p2), max(p1, p2)))
    return sorted(doubles)


def make_uccgsd_operator_pool(num_qubits, only_singles=False,
                              only_doubles=False):
    ops = []
    if not only_doubles:
        for p, q in _generate_uccgsd_singles(num_qubits):
            ops.append(_uccgsd_single(p, q))
    if not only_singles:
        for (pq, rs) in _generate_uccgsd_doubles(num_qubits):
            ops.append(_uccgsd_double(pq[0], pq[1], rs[0], rs[1]))
    return ops


def make_upccgsd_operator_pool(num_spin_orbitals, only_doubles=False):
    if num_spin_orbitals % 2 != 0:
        raise ValueError("make_upccgsd_operator_pool expects an even number "
                         "of spin orbitals.")
    ops = []
    if not only_doubles:
        for p, q in _generate_uccgsd_singles(num_spin_orbitals):
            if p % 2 == q % 2:
                ops.append(_uccgsd_single(p, q))
    num_spatial = num_spin_orbitals // 2
    for p in range(num_spatial):
        for q in range(p + 1, num_spatial):
            ops.append(
                _uccgsd_double(2 * q + 1, 2 * q, 2 * p + 1, 2 * p))
    return ops


def _pauli_lists_from_pool(ops, num_qubits):
    words_list = []
    coefficients_list = []
    for op in ops:
        words = []
        coefficients = []
        for term in cudaq.SpinOperator(op):
            words.append(str(term.get_pauli_word(num_qubits)))
            coefficients.append(float(term.evaluate_coefficient().real))
        words_list.append(words)
        coefficients_list.append(coefficients)
    return words_list, coefficients_list


def get_uccgsd_pauli_lists(num_qubits, only_singles=False,
                           only_doubles=False):
    ops = make_uccgsd_operator_pool(num_qubits, only_singles, only_doubles)
    return _pauli_lists_from_pool(ops, num_qubits)


def get_upccgsd_pauli_lists(num_spin_orbitals, only_doubles=False):
    ops = make_upccgsd_operator_pool(num_spin_orbitals, only_doubles)
    return _pauli_lists_from_pool(ops, num_spin_orbitals)


# ============================================================================
# Fixed-parameter UCC plans
# ============================================================================


@dataclass
class FixedParameterUccResources:
    num_qubits: int = 0
    num_excitations: int = 0
    num_pauli_rotations: int = 0
    max_pauli_rotations_per_excitation: int = 0


@dataclass
class FixedParameterUccPlan:
    num_qubits: int = 0
    parameters: list = field(default_factory=list)
    pauli_words: list = field(default_factory=list)  # list of str groups
    coefficients: list = field(default_factory=list)

    def kernel(self, num_electrons=None, occupied_orbitals=None):
        """A ready ``@cudaq.kernel()``: Hartree-Fock reference + UCC product.

        Provide ``num_electrons`` (closed shell) or explicit
        ``occupied_orbitals``. The grouped data is flattened internally
        because nested lists cannot be captured by Python kernels.
        """
        validate_fixed_parameter_ucc_plan(self)
        if (num_electrons is None) == (occupied_orbitals is None):
            raise ValueError(
                "provide exactly one of num_electrons or occupied_orbitals")
        occupation = (list(range(int(num_electrons)))
                      if occupied_orbitals is None else
                      [int(i) for i in occupied_orbitals])
        validate_hartree_fock_occupation(self.num_qubits, occupation)

        num_qubits = int(self.num_qubits)
        flat_angles = []
        flat_words = []
        for theta, words, coeffs in zip(self.parameters, self.pauli_words,
                                        self.coefficients):
            for word, coeff in zip(words, coeffs):
                flat_angles.append(float(theta) * float(coeff))
                flat_words.append(cudaq.pauli_word(str(word)))

        if not flat_words:
            @cudaq.kernel
            def hf_only():
                qubits = cudaq.qvector(num_qubits)
                hartree_fock_occupation(qubits, occupation)

            return hf_only

        @cudaq.kernel
        def hf_then_ucc():
            qubits = cudaq.qvector(num_qubits)
            hartree_fock_occupation(qubits, occupation)
            _fixed_parameter_ucc_flat(qubits, flat_angles, flat_words)

        return hf_then_ucc

    def state(self, num_electrons=None, occupied_orbitals=None):
        """Simulate the preparation and return the statevector (numpy)."""
        import numpy as np

        return np.asarray(cudaq.get_state(
            self.kernel(num_electrons, occupied_orbitals)),
                          dtype=np.complex128)

    def resources(self) -> FixedParameterUccResources:
        return estimate_fixed_parameter_ucc_resources(self)


def validate_fixed_parameter_ucc_plan(plan, coefficient_tolerance=1.0e-12):
    if coefficient_tolerance < 0.0:
        raise ValueError("fixed_parameter_ucc error - coefficient tolerance "
                         "must be non-negative.")
    if (len(plan.parameters) != len(plan.pauli_words)
            or len(plan.pauli_words) != len(plan.coefficients)):
        raise ValueError(
            "fixed_parameter_ucc error - parameters, Pauli-word groups, and "
            "coefficient groups must have the same length.")
    for words, coeffs in zip(plan.pauli_words, plan.coefficients):
        if len(words) != len(coeffs):
            raise ValueError("fixed_parameter_ucc error - each Pauli-word "
                             "group must match its coefficient group.")
        for word in words:
            if len(str(word)) > plan.num_qubits:
                raise ValueError("fixed_parameter_ucc error - Pauli word "
                                 "exceeds plan.num_qubits.")


def make_fixed_parameter_ucc_plan(pool_or_words,
                                  coefficients_or_parameters,
                                  parameters=None,
                                  num_qubits=0,
                                  coefficient_tolerance=1.0e-12
                                  ) -> FixedParameterUccPlan:
    """Build a plan from an operator pool OR from word/coefficient lists.

    Forms (mirroring the two C++ overloads):
      make_fixed_parameter_ucc_plan(operator_pool, parameters, num_qubits=n)
      make_fixed_parameter_ucc_plan(words, coefficients, parameters, n)
    """
    if parameters is None:
        operator_pool = list(pool_or_words)
        pool_parameters = list(coefficients_or_parameters)
        if len(operator_pool) != len(pool_parameters):
            raise ValueError(
                "fixed_parameter_ucc error - operator pool and parameter "
                "vector must have the same length.")
        if coefficient_tolerance < 0.0:
            raise ValueError("fixed_parameter_ucc error - coefficient "
                             "tolerance must be non-negative.")
        words_list = []
        coefficients_list = []
        for op in operator_pool:
            words = []
            coefficients = []
            for term in cudaq.SpinOperator(op):
                coefficient = term.evaluate_coefficient()
                if abs(coefficient.imag) > coefficient_tolerance:
                    raise ValueError(
                        "fixed_parameter_ucc error - only real operator-pool "
                        "coefficients are supported.")
                if abs(coefficient.real) <= coefficient_tolerance:
                    continue
                words.append(str(term.get_pauli_word(num_qubits)))
                coefficients.append(float(coefficient.real))
            words_list.append(words)
            coefficients_list.append(coefficients)
        plan = FixedParameterUccPlan(num_qubits=int(num_qubits),
                                     parameters=pool_parameters,
                                     pauli_words=words_list,
                                     coefficients=coefficients_list)
    else:
        words_list = [[str(w) for w in group] for group in pool_or_words]
        if num_qubits == 0:
            num_qubits = max(
                (len(w) for group in words_list for w in group), default=0)
        plan = FixedParameterUccPlan(
            num_qubits=int(num_qubits),
            parameters=list(parameters),
            pauli_words=words_list,
            coefficients=[list(g) for g in coefficients_or_parameters])

    validate_fixed_parameter_ucc_plan(plan, coefficient_tolerance)
    return plan


def make_fixed_parameter_uccsd_plan(num_qubits, num_electrons, parameters,
                                    spin_number=0,
                                    coefficient_tolerance=1.0e-12):
    pool = make_uccsd_operator_pool(num_qubits, num_electrons, spin_number)
    return make_fixed_parameter_ucc_plan(pool, parameters,
                                         num_qubits=num_qubits,
                                         coefficient_tolerance=
                                         coefficient_tolerance)


def make_fixed_parameter_uccgsd_plan(num_qubits, parameters,
                                     only_singles=False, only_doubles=False,
                                     coefficient_tolerance=1.0e-12):
    words, coefficients = get_uccgsd_pauli_lists(num_qubits, only_singles,
                                                 only_doubles)
    return make_fixed_parameter_ucc_plan(words, coefficients, parameters,
                                         num_qubits, coefficient_tolerance)


def make_fixed_parameter_upccgsd_plan(num_spin_orbitals, parameters,
                                      only_doubles=False,
                                      coefficient_tolerance=1.0e-12):
    words, coefficients = get_upccgsd_pauli_lists(num_spin_orbitals,
                                                  only_doubles)
    return make_fixed_parameter_ucc_plan(words, coefficients, parameters,
                                         num_spin_orbitals,
                                         coefficient_tolerance)


def estimate_fixed_parameter_ucc_resources(
        plan) -> FixedParameterUccResources:
    estimate = FixedParameterUccResources(num_qubits=int(plan.num_qubits),
                                          num_excitations=len(
                                              plan.pauli_words))
    for group in plan.pauli_words:
        estimate.num_pauli_rotations += len(group)
        estimate.max_pauli_rotations_per_excitation = max(
            estimate.max_pauli_rotations_per_excitation, len(group))
    return estimate


__all__ = [
    "FixedParameterUccPlan",
    "FixedParameterUccResources",
    "HartreeFockResources",
    "estimate_fixed_parameter_ucc_resources",
    "estimate_hartree_fock_occupation_resources",
    "estimate_hartree_fock_resources",
    "fixed_parameter_ucc",
    "get_uccgsd_pauli_lists",
    "get_uccsd_excitations",
    "get_upccgsd_pauli_lists",
    "hartree_fock",
    "hartree_fock_occupation",
    "make_fixed_parameter_ucc_plan",
    "make_fixed_parameter_uccgsd_plan",
    "make_fixed_parameter_uccsd_plan",
    "make_fixed_parameter_upccgsd_plan",
    "make_hartree_fock_occupation",
    "make_uccgsd_operator_pool",
    "make_uccsd_operator_pool",
    "make_upccgsd_operator_pool",
    "state_from",
    "validate_fixed_parameter_ucc_plan",
    "validate_hartree_fock_occupation",
]
