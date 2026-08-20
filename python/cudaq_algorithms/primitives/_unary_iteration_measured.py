# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unary iteration with measurement-based ancilla uncomputation (MBU).

``measured_unary_iteration_kernels`` mints the same SELECT channel as the
coherent walk in ``_unary_iteration`` — apply a per-address body exactly
when the address register equals ``k`` — but disposes of every temporary
AND ancilla by *measuring it away* instead of running the AND backwards.
This is the accounting behind the literature's headline "unary iteration
costs ``N - 1`` Toffolis" (Babbush et al., `arXiv:1805.03662`, Fig. 7):
AND computes are paid for, AND uncomputes are free. The uncompute gadget
is Gidney's (`arXiv:1709.06648`): measure the ladder line in the X basis;
the stored AND value is erased, but the random outcome may kick a ``-1``
phase onto the branch where both AND inputs are 1 — outcome 1 is repaired
by a CZ between the two inputs (X-conjugated exactly as the compute was)
plus a classically controlled X resetting the measured line to |0>. Both
outcomes are steered to the same final state, so the channel is a
deterministic unitary on (address, target) even though the procedure is
not a unitary circuit. (Qualtran's ``unary_iteration_bloq``/``And`` were
consulted as reference for conventions only; no code was adopted.)

The walk (the plain retrace form of Fig. 7, not the fused variant):
descend to leaf 0 computing one AND per level onto a clean ladder line
(X-conjugated address controls for 0-bits), apply ``body(0)``, and
between consecutive leaves retrace up to the common ancestor with one
measured uncompute per level, cross to the sibling with a single CNOT
from the parent line, and recompute back down with fresh ANDs. After the
last leaf the whole path is measured away. The uncontrolled root line is
linear in the address MSB (CNOT on the X-conjugated wire — no AND, no
measurement); with ``controlled=True`` the external control folds into
the tree root as one extra AND, and every line below is automatically
ANDed with the control.

Costs on a full tree of ``N = 2^num_address_bits`` addresses:
``N - 2`` AND computes uncontrolled, ``N - 1`` controlled, and exactly as
many measurements (every AND-computed line is measured away; the linear
root is not). Partial trees cost less; ``toffoli_count`` and
``num_measurements`` report the exact numbers. Compare the strictly
unitary fused walk's ``3 N / 2 - 5`` (uncontrolled; ``3 N / 2 - 1``
controlled). The T-count gap is larger than the Toffoli gap suggests:
every Toffoli here is a compute-from-|0> AND worth 4 T (Gidney), so the
measured walk is ~4 N T against the fused unitary walk's ~7.5 N T.

Why controlled MBU is legitimate — and why it is constructed explicitly:
nothing measured is ever controlled. The control folds into the tree
root at *construction*, so on the control-|0> branch every ladder line
simply stays |0>; the X-basis measurement of a |0> ladder line yields
uniformly random outcomes, but the fix-up CZ acts as the identity there
(its parent-line input is 0), so the gadget is unconditionally correct
across superpositions of the control. The constraint is framework, not
physics: CUDA-Q's automatic control-variant generation must not be
applied to a measuring kernel. On cudaq 0.15.1 ``cudaq.control`` on such
a kernel is *silently accepted* and produces the wrong channel on a
superposed control (the measurement collapses across control branches
and renormalizes them — not even outcome-deterministic); the test suite
pins that boundary. ``cudaq.sample`` rejects entry kernels that branch
on measurement results (the check does not see feedback inside
sub-kernels); use ``cudaq.run`` / ``cudaq.get_state``.

Public surface: this module mirrors ``unary_iteration_kernels`` exactly —
the same factory-time body callback contract and instruction vocabulary
(the extended body gates included: they are unitary gates on the
target/work registers and orthogonal to the uncompute strategy), the same
register signatures ``kernel(address, ladder, target[, work])`` with a
leading one-qubit ``control`` view when ``controlled=True``, the same
``describe()`` facility (the measured gadget appears in the decoded tape
as its own opcode). ``kernel_adj`` is the inverse walk: the leaves are
visited in reverse order with each address's (self-inverse) body gates
reversed, so the reversed walk's descents are AND computes and its
retraces measured uncomputes — the adjoint of an MBU circuit is itself an
MBU circuit, not a reversed tape.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Sequence

import cudaq

from ._unary_iteration import (_BODY_OPCODES, _EXTENDED_BODY_GATES, _OP_CCX,
                               _OP_CCX_CTRL, _OP_CX_ADDR_LADDER,
                               _OP_CX_CTRL_LADDER, _OP_CX_LADDER_LADDER,
                               _OP_DESCRIPTIONS, _OP_X_ADDR, _OP_X_LADDER,
                               _TOFFOLI_OPCODES, _WORK_OPERANDS,
                               _trailing_ones)

__all__ = ["MeasuredUnaryIterationKernels", "measured_unary_iteration_kernels"]

# Keep-alive registry (see _unary_iteration for the rationale); the
# minted kernel names below (measured_walk*) are unique to this module.
_LIVE_KERNELS: list = []


def _retain(*kernels) -> None:
    _LIVE_KERNELS.extend(kernels)


# Measured-uncompute opcodes, appended after the coherent walk's 0..20.
# Both erase ladder[a]: h(ladder[a]); m = mz(ladder[a]); on outcome 1,
# x(ladder[a]) resets the line to |0> and the CZ repairs the phase kick
# against the same two inputs the AND was computed from (any needed
# X-conjugation of the address wire is emitted around the gadget).
_OP_MEAS_UNCOMPUTE = 21  # ... fixup z.ctrl(ladder[b], address[c])
_OP_MEAS_UNCOMPUTE_CTRL = 22  # ... fixup z.ctrl(control[0], address[c])

_MEASURED_OPCODES = (_OP_MEAS_UNCOMPUTE, _OP_MEAS_UNCOMPUTE_CTRL)

_OP_DESCRIPTIONS_MEASURED = dict(_OP_DESCRIPTIONS)
_OP_DESCRIPTIONS_MEASURED[_OP_MEAS_UNCOMPUTE] = (
    "h(ladder[{a}]); mz -> fixup z.ctrl(ladder[{b}], address[{c}]), "
    "x(ladder[{a}])  # measured uncompute")
_OP_DESCRIPTIONS_MEASURED[_OP_MEAS_UNCOMPUTE_CTRL] = (
    "h(ladder[{a}]); mz -> fixup z.ctrl(control[0], address[{c}]), "
    "x(ladder[{a}])  # measured uncompute")


@dataclass(frozen=True)
class MeasuredUnaryIterationKernels:
    """The minted measured walk (see the module docstring for signatures).

    Attributes
    ----------
    kernel, kernel_adj
        The walk and its inverse walk (``kernel_adj`` is ``None`` when
        minted with ``include_adjoint=False``). Both contain mid-circuit
        measurements: run them with ``cudaq.get_state`` / ``cudaq.run``
        (``cudaq.sample`` rejects them) and never pass them to
        ``cudaq.control`` — the controlled variant must be minted with
        ``controlled=True`` instead.
    num_address, num_ladder, num_items
        Register widths (``num_ladder == num_address`` clean ancillas)
        and the number of iterated addresses.
    controlled
        Whether ``kernel`` takes the leading one-qubit control view.
    num_work
        Width of the trailing clean ``work`` view (0: no work view in the
        signature).
    toffoli_count
        Toffolis in one application — AND computes only (plus any body
        Toffolis); the uncomputes are measured and cost none.
    num_measurements
        Mid-circuit measurements in one application (one per measured
        uncompute, equal to the walk's AND computes on a trivial body).
    """

    kernel: Any
    kernel_adj: Any
    num_address: int
    num_ladder: int
    num_items: int
    controlled: bool
    toffoli_count: int
    num_measurements: int
    num_work: int = 0
    ops: Any = None

    def describe(self) -> str:
        """Decode the instruction tape into a human-readable gate listing.

        One line per instruction, in execution order. Lines tagged
        ``# Toffoli`` are the AND computes ``toffoli_count`` counts;
        lines tagged ``# measured uncompute`` are the Gidney gadgets
        ``num_measurements`` counts (each is one h, one mz and, on
        outcome 1, one fix-up CZ and one reset X).
        """
        control = ", controlled" if self.controlled else ""
        header = (f"measured unary-iteration walk: {self.num_items} "
                  f"addresses over {self.num_address} bits{control}; "
                  f"{self.toffoli_count} Toffolis, "
                  f"{self.num_measurements} measurements")
        lines = [
            _OP_DESCRIPTIONS_MEASURED[op].format(a=a, b=b, c=c)
            for op, a, b, c in self.ops
        ]
        return "\n".join([header] + lines)


def _measured_walk_toffoli_count(num_address_bits: int,
                                 num_items: int,
                                 controlled: bool = False) -> int:
    """Exact AND-compute count of the measured walk, without emitting it.

    Descent pays one AND per level below the root (the root is linear
    uncontrolled, one AND controlled), and the transition from leaf ``k``
    recomputes one AND per trailing one of ``k``; sibling crossings are
    CNOTs. On full trees this is ``N - 2`` uncontrolled, ``N - 1``
    controlled. The measurement count equals this on any tree shape
    (every AND-computed line is measured away exactly once).
    """
    n = num_address_bits
    total = n - 1 + (1 if controlled else 0)
    for k in range(num_items - 1):
        total += _trailing_ones(k)
    return total


def _emit_body_gates(ops: list, line: int, k: int,
                     gates: Sequence[tuple]) -> None:
    """Append the validated body instructions for address ``k``."""
    for item in gates:
        gate = item[0]
        operands = item[1:]
        if gate in _BODY_OPCODES:
            if len(operands) != 1:
                raise ValueError(f"body({k}) returned {gate!r} with "
                                 f"{len(operands)} operands (expected 1)")
            target = operands[0]
            if int(target) != target or target < 0:
                raise ValueError(f"body({k}) returned an invalid target "
                                 f"qubit index {target!r}")
            ops.append((_BODY_OPCODES[gate], line, int(target), 0))
        elif gate in _EXTENDED_BODY_GATES:
            opcode, kinds, leaf_slot = _EXTENDED_BODY_GATES[gate]
            if len(operands) != len(kinds):
                raise ValueError(
                    f"body({k}) returned {gate!r} with {len(operands)} "
                    f"operands (expected {len(kinds)})")
            slots = [0, 0, 0]
            cursor = 0
            if leaf_slot is not None:
                slots[leaf_slot] = line
                cursor = leaf_slot + 1
            for kind, operand in zip(kinds, operands):
                if int(operand) != operand or operand < 0:
                    name = "target" if kind == "t" else "work"
                    raise ValueError(f"body({k}) returned an invalid {name} "
                                     f"qubit index {operand!r}")
                slots[cursor] = int(operand)
                cursor += 1
            ops.append((opcode, slots[0], slots[1], slots[2]))
        else:
            raise ValueError(
                f"body({k}) returned unsupported gate {gate!r}: the "
                "unary-iteration body instruction set is 'x', 'y', 'z' "
                "plus the extended gates documented in "
                "cudaq_algorithms.primitives._unary_iteration")


def _emit_measured_walk(num_address_bits: int, num_items: int,
                        controlled: bool, bodies: Sequence[Sequence[tuple]],
                        reverse: bool) -> list:
    """Flatten the measured retrace walk into (opcode, a, b, c) tuples.

    ``reverse=True`` emits the inverse walk: the leaves are visited in
    descending address order with each body's (self-inverse) gates
    reversed. The transition machinery is direction-agnostic — between
    consecutive leaves it measures away the levels below the common
    ancestor against the *outgoing* leaf's bits and recomputes them
    against the incoming leaf's bits — so one emitter serves both.
    """
    n = num_address_bits
    ops: list[tuple[int, int, int, int]] = []

    def emit(opcode: int, a: int = 0, b: int = 0, c: int = 0) -> None:
        ops.append((opcode, a, b, c))

    def wire(level: int) -> int:
        """Address wire examined at 1-based tree level ``level``."""
        return n - level

    def bit_of(leaf: int, level: int) -> bool:
        return ((leaf >> wire(level)) & 1) == 1

    def compute_line(level: int, bit_is_one: bool) -> None:
        """ladder[level-1] <- parent AND (address bit == value): the AND
        compute (a Toffoli onto the clean line; linear at the
        uncontrolled root)."""
        a = wire(level)
        if not bit_is_one:
            emit(_OP_X_ADDR, a)
        if level == 1:
            if controlled:
                emit(_OP_CCX_CTRL, 0, a, 0)
            else:
                emit(_OP_CX_ADDR_LADDER, a, 0)
        else:
            emit(_OP_CCX, level - 2, a, level - 1)
        if not bit_is_one:
            emit(_OP_X_ADDR, a)

    def uncompute_line(level: int, bit_is_one: bool) -> None:
        """Erase ladder[level-1] (currently parent AND (bit == value)):
        the Gidney measured uncompute, X-conjugated as the compute was.
        The uncontrolled root is linear and uncomputed by its own CNOT
        (no measurement)."""
        a = wire(level)
        if level == 1 and not controlled:
            compute_line(level, bit_is_one)  # linear, self-inverse
            return
        if not bit_is_one:
            emit(_OP_X_ADDR, a)
        if level == 1:
            emit(_OP_MEAS_UNCOMPUTE_CTRL, 0, 0, a)
        else:
            emit(_OP_MEAS_UNCOMPUTE, level - 1, level - 2, a)
        if not bit_is_one:
            emit(_OP_X_ADDR, a)

    def cross_to_sibling(level: int) -> None:
        """Flip ladder[level-1] between the two siblings: given the
        parent, exactly one of the two is active, so their XOR is the
        parent line itself."""
        if level == 1:
            if controlled:
                emit(_OP_CX_CTRL_LADDER, 0, 0)
            else:
                emit(_OP_X_LADDER, 0)
        else:
            emit(_OP_CX_LADDER_LADDER, level - 2, level - 1)

    def body_at(k: int) -> None:
        gates = bodies[k]
        if reverse:
            gates = list(reversed(gates))
        _emit_body_gates(ops, n - 1, k, gates)

    leaves = list(range(num_items))
    if reverse:
        leaves.reverse()

    # Descent to the first leaf.
    for level in range(1, n + 1):
        compute_line(level, bit_of(leaves[0], level))
    body_at(leaves[0])

    for previous, leaf in zip(leaves, leaves[1:]):
        # Retrace up to the common ancestor, measuring the path away ...
        d = n - (previous ^ leaf).bit_length() + 1  # topmost changed level
        for level in range(n, d, -1):
            uncompute_line(level, bit_of(previous, level))
        # ... cross to the sibling subtree, and descend into it.
        cross_to_sibling(d)
        for level in range(d + 1, n + 1):
            compute_line(level, bit_of(leaf, level))
        body_at(leaf)

    # Measure the final path away.
    for level in range(n, 0, -1):
        uncompute_line(level, bit_of(leaves[-1], level))
    return ops


def measured_unary_iteration_kernels(
        num_address_bits: int,
        num_items: int,
        body: Callable[[int], Sequence[tuple]],
        *,
        controlled: bool = False,
        include_adjoint: bool = True,
        num_work: int | None = None) -> MeasuredUnaryIterationKernels:
    """Mint the measured unary-iteration walk for a factory-time body.

    The parameters and the body contract are identical to
    ``unary_iteration_kernels`` (see ``_unary_iteration``); ``body`` is
    called exactly once per address, in address order, and the returned
    gate lists are reused (reversed) for the inverse walk. The minted
    kernels contain mid-circuit measurements: they compose as sub-kernels
    of plain kernels, but must not be passed to ``cudaq.control`` (mint
    with ``controlled=True`` instead) nor run under ``cudaq.sample``.

    Parameters
    ----------
    num_address_bits
        Address register width (>= 1).
    num_items
        Number of iterated addresses (``1 <= num_items <=
        2^num_address_bits``); addresses ``k >= num_items`` are never
        entered and the walk acts as the identity on them.
    body
        Factory-time callback ``k -> sequence of (gate, operands...)``.
    controlled
        Mint the externally controlled walk (leading one-qubit view);
        the control folds into the tree root as one extra AND.
    include_adjoint
        Mint ``kernel_adj`` (the reverse-order walk with reversed
        bodies); pass ``False`` when the walk is known to be an
        involution (commuting self-inverse body, e.g. X-only QROM).
    num_work
        Width of the trailing clean ``work`` view. ``None`` (default)
        infers the width from the body's work usage.
    """
    if int(num_address_bits) != num_address_bits or num_address_bits < 1:
        raise ValueError("num_address_bits must be a positive integer")
    num_address_bits = int(num_address_bits)
    capacity = 1 << num_address_bits
    if int(num_items) != num_items or not 1 <= num_items <= capacity:
        raise ValueError(
            f"num_items must be an integer in [1, 2^num_address_bits = "
            f"{capacity}], got {num_items}")
    num_items = int(num_items)

    bodies = [list(body(k)) for k in range(num_items)]
    ops = _emit_measured_walk(num_address_bits,
                              num_items,
                              bool(controlled),
                              bodies,
                              reverse=False)
    required_work = 0
    for op in ops:
        for position in _WORK_OPERANDS.get(op[0], ()):
            required_work = max(required_work, op[1 + position] + 1)
    if num_work is None:
        num_work = required_work
    if int(num_work) != num_work or num_work < required_work:
        raise ValueError(
            f"num_work must be an integer >= the body's work usage "
            f"({required_work}), got {num_work}")
    num_work = int(num_work)

    toffolis = sum(1 for op in ops if op[0] in _TOFFOLI_OPCODES)
    measurements = sum(1 for op in ops if op[0] in _MEASURED_OPCODES)
    kernel = _mint_measured_interpreter(ops, bool(controlled), num_work > 0)
    kernel_adj = None
    if include_adjoint:
        adj_ops = _emit_measured_walk(num_address_bits,
                                      num_items,
                                      bool(controlled),
                                      bodies,
                                      reverse=True)
        kernel_adj = _mint_measured_interpreter(adj_ops, bool(controlled),
                                                num_work > 0)
    return MeasuredUnaryIterationKernels(kernel=kernel,
                                         kernel_adj=kernel_adj,
                                         num_address=num_address_bits,
                                         num_ladder=num_address_bits,
                                         num_items=num_items,
                                         controlled=bool(controlled),
                                         toffoli_count=toffolis,
                                         num_measurements=measurements,
                                         num_work=num_work,
                                         ops=tuple(ops))


def _mint_measured_interpreter(ops: list, controlled: bool, has_work: bool):
    """Mint the flat interpreter kernel over a flattened instruction list.

    Identical to the coherent interpreter plus the two measured-uncompute
    opcodes (21/22), whose classically controlled fix-ups run inside the
    dispatch loop — the load-bearing CUDA-Q capability this module rests
    on (verified on cudaq 0.15.1 / qpp-cpu).
    """
    num_ops = len(ops)
    opcodes = [op[0] for op in ops]
    ops_a = [op[1] for op in ops]
    ops_b = [op[2] for op in ops]
    ops_c = [op[3] for op in ops]
    if num_ops == 0:
        # An empty list must never cross the kernel boundary
        # (cuda-quantum#4847): pad with one never-dispatched entry.
        opcodes, ops_a, ops_b, ops_c = [-1], [0], [0], [0]

    if controlled and has_work:

        @cudaq.kernel
        def measured_walk_work_ctrl(control: cudaq.qview, address: cudaq.qview,
                                    ladder: cudaq.qview, target: cudaq.qview,
                                    work: cudaq.qview):
            fired = False
            for i in range(num_ops):
                op = opcodes[i]
                a = ops_a[i]
                b = ops_b[i]
                c = ops_c[i]
                if op == 0:
                    x(address[a])
                if op == 1:
                    x(ladder[a])
                if op == 2:
                    cx(address[a], ladder[b])
                if op == 3:
                    cx(ladder[a], ladder[b])
                if op == 4:
                    x.ctrl(ladder[a], address[b], ladder[c])
                if op == 5:
                    x.ctrl(ladder[a], target[b])
                if op == 6:
                    y.ctrl(ladder[a], target[b])
                if op == 7:
                    z.ctrl(ladder[a], target[b])
                if op == 8:
                    cx(control[0], ladder[b])
                if op == 9:
                    x.ctrl(control[0], address[b], ladder[c])
                if op == 10:
                    x(target[a])
                if op == 11:
                    cx(target[a], target[b])
                if op == 12:
                    x.ctrl(target[a], target[b], work[c])
                if op == 13:
                    x.ctrl(work[a], target[b], work[c])
                if op == 14:
                    cx(target[a], work[b])
                if op == 15:
                    x.ctrl(ladder[a], work[b], target[c])
                if op == 16:
                    z.ctrl(ladder[a], work[b])
                if op == 17:
                    z(ladder[a])
                if op == 21:
                    h(ladder[a])
                    fired = mz(ladder[a])
                    if fired:
                        x(ladder[a])
                        z.ctrl(ladder[b], address[c])
                if op == 22:
                    h(ladder[a])
                    fired = mz(ladder[a])
                    if fired:
                        x(ladder[a])
                        z.ctrl(control[0], address[c])

        _retain(measured_walk_work_ctrl)
        return measured_walk_work_ctrl

    if has_work:

        @cudaq.kernel
        def measured_walk_work(address: cudaq.qview, ladder: cudaq.qview,
                               target: cudaq.qview, work: cudaq.qview):
            fired = False
            for i in range(num_ops):
                op = opcodes[i]
                a = ops_a[i]
                b = ops_b[i]
                c = ops_c[i]
                if op == 0:
                    x(address[a])
                if op == 1:
                    x(ladder[a])
                if op == 2:
                    cx(address[a], ladder[b])
                if op == 3:
                    cx(ladder[a], ladder[b])
                if op == 4:
                    x.ctrl(ladder[a], address[b], ladder[c])
                if op == 5:
                    x.ctrl(ladder[a], target[b])
                if op == 6:
                    y.ctrl(ladder[a], target[b])
                if op == 7:
                    z.ctrl(ladder[a], target[b])
                if op == 10:
                    x(target[a])
                if op == 11:
                    cx(target[a], target[b])
                if op == 12:
                    x.ctrl(target[a], target[b], work[c])
                if op == 13:
                    x.ctrl(work[a], target[b], work[c])
                if op == 14:
                    cx(target[a], work[b])
                if op == 15:
                    x.ctrl(ladder[a], work[b], target[c])
                if op == 16:
                    z.ctrl(ladder[a], work[b])
                if op == 17:
                    z(ladder[a])
                if op == 21:
                    h(ladder[a])
                    fired = mz(ladder[a])
                    if fired:
                        x(ladder[a])
                        z.ctrl(ladder[b], address[c])

        _retain(measured_walk_work)
        return measured_walk_work

    if controlled:

        @cudaq.kernel
        def measured_walk_ctrl(control: cudaq.qview, address: cudaq.qview,
                               ladder: cudaq.qview, target: cudaq.qview):
            fired = False
            for i in range(num_ops):
                op = opcodes[i]
                a = ops_a[i]
                b = ops_b[i]
                c = ops_c[i]
                if op == 0:
                    x(address[a])
                if op == 1:
                    x(ladder[a])
                if op == 2:
                    cx(address[a], ladder[b])
                if op == 3:
                    cx(ladder[a], ladder[b])
                if op == 4:
                    x.ctrl(ladder[a], address[b], ladder[c])
                if op == 5:
                    x.ctrl(ladder[a], target[b])
                if op == 6:
                    y.ctrl(ladder[a], target[b])
                if op == 7:
                    z.ctrl(ladder[a], target[b])
                if op == 8:
                    cx(control[0], ladder[b])
                if op == 9:
                    x.ctrl(control[0], address[b], ladder[c])
                if op == 10:
                    x(target[a])
                if op == 11:
                    cx(target[a], target[b])
                if op == 17:
                    z(ladder[a])
                if op == 21:
                    h(ladder[a])
                    fired = mz(ladder[a])
                    if fired:
                        x(ladder[a])
                        z.ctrl(ladder[b], address[c])
                if op == 22:
                    h(ladder[a])
                    fired = mz(ladder[a])
                    if fired:
                        x(ladder[a])
                        z.ctrl(control[0], address[c])

        _retain(measured_walk_ctrl)
        return measured_walk_ctrl

    @cudaq.kernel
    def measured_walk(address: cudaq.qview, ladder: cudaq.qview,
                      target: cudaq.qview):
        fired = False
        for i in range(num_ops):
            op = opcodes[i]
            a = ops_a[i]
            b = ops_b[i]
            c = ops_c[i]
            if op == 0:
                x(address[a])
            if op == 1:
                x(ladder[a])
            if op == 2:
                cx(address[a], ladder[b])
            if op == 3:
                cx(ladder[a], ladder[b])
            if op == 4:
                x.ctrl(ladder[a], address[b], ladder[c])
            if op == 5:
                x.ctrl(ladder[a], target[b])
            if op == 6:
                y.ctrl(ladder[a], target[b])
            if op == 7:
                z.ctrl(ladder[a], target[b])
            if op == 10:
                x(target[a])
            if op == 11:
                cx(target[a], target[b])
            if op == 17:
                z(ladder[a])
            if op == 21:
                h(ladder[a])
                fired = mz(ladder[a])
                if fired:
                    x(ladder[a])
                    z.ctrl(ladder[b], address[c])

    _retain(measured_walk)
    return measured_walk
