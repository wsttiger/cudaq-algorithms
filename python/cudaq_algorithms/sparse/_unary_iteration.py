# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unary iteration: apply a per-address body once for each address value.

``unary_iteration_kernels`` mints a flat device kernel that walks a binary
tree over the address register (Babbush et al., `arXiv:1805.03662`):
one clean ladder ancilla per address bit carries the "this subtree is
active" line, computed with a Toffoli-and-CNOT AND gadget per internal
node, and the caller's body gates for address ``k`` fire controlled on
the leaf line — i.e. exactly when the address register equals ``k``.
Cost: at most ``2 (num_items - 1)`` Toffolis (skipped subtrees for
``num_items < 2^num_address_bits`` are never entered), ``num_address_bits``
clean ladder ancillas (|0> on entry, returned to |0>).

The callback pattern (READ THIS — it is the template for QROM and for the
sparse-LCU SELECT)
------------------------------------------------------------------------

CUDA-Q kernels are compiled from fixed Python source, so a callback can
never execute *inside* a kernel. The callback therefore runs at **factory
time**: ``body(k)`` is called once per address on the host and must return
the gate list for that address as data — a sequence of ``(gate, target)``
pairs with ``gate`` in ``{"x", "y", "z"}`` acting on ``target``-th qubit
of the target register, each implicitly controlled on the address-``k``
line. The emitter flattens the whole tree walk (ladder gadgets + body
gates) into parallel opcode/operand integer lists, and the minted kernel
is a single flat interpreter loop over those captured lists. Flatness is
load-bearing: CUDA-Q's control-variant generation rejects kernels that
call other kernels, so anything built this way stays
``cudaq.control``-compatible and can sit inside a controlled SELECT.

Kernel signatures (all registers little-endian, qubit 0 = LSB, per
``docs/conventions.md``):

- ``kernel(address: qview, ladder: qview, target: qview)`` — walk with
  the body applied at each active address.
- with ``controlled=True``:
  ``kernel(control: qview, address: qview, ladder: qview, target: qview)``
  where ``control`` is a one-qubit view rooting the tree — the whole walk
  acts as the identity when the control is |0>. (A separate one-qubit
  view, not a leading qubit of a combined register, so callers never mix
  a qview with a bare qubit in one control set.)

``kernel_adj`` is the hand-written inverse (``cudaq.adjoint`` is
off-limits, cuda-quantum#4897/#4898): every emitted gate is self-inverse
(X/CX/CCX and controlled Paulis), so the inverse is the same interpreter
over the reversed instruction list. When the body gates at every address
mutually commute and square to identity (e.g. the X-only QROM write), the
walk itself is an involution and ``include_adjoint=False`` skips minting
the redundant inverse.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Sequence

import cudaq

from ._sparse_oracle import _retain

__all__ = ["UnaryIterationKernels", "unary_iteration_kernels"]

# Interpreter opcodes (operands a, b, c index the signature's registers).
_OP_X_ADDR = 0  # x(address[a])
_OP_X_LADDER = 1  # x(ladder[a])
_OP_CX_ADDR_LADDER = 2  # cx(address[a], ladder[b])
_OP_CX_LADDER_LADDER = 3  # cx(ladder[a], ladder[b])
_OP_CCX = 4  # x.ctrl(ladder[a], address[b], ladder[c])
_OP_BODY_X = 5  # x.ctrl(ladder[a], target[b])
_OP_BODY_Y = 6  # y.ctrl(ladder[a], target[b])
_OP_BODY_Z = 7  # z.ctrl(ladder[a], target[b])
_OP_CX_CTRL_LADDER = 8  # cx(control[0], ladder[b])
_OP_CCX_CTRL = 9  # x.ctrl(control[0], address[b], ladder[c])

_BODY_OPCODES = {"x": _OP_BODY_X, "y": _OP_BODY_Y, "z": _OP_BODY_Z}


@dataclass(frozen=True)
class UnaryIterationKernels:
    """The minted walk (see the module docstring for signatures).

    Attributes
    ----------
    kernel, kernel_adj
        The walk and its hand-written inverse (``kernel_adj`` is ``None``
        when minted with ``include_adjoint=False``).
    num_address, num_ladder, num_items
        Register widths (``num_ladder == num_address`` clean ancillas)
        and the number of iterated addresses.
    controlled
        Whether ``kernel`` takes the leading one-qubit control view.
    toffoli_count
        Toffolis in one application (cost accounting).
    """

    kernel: Any
    kernel_adj: Any
    num_address: int
    num_ladder: int
    num_items: int
    controlled: bool
    toffoli_count: int


def _emit_walk(num_address_bits: int, num_items: int, controlled: bool,
               body: Callable[[int], Sequence[tuple[str, int]]]) -> list:
    """Flatten the tree walk into (opcode, a, b, c) instructions."""
    ops: list[tuple[int, int, int, int]] = []

    def emit(opcode: int, a: int = 0, b: int = 0, c: int = 0) -> None:
        ops.append((opcode, a, b, c))

    def emit_body(line: int, k: int) -> None:
        for gate, target in body(k):
            if gate not in _BODY_OPCODES:
                raise ValueError(
                    f"body({k}) returned unsupported gate {gate!r}: the "
                    "unary-iteration body instruction set is 'x', 'y', 'z'")
            if int(target) != target or target < 0:
                raise ValueError(f"body({k}) returned an invalid target "
                                 f"qubit index {target!r}")
            emit(_BODY_OPCODES[gate], line, int(target))

    def node(ctrl_line: int, bit: int, base: int) -> None:
        """Walk the subtree over address bits [0, bit] rooted at ``base``.

        ``ctrl_line``: ladder index of the active line, or -1 for the
        tree root (external control / unconditioned).
        """
        if bit < 0:
            emit_body(ctrl_line, base)
        else:
            anc = num_address_bits - 1 - bit  # ladder level for this bit
            has_right = base + (1 << bit) < num_items
            if ctrl_line < 0 and not controlled:
                # Root, unconditioned: anc <- NOT address[bit].
                emit(_OP_X_ADDR, bit)
                emit(_OP_CX_ADDR_LADDER, bit, anc)
                emit(_OP_X_ADDR, bit)
                node(anc, bit - 1, base)
                if has_right:
                    emit(_OP_X_LADDER, anc)  # anc: NOT b -> b
                    node(anc, bit - 1, base + (1 << bit))
                    emit(_OP_CX_ADDR_LADDER, bit, anc)  # anc: b -> 0
                else:
                    emit(_OP_X_ADDR, bit)
                    emit(_OP_CX_ADDR_LADDER, bit, anc)
                    emit(_OP_X_ADDR, bit)
            else:
                # anc <- ctrl AND NOT address[bit] (Toffoli then CNOT).
                if ctrl_line < 0:
                    emit(_OP_CCX_CTRL, 0, bit, anc)
                    emit(_OP_CX_CTRL_LADDER, 0, anc)
                else:
                    emit(_OP_CCX, ctrl_line, bit, anc)
                    emit(_OP_CX_LADDER_LADDER, ctrl_line, anc)
                node(anc, bit - 1, base)
                if has_right:
                    # anc: ctrl AND NOT b -> ctrl AND b.
                    if ctrl_line < 0:
                        emit(_OP_CX_CTRL_LADDER, 0, anc)
                    else:
                        emit(_OP_CX_LADDER_LADDER, ctrl_line, anc)
                    node(anc, bit - 1, base + (1 << bit))
                    if ctrl_line < 0:
                        emit(_OP_CCX_CTRL, 0, bit, anc)
                    else:
                        emit(_OP_CCX, ctrl_line, bit, anc)
                else:
                    if ctrl_line < 0:
                        emit(_OP_CX_CTRL_LADDER, 0, anc)
                        emit(_OP_CCX_CTRL, 0, bit, anc)
                    else:
                        emit(_OP_CX_LADDER_LADDER, ctrl_line, anc)
                        emit(_OP_CCX, ctrl_line, bit, anc)

    node(-1, num_address_bits - 1, 0)
    return ops


def unary_iteration_kernels(
        num_address_bits: int,
        num_items: int,
        body: Callable[[int], Sequence[tuple[str, int]]],
        *,
        controlled: bool = False,
        include_adjoint: bool = True) -> UnaryIterationKernels:
    """Mint the unary-iteration walk for a factory-time body callback.

    Parameters
    ----------
    num_address_bits
        Address register width (>= 1).
    num_items
        Number of iterated addresses (``1 <= num_items <=
        2^num_address_bits``); addresses ``k >= num_items`` are never
        entered and the walk acts as the identity on them.
    body
        Factory-time callback ``k -> sequence of (gate, target)`` (see
        the module docstring). Called exactly once per address, in
        address order, during this factory call.
    controlled
        Mint the externally controlled walk (leading one-qubit view).
    include_adjoint
        Mint ``kernel_adj``; pass ``False`` when the walk is known to be
        an involution (commuting self-inverse body, e.g. X-only QROM).
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

    ops = _emit_walk(num_address_bits, num_items, bool(controlled), body)
    toffolis = sum(1 for op in ops if op[0] in (_OP_CCX, _OP_CCX_CTRL))
    kernel = _mint_interpreter(ops, bool(controlled))
    kernel_adj = None
    if include_adjoint:
        kernel_adj = _mint_interpreter(list(reversed(ops)), bool(controlled))
    return UnaryIterationKernels(kernel=kernel,
                                 kernel_adj=kernel_adj,
                                 num_address=num_address_bits,
                                 num_ladder=num_address_bits,
                                 num_items=num_items,
                                 controlled=bool(controlled),
                                 toffoli_count=toffolis)


def _mint_interpreter(ops: list, controlled: bool):
    """Mint the flat interpreter kernel over a flattened instruction list.

    Every emitted gate is self-inverse, so the inverse walk is this same
    interpreter over the reversed list — how ``kernel_adj`` is minted.
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

    if controlled:

        @cudaq.kernel
        def sparse_unary_walk_ctrl(control: cudaq.qview, address: cudaq.qview,
                                   ladder: cudaq.qview, target: cudaq.qview):
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

        _retain(sparse_unary_walk_ctrl)
        return sparse_unary_walk_ctrl

    @cudaq.kernel
    def sparse_unary_walk(address: cudaq.qview, ladder: cudaq.qview,
                          target: cudaq.qview):
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

    _retain(sparse_unary_walk)
    return sparse_unary_walk
