# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Fermionic basis changes on an existing register, built from H, S, CNOT,
Rz and R1 (bit-controlled Rz and R1 for programmable plans).

The paired Jordan-Wigner Pauli rotations implement each complete Givens
mixing, including parity on intervening physical modes. Parameter words
remain caller-owned; this backend neither allocates qubits nor performs lookup.

Controlled variants take the enable at register index zero and shift every
other index by one. Each Pauli rotation is ``V Rz V†`` with a Clifford
scaffold ``V``, so the enable controls only the Rz and R1 rotations; with
the enable off the scaffold cancels exactly.
"""

import cudaq


# Gate tape: (opcode, first, second, parameter index, parameter sign).
# H=0, S=1, S-adjoint=2, CX=3, RZ=4, exact number phase R1=5.
def _gate_tape(schedule, adjoint=False):
    tape = []
    for operation in schedule.operations:
        first = schedule.structure.mode_map[operation.first]
        slot, sign = operation.parameter_index, operation.scale
        if operation.kind == "phase":
            tape.append((5, first, first, slot, sign))
            continue
        second = schedule.structure.mode_map[operation.second]
        if first > second:
            first, second, sign = second, first, -sign
        # G_pq(theta) = R_(Y Z... X)(-theta) R_(X Z... Y)(theta).
        # Each parity ladder includes all intervening physical modes.
        for y_endpoint, angle_sign in ((first, -sign), (second, sign)):
            tape.append((2, y_endpoint, y_endpoint, slot, 0.))
            tape.extend((0, p, p, slot, 0.) for p in (first, second))
            tape.extend((3, p, p + 1, slot, 0.) for p in range(first, second))
            tape.append((4, second, second, slot, angle_sign))
            tape.extend((3, p, p + 1, slot, 0.)
                        for p in range(second - 1, first - 1, -1))
            tape.extend((0, p, p, slot, 0.) for p in (first, second))
            tape.append((1, y_endpoint, y_endpoint, slot, 0.))
    if adjoint:
        tape = [(2 if code == 1 else 1 if code == 2 else code, first, second,
                 slot, -sign)
                for code, first, second, slot, sign in reversed(tape)]
    return tuple(tape)


def _capture(rows, used_second, offset):
    """Split ``(code, first, second, angle)`` rows into captured lists.

    CUDA-Q 0.16 fuses value-identical captured lists, so a tape without
    CNOT rows would turn the unused ``x.ctrl(register[first[k]],
    register[second[k]])`` branch into a CNOT with control equal to target,
    and lowering under ``cudaq.control`` never terminates
    (cuda-quantum#5794). An unused second index is therefore ``first + 1``,
    so ``second[k] != first[k]`` on every row and the two lists never
    coincide.
    """
    codes = [row[0] for row in rows]
    first = [row[1] + offset for row in rows]
    second = [(row[2] if row[0] in used_second else row[1] + 1) + offset
              for row in rows]
    angles = [row[3] for row in rows]
    return codes, first, second, angles


def classical_kernel(schedule, adjoint=False, controlled=False):
    """Capture a numerical schedule in a kernel accepting an existing qview.

    With ``controlled=True`` the enable is register index 0, every other
    index shifts by one, and only the rotations take the enable.
    """
    rows = [
        (code, first, second, sign * schedule.parameters[slot])
        for code, first, second, slot, sign in _gate_tape(schedule, adjoint)
    ]
    codes, first, second, angles = _capture(rows, (3, ), int(controlled))

    if controlled:

        @cudaq.kernel
        def apply_controlled_basis(register: cudaq.qview):
            for k in range(len(codes)):
                if codes[k] == 0:
                    h(register[first[k]])
                elif codes[k] == 1:
                    s(register[first[k]])
                elif codes[k] == 2:
                    s.adj(register[first[k]])
                elif codes[k] == 3:
                    x.ctrl(register[first[k]], register[second[k]])
                elif codes[k] == 4:
                    rz.ctrl(angles[k], register[0], register[first[k]])
                else:
                    r1.ctrl(angles[k], register[0], register[first[k]])

        return apply_controlled_basis

    @cudaq.kernel
    def apply_basis(register: cudaq.qview):
        for k in range(len(codes)):
            if codes[k] == 0:
                h(register[first[k]])
            elif codes[k] == 1:
                s(register[first[k]])
            elif codes[k] == 2:
                s.adj(register[first[k]])
            elif codes[k] == 3:
                x.ctrl(register[first[k]], register[second[k]])
            elif codes[k] == 4:
                rz(angles[k], register[first[k]])
            else:
                r1(angles[k], register[first[k]])

    return apply_basis


def programmable_kernel(schedule,
                        parameter_registers,
                        weights,
                        adjoint=False,
                        controlled=False):
    """Apply coherent parameter words through ordinary controlled rotations.

    ``parameter_registers`` contains one validated word per slot, in
    little-endian bit order; ``weights`` gives radians per word bit. Each
    word bit controls a rotation of ``sign * weights[bit]`` on the target
    qubit. The host expands those rotations into rows of (opcode, control,
    target, angle), so the kernel is one flat loop that CUDA-Q can reverse,
    with no mutable weight accumulator or machine-width integer shifts.
    Every slot is retained even when its numerical schedule value is zero.
    With ``controlled=True`` the enable is register index 0, every other
    index shifts by one, and only the rotations take the enable.
    """
    rows = []
    for code, first, second, slot, sign in _gate_tape(schedule, adjoint):
        if code < 4:
            # Clifford rows ignore their zero angle, as in classical_kernel.
            rows.append((code, first, second, 0.))
            continue
        # A rotation row names its target in both fields; each word bit
        # controls one weighted rotation of that target.
        word = parameter_registers[slot]
        rows.extend((code, word[bit], second, float(sign * weight))
                    for bit, weight in enumerate(weights))
    codes, first, second, angles = _capture(rows, (3, 4, 5), int(controlled))

    if controlled:

        @cudaq.kernel
        def apply_controlled_word_basis(register: cudaq.qview):
            for k in range(len(codes)):
                if codes[k] == 0:
                    h(register[first[k]])
                elif codes[k] == 1:
                    s(register[first[k]])
                elif codes[k] == 2:
                    s.adj(register[first[k]])
                elif codes[k] == 3:
                    x.ctrl(register[first[k]], register[second[k]])
                elif codes[k] == 4:
                    rz.ctrl(angles[k], register[0], register[first[k]],
                            register[second[k]])
                else:
                    r1.ctrl(angles[k], register[0], register[first[k]],
                            register[second[k]])

        return apply_controlled_word_basis

    @cudaq.kernel
    def apply_word_basis(register: cudaq.qview):
        for k in range(len(codes)):
            if codes[k] == 0:
                h(register[first[k]])
            elif codes[k] == 1:
                s(register[first[k]])
            elif codes[k] == 2:
                s.adj(register[first[k]])
            elif codes[k] == 3:
                x.ctrl(register[first[k]], register[second[k]])
            elif codes[k] == 4:
                rz.ctrl(angles[k], register[first[k]], register[second[k]])
            else:
                r1.ctrl(angles[k], register[first[k]], register[second[k]])

    return apply_word_basis


def programmable_split_kernel(schedule, parameter_registers, weights,
                              adjoint=False):
    """Word-driven kernel on separately allocated system and parameter views.

    Parameter indices are relative to the span after the physical system,
    preserving explicit layouts (including gaps). This is the same gate tape
    as ``programmable_kernel``; it does not allocate or load angle words.
    """
    span = schedule.structure.num_system_qubits
    tape = _gate_tape(schedule, adjoint)
    rows = []
    for code, first, second, slot, sign in tape:
        if code < 4:
            rows.append((code, first, second, 0, 0.))
        else:
            rows.extend((code, second, second,
                         parameter_registers[slot][bit]-span,
                         float(sign*weight))
                        for bit, weight in enumerate(weights))
    codes = [r[0] for r in rows]
    targets = [r[1] for r in rows]
    cx_left = [r[1] if r[0] == 3 else 0 for r in rows]
    cx_right = [r[2] if r[0] == 3 else 1 for r in rows]
    word_controls = [r[3] for r in rows]
    angles = [r[4] for r in rows]

    @cudaq.kernel
    def apply_split_word_basis(system: cudaq.qview, parameters: cudaq.qview):
        for k in range(len(codes)):
            if codes[k] == 0:
                h(system[targets[k]])
            elif codes[k] == 1:
                s(system[targets[k]])
            elif codes[k] == 2:
                s.adj(system[targets[k]])
            elif codes[k] == 3:
                x.ctrl(system[cx_left[k]], system[cx_right[k]])
            elif codes[k] == 4:
                rz.ctrl(angles[k], parameters[word_controls[k]], system[targets[k]])
            else:
                r1.ctrl(angles[k], parameters[word_controls[k]], system[targets[k]])

    return apply_split_word_basis


def phase_gradient_kernel(schedule, parameter_registers, resource,
                          adjoint=False, parameter_slots=None):
    """Separate-view basis change consuming a reusable phase-gradient state.

    Each complete real Givens has two opposite Rz angles on the same word.
    Replacing these by R1 gives opposite word-dependent scalar phases which
    cancel, also under coherent control. Number phases already are R1.
    This identity is for the complete paired tape, not a standalone Rz.
    """
    span=schedule.structure.num_system_qubits
    width=resource.width
    tape=_gate_tape(schedule,adjoint)
    compact=parameter_slots is not None
    slots=tuple(range(len(parameter_registers))) if parameter_slots is None else tuple(parameter_slots)
    positions={slot:i*width for i,slot in enumerate(slots)}
    if compact:tape=[row for row in tape if row[0]<4 or row[3] in positions]
    codes=[row[0] for row in tape]
    targets=[row[1] if row[0]<4 else row[2] for row in tape]
    left=[row[1] if row[0]==3 else 0 for row in tape]
    right=[row[2] if row[0]==3 else 1 for row in tape]
    offsets=[]
    for code,first,second,slot,sign in tape:
        if compact:
            offsets.append(positions.get(slot,0))
            continue
        word=parameter_registers[slot]
        if tuple(word)!=tuple(range(word[0],word[0]+width)):
            raise ValueError('phase-gradient backend requires contiguous parameter words')
        offsets.append(word[0]-span)
    # Avoid captured vector<bool> (packed bits) at the CUDA-Q boundary.
    signs=[1 if row[4]>0 else -1 for row in tape]
    forward=resource.phase_kernel();inverse=resource.phase_kernel(adjoint=True)
    @cudaq.kernel
    def apply_gradient_basis(system:cudaq.qview,parameters:cudaq.qview,
                             gradient:cudaq.qview,scratch:cudaq.qview):
        for k in range(len(codes)):
            if codes[k]==0:
                h(system[targets[k]])
            elif codes[k]==1:
                s(system[targets[k]])
            elif codes[k]==2:
                s.adj(system[targets[k]])
            elif codes[k]==3:
                x.ctrl(system[left[k]],system[right[k]])
            elif signs[k]>0:
                forward(parameters[offsets[k]:offsets[k]+width],system[targets[k]],gradient,scratch)
            else:
                inverse(parameters[offsets[k]:offsets[k]+width],system[targets[k]],gradient,scratch)
    return apply_gradient_basis


def one_particle_gradient_kernel(schedule,parameter_registers,resource,adjoint=False,parameter_slots=None):
    """Exact two-mode one-particle restriction on a single logical qubit.

    |0> means occupation of logical mode zero and |1> mode one. This is a
    validation/simulation sector encoding, not full-Fock-space synthesis.
    Unconditional word additions retain the otherwise lost scalar phases.
    """
    from ..primitives import add_register,subtract_register
    if schedule.structure.num_modes!=2 or schedule.structure.kind!='partial':
        raise ValueError('one-particle restriction currently requires a two-mode partial plan')
    width=resource.width;span=schedule.structure.num_system_qubits
    operations=list(schedule.operations)
    if parameter_slots is not None:
        positions={slot:i*width for i,slot in enumerate(parameter_slots)}
        operations=[op for op in operations if op.parameter_index in positions]
    if adjoint:operations=list(reversed(operations))
    kinds=[0 if op.kind=='givens' else op.first+1 for op in operations]
    signs=[int(op.scale*(-1 if adjoint else 1)) for op in operations]
    offsets=[(parameter_registers[op.parameter_index][0]-span if parameter_slots is None
              else positions[op.parameter_index]) for op in operations]
    for word in parameter_registers:
        if tuple(word)!=tuple(range(word[0],word[0]+width)):
            raise ValueError('contiguous angle words required')
    positive=resource.phase_kernel();negative=resource.phase_kernel(adjoint=True)
    @cudaq.kernel
    def apply_particle_basis(system:cudaq.qview,parameters:cudaq.qview,
                             gradient:cudaq.qview,scratch:cudaq.qview):
        for k in range(len(kinds)):
            if kinds[k]==0:
                s.adj(system[0]);h(system[0])
                if signs[k]>0:
                    positive(parameters[offsets[k]:offsets[k]+width],system[0],gradient,scratch)
                    positive(parameters[offsets[k]:offsets[k]+width],system[0],gradient,scratch)
                else:
                    negative(parameters[offsets[k]:offsets[k]+width],system[0],gradient,scratch)
                    negative(parameters[offsets[k]:offsets[k]+width],system[0],gradient,scratch)
                h(system[0]);s(system[0])
                if signs[k]>0:
                    subtract_register(parameters[offsets[k]:offsets[k]+width],gradient,scratch.back(1))
                else:
                    add_register(parameters[offsets[k]:offsets[k]+width],gradient,scratch.back(1))
            elif kinds[k]==1:
                if signs[k]>0:
                    add_register(parameters[offsets[k]:offsets[k]+width],gradient,scratch.back(1))
                    negative(parameters[offsets[k]:offsets[k]+width],system[0],gradient,scratch)
                else:
                    subtract_register(parameters[offsets[k]:offsets[k]+width],gradient,scratch.back(1))
                    positive(parameters[offsets[k]:offsets[k]+width],system[0],gradient,scratch)
            else:
                if signs[k]>0:
                    positive(parameters[offsets[k]:offsets[k]+width],system[0],gradient,scratch)
                else:
                    negative(parameters[offsets[k]:offsets[k]+width],system[0],gradient,scratch)
    return apply_particle_basis
