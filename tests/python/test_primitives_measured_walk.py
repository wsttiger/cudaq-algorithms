# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the measured (MBU) unary-iteration walk.

The crux is behavioral equivalence: the measured walk and the coherent
fused walk must produce bitwise the same final statevector on the same
inputs — including nontrivially *superposed* address, target and control
states, where a misplaced fix-up CZ shows up as a relative sign that no
computational-basis input can see. Equivalence on superposed inputs plus
repeated-run bitwise consistency together prove the measure-and-fix-up
channel is the deterministic SELECT unitary, not merely one lucky
trajectory of it.

The remaining tests pin the bookkeeping (``N - 2`` / ``N - 1`` AND
computes on full trees, one measurement per AND, cross-checked against
``cudaq.estimate_resources``), the inverse walk, and the framework
boundary discovered while building this: on cudaq 0.15.1,
``cudaq.control`` applied to a measuring kernel is silently accepted but
does not implement the controlled channel on a superposed control, and
``cudaq.sample`` rejects measurement-branching kernels — but only when
the feedback sits in the entry kernel, not in a sub-kernel.
"""

import numpy as np
import pytest

import cudaq

from cudaq_algorithms.primitives import (measured_unary_iteration_kernels,
                                         unary_iteration_kernels)
from cudaq_algorithms.primitives._unary_iteration_measured import (
    _MEASURED_OPCODES, _measured_walk_toffoli_count)

_BODIES = {
    0: [("x", 0), ("y", 1)],
    1: [("z", 0)],
    2: [("y", 0), ("x", 1), ("z", 1)],
    4: [("x", 1)],
    5: [("y", 1), ("z", 0)],
}


def _body(k):
    return _BODIES.get(k, [])


def _superposed_states(num_bits, controlled, kernel):
    """Run ``kernel`` on the superposed reference input (uniform address,
    rotated control and targets) and return the statevector."""

    if controlled:

        @cudaq.kernel
        def run():
            control = cudaq.qvector(1)
            address_reg = cudaq.qvector(num_bits)
            ladder = cudaq.qvector(num_bits)
            target = cudaq.qvector(2)
            ry(0.7, control[0])
            for i in range(num_bits):
                h(address_reg[i])
            ry(0.3, target[0])
            ry(1.1, target[1])
            kernel(control, address_reg, ladder, target)
    else:

        @cudaq.kernel
        def run():
            address_reg = cudaq.qvector(num_bits)
            ladder = cudaq.qvector(num_bits)
            target = cudaq.qvector(2)
            for i in range(num_bits):
                h(address_reg[i])
            ry(0.3, target[0])
            ry(1.1, target[1])
            kernel(address_reg, ladder, target)

    return np.array(cudaq.get_state(run))


# ----------------------------------------------------------------------
# 1. Behavioral equivalence with the coherent walk (the crux)
# ----------------------------------------------------------------------


@pytest.mark.parametrize("controlled", [False, True])
@pytest.mark.parametrize("num_bits,num_items", [(1, 2), (2, 3), (3, 8), (3, 5),
                                                (4, 11)])
def test_measured_walk_equals_coherent_walk_on_superpositions(
        num_bits, num_items, controlled):
    # Superposed address x superposed target (x rotated control): the
    # fix-up CZ placement is invisible on computational-basis inputs and
    # shows up here as a relative sign. 1e-12, bitwise across runs.
    coherent = unary_iteration_kernels(num_bits,
                                       num_items,
                                       _body,
                                       controlled=controlled,
                                       include_adjoint=False)
    measured = measured_unary_iteration_kernels(num_bits,
                                                num_items,
                                                _body,
                                                controlled=controlled,
                                                include_adjoint=False)
    reference = _superposed_states(num_bits, controlled, coherent.kernel)
    # Repeated runs: the measurement outcomes differ, the state must not
    # (this is also the repeated-run determinism pin — both outcomes of
    # every gadget are steered to the same final state).
    for _ in range(6):
        state = _superposed_states(num_bits, controlled, measured.kernel)
        np.testing.assert_allclose(state, reference, atol=1e-12)


def test_measured_walk_ladder_returns_to_zero_mid_walk_branches():
    # A partial tree whose transitions exercise multi-level retraces
    # (fix-ups firing mid-walk) on a basis input: amplitude must sit
    # entirely in the ladder-|0> block afterwards. (Given the
    # equivalence test this is implied; asserted directly for locality.)
    walk = measured_unary_iteration_kernels(3, 6, _body)
    kernel = walk.kernel

    for address in range(8):

        @cudaq.kernel
        def run():
            address_reg = cudaq.qvector(3)
            ladder = cudaq.qvector(3)
            target = cudaq.qvector(2)
            for i in range(3):
                if ((address >> i) & 1) == 1:
                    x(address_reg[i])
            kernel(address_reg, ladder, target)

        state = np.array(cudaq.get_state(run))
        # Layout: address [0,3), ladder [3,6), target [6,8).
        for index in np.nonzero(np.abs(state) > 1e-12)[0]:
            assert (index >> 3) & 0b111 == 0, (
                f"ladder not |0> at address {address}")


# ----------------------------------------------------------------------
# 2. Cost bookkeeping: N - 2 / N - 1 AND computes, one measurement each
# ----------------------------------------------------------------------


def test_measured_walk_toffoli_and_measurement_accounting():
    for num_bits in range(1, 8):
        capacity = 1 << num_bits
        for controlled in (False, True):
            walk = measured_unary_iteration_kernels(num_bits,
                                                    capacity,
                                                    lambda k: [("x", 0)],
                                                    controlled=controlled,
                                                    include_adjoint=False)
            expected = capacity - 2 + (1 if controlled else 0)
            assert walk.toffoli_count == expected
            assert walk.num_measurements == walk.toffoli_count
            assert walk.toffoli_count == _measured_walk_toffoli_count(
                num_bits, capacity, controlled)
        # Partial trees: the analytic counter matches the emitted tape.
        for num_items in range(1, capacity + 1):
            walk = measured_unary_iteration_kernels(num_bits,
                                                    num_items,
                                                    lambda k: [],
                                                    include_adjoint=False)
            assert walk.toffoli_count == _measured_walk_toffoli_count(
                num_bits, num_items, False)
            assert walk.num_measurements == sum(1 for op in walk.ops
                                                if op[0] in _MEASURED_OPCODES)


@pytest.mark.skipif(not hasattr(cudaq, "estimate_resources"),
                    reason="cudaq.estimate_resources is not available in "
                    "this CUDA-Q")
def test_measured_walk_costs_match_the_compiler():
    # cudaq.estimate_resources accepts measuring kernels (probed on
    # cudaq 0.15.1): the synthesized ccx count must equal the emitter's
    # AND-compute bookkeeping and the mz count the measurement
    # bookkeeping — both are unconditional operations. The fix-up cz
    # sits inside the classically conditional branch, which the tracer
    # resolves along ONE deterministic trajectory (here 10 of the 14
    # branches fire, repeatably), so only its range is asserted.
    walk = measured_unary_iteration_kernels(4,
                                            16,
                                            lambda k: [("x", 0)],
                                            include_adjoint=False)
    kernel = walk.kernel

    @cudaq.kernel
    def harness():
        address_reg = cudaq.qvector(4)
        ladder = cudaq.qvector(4)
        target = cudaq.qvector(1)
        kernel(address_reg, ladder, target)

    resources = cudaq.estimate_resources(harness)
    assert resources.count_controls("x", 2) == walk.toffoli_count == 14
    assert resources.count("mz") == walk.num_measurements == 14
    assert 0 <= resources.count_controls("z", 1) <= walk.num_measurements


# ----------------------------------------------------------------------
# 3. The framework boundary: cudaq.control / cudaq.sample on measuring
#    kernels
# ----------------------------------------------------------------------


def test_cudaq_control_on_measuring_kernel_is_not_the_controlled_channel():
    # The physics allows controlled MBU (fold the control into the tree
    # root: nothing measured is ever controlled, and the fix-up CZ is
    # the identity on the control-off branch — which is exactly what
    # controlled=True mints). The framework path is the problem: on
    # cudaq 0.15.1 cudaq.control on a measuring kernel is silently
    # accepted and yields, on a superposed control, a state that is NOT
    # the controlled unitary's (the measurement collapses across control
    # branches and renormalizes them). Pin whichever failure mode this
    # CUDA-Q exhibits: an outright rejection, or the wrong channel.
    measured = measured_unary_iteration_kernels(2,
                                                4,
                                                _body,
                                                include_adjoint=False)
    inner = measured.kernel
    correct = measured_unary_iteration_kernels(2,
                                               4,
                                               _body,
                                               controlled=True,
                                               include_adjoint=False)
    reference = _superposed_states(2, True, correct.kernel)

    @cudaq.kernel
    def naive_controlled():
        control = cudaq.qvector(1)
        address_reg = cudaq.qvector(2)
        ladder = cudaq.qvector(2)
        target = cudaq.qvector(2)
        ry(0.7, control[0])
        for i in range(2):
            h(address_reg[i])
        ry(0.3, target[0])
        ry(1.1, target[1])
        cudaq.control(inner, control[0], address_reg, ladder, target)

    try:
        # Every trajectory must differ from the correct channel (there
        # are finitely many outcome patterns; a handful of runs already
        # exercised both branches of every gadget when this was pinned).
        for _ in range(6):
            state = np.array(cudaq.get_state(naive_controlled))
            assert not np.allclose(state, reference, atol=1e-9), (
                "cudaq.control on a measuring kernel now implements the "
                "controlled channel — revisit the explicit controlled "
                "construction and this pin")
    except RuntimeError:
        # Also acceptable: a CUDA-Q that rejects the construction
        # outright (the boundary enforced instead of silent).
        pass


def test_cudaq_sample_rejects_measurement_branching_kernel():
    # cudaq.sample refuses kernels that branch on measurement results —
    # but the check inspects only the ENTRY kernel: the same feedback
    # inside a minted sub-kernel slips past it (found while pinning
    # this; the sub-kernel path still executes, with a deprecation
    # warning about named measurement results). Pin the boundary where
    # it is enforced, on a directly sampled feedback kernel.
    with pytest.raises(RuntimeError, match="conditional feedback"):
        cudaq.sample(_direct_feedback, shots_count=10)


@cudaq.kernel
def _direct_feedback():
    q = cudaq.qvector(2)
    h(q[0])
    fired = mz(q[0])
    if fired:
        x(q[1])


# ----------------------------------------------------------------------
# 4. The inverse walk
# ----------------------------------------------------------------------


@pytest.mark.parametrize("controlled", [False, True])
def test_measured_walk_adjoint_composes_to_identity(controlled):
    # Non-commuting multi-gate bodies so the walk is not an involution;
    # only the reverse-order walk with reversed bodies restores the
    # state. Superposed inputs, repeated runs.
    def body(k):
        if k % 2 == 1:
            return [("x", 0), ("z", 0), ("y", 1)]
        return [("x", 1)]

    walk = measured_unary_iteration_kernels(3, 6, body, controlled=controlled)
    kernel = walk.kernel
    kernel_adj = walk.kernel_adj

    if controlled:

        @cudaq.kernel
        def run():
            control = cudaq.qvector(1)
            address_reg = cudaq.qvector(3)
            ladder = cudaq.qvector(3)
            target = cudaq.qvector(2)
            ry(0.7, control[0])
            for i in range(3):
                h(address_reg[i])
            ry(0.3, target[0])
            ry(1.1, target[1])
            kernel(control, address_reg, ladder, target)
            kernel_adj(control, address_reg, ladder, target)

        @cudaq.kernel
        def prepare_only():
            control = cudaq.qvector(1)
            address_reg = cudaq.qvector(3)
            ladder = cudaq.qvector(3)
            target = cudaq.qvector(2)
            ry(0.7, control[0])
            for i in range(3):
                h(address_reg[i])
            ry(0.3, target[0])
            ry(1.1, target[1])
    else:

        @cudaq.kernel
        def run():
            address_reg = cudaq.qvector(3)
            ladder = cudaq.qvector(3)
            target = cudaq.qvector(2)
            for i in range(3):
                h(address_reg[i])
            ry(0.3, target[0])
            ry(1.1, target[1])
            kernel(address_reg, ladder, target)
            kernel_adj(address_reg, ladder, target)

        @cudaq.kernel
        def prepare_only():
            address_reg = cudaq.qvector(3)
            ladder = cudaq.qvector(3)
            target = cudaq.qvector(2)
            for i in range(3):
                h(address_reg[i])
            ry(0.3, target[0])
            ry(1.1, target[1])

    reference = np.array(cudaq.get_state(prepare_only))
    for _ in range(5):
        state = np.array(cudaq.get_state(run))
        np.testing.assert_allclose(state, reference, atol=1e-12)


# ----------------------------------------------------------------------
# 5. Surface parity with the coherent walk
# ----------------------------------------------------------------------


def test_measured_walk_validation_raises():
    body = lambda k: []
    with pytest.raises(ValueError, match="num_address_bits must be a"):
        measured_unary_iteration_kernels(0, 1, body)
    with pytest.raises(ValueError, match=r"num_items must be an integer in"):
        measured_unary_iteration_kernels(2, 0, body)
    with pytest.raises(ValueError, match=r"num_items must be an integer in"):
        measured_unary_iteration_kernels(2, 5, body)
    with pytest.raises(ValueError, match="unsupported gate 'q'"):
        measured_unary_iteration_kernels(2, 4, lambda k: [("q", 0)])
    with pytest.raises(ValueError, match="invalid target qubit index"):
        measured_unary_iteration_kernels(2, 4, lambda k: [("x", -1)])


def test_measured_describe_decodes_the_tape():
    walk = measured_unary_iteration_kernels(2, 4, lambda k: [("x", 0)])
    text = walk.describe()
    lines = text.splitlines()
    assert lines[0].startswith("measured unary-iteration walk: 4 addresses")
    assert "2 measurements" in lines[0]
    assert sum("# Toffoli" in line for line in lines) == walk.toffoli_count
    gadget_lines = [line for line in lines if "# measured uncompute" in line]
    assert len(gadget_lines) == walk.num_measurements
    assert "mz -> fixup z.ctrl" in gadget_lines[0]
    body_lines = [line for line in lines if line.startswith("x(target[0])")]
    assert len(body_lines) == 4


def test_measured_walk_extended_body_vocabulary_ports():
    # The extended gates are unitary gates on target/work and orthogonal
    # to the uncompute strategy: a body using the AND-ladder gadgets and
    # the sign leaf must agree with the coherent walk gate for gate.
    def body(k):
        if k == 1:
            return [("sign", )]
        if k == 2:
            return [("and_tt", 0, 1, 0), ("z_w", 0), ("and_tt", 0, 1, 0)]
        return [("free_x", 0), ("x", 1), ("free_x", 0)]

    coherent = unary_iteration_kernels(2, 4, body, include_adjoint=False)
    measured = measured_unary_iteration_kernels(2,
                                                4,
                                                body,
                                                include_adjoint=False)
    assert measured.num_work == coherent.num_work == 1
    kc = coherent.kernel
    km = measured.kernel

    @cudaq.kernel
    def run_coherent():
        address_reg = cudaq.qvector(2)
        ladder = cudaq.qvector(2)
        target = cudaq.qvector(2)
        work = cudaq.qvector(1)
        for i in range(2):
            h(address_reg[i])
        ry(0.3, target[0])
        ry(1.1, target[1])
        kc(address_reg, ladder, target, work)

    @cudaq.kernel
    def run_measured():
        address_reg = cudaq.qvector(2)
        ladder = cudaq.qvector(2)
        target = cudaq.qvector(2)
        work = cudaq.qvector(1)
        for i in range(2):
            h(address_reg[i])
        ry(0.3, target[0])
        ry(1.1, target[1])
        km(address_reg, ladder, target, work)

    reference = np.array(cudaq.get_state(run_coherent))
    for _ in range(4):
        state = np.array(cudaq.get_state(run_measured))
        np.testing.assert_allclose(state, reference, atol=1e-12)
