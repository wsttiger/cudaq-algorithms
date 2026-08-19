# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Compiler-measured Toffoli cost of the unary-iteration walk.

The walk documents an O(N) Toffoli cost (at most ``2 * (N - 1)`` for N
iterated addresses) and reports its own emission count as
``toffoli_count``. These tests hold that claim against the compiler
rather than the emitter: ``cudaq.estimate_resources`` counts the ``ccx``
operations actually synthesized, which (a) must equal the emitter's
bookkeeping — a multi-control expansion or a decomposition surprise
would break the equality — and (b) must obey the explicit linear bound,
with the per-doubling increment making the O(N) scaling directly visible.

``estimate_resources`` traces the kernel without executing it, so no
simulator target or tolerance is involved; every assertion is an exact
integer comparison.
"""

import cudaq
import pytest

from cudaq_algorithms.sparse import unary_iteration_kernels

pytestmark = pytest.mark.skipif(
    not hasattr(cudaq, "estimate_resources"),
    reason="cudaq.estimate_resources is not available in this CUDA-Q")


def _compiled_toffolis(num_address_bits: int, num_items: int,
                       controlled: bool) -> tuple[int, int]:
    """Return (emitter toffoli_count, compiler ccx count) for one walk."""
    walk = unary_iteration_kernels(num_address_bits,
                                   num_items,
                                   lambda k: [("x", 0)],
                                   controlled=controlled,
                                   include_adjoint=False)
    kernel = walk.kernel
    width = num_address_bits

    if controlled:

        @cudaq.kernel
        def harness():
            control = cudaq.qvector(1)
            address = cudaq.qvector(width)
            ladder = cudaq.qvector(width)
            target = cudaq.qvector(1)
            kernel(control, address, ladder, target)
    else:

        @cudaq.kernel
        def harness():
            address = cudaq.qvector(width)
            ladder = cudaq.qvector(width)
            target = cudaq.qvector(1)
            kernel(address, ladder, target)

    resources = cudaq.estimate_resources(harness)
    # A Toffoli is an x with two controls; ``count_controls`` is the
    # arity-aware accessor (``count("ccx")`` matches nothing -- the
    # display name is not the lookup key).
    return walk.toffoli_count, resources.count_controls("x", 2)


@pytest.mark.parametrize("num_address_bits,num_items", [(2, 4), (3, 8), (3, 5),
                                                        (4, 16), (2, 1)])
@pytest.mark.parametrize("controlled", [False, True])
def test_compiled_toffolis_match_emitter_count(num_address_bits, num_items,
                                               controlled):
    # The emitter's cost accounting is only trustworthy if the compiler
    # synthesizes exactly the Toffolis it claims to emit.
    claimed, compiled = _compiled_toffolis(num_address_bits, num_items,
                                           controlled)
    assert compiled == claimed


def test_toffoli_cost_is_linear_in_num_items():
    # O(N) with the documented explicit constant: at most 2 * (N - 1)
    # Toffolis for a full tree of N addresses...
    compiled = {}
    for num_address_bits in (2, 3, 4, 5):
        num_items = 1 << num_address_bits
        _, count = _compiled_toffolis(num_address_bits,
                                      num_items,
                                      controlled=False)
        assert count <= 2 * (num_items - 1)
        compiled[num_items] = count

    # ...and the growth is linear per added address: doubling N adds the
    # N new addresses at no more than 2 Toffolis each. (The measured cost
    # is affine, 2N - 4, so the doubling *ratio* approaches 2 from above
    # — the increment, not the ratio, is the robust linearity check.)
    for num_items in (4, 8, 16):
        increment = compiled[2 * num_items] - compiled[num_items]
        assert 0 < increment <= 2 * num_items, (
            f"doubling N from {num_items} added {increment} Toffolis "
            f"(> 2 per new address): {compiled}")


def test_partial_tree_costs_no_more_than_full():
    # Skipped subtrees are never entered: a partial tree (num_items below
    # capacity) cannot cost more Toffolis than the full tree.
    _, full = _compiled_toffolis(4, 16, controlled=False)
    _, partial = _compiled_toffolis(4, 11, controlled=False)
    assert partial <= full


def test_wide_ancilla_even_moment_refuses_fast():
    # The even-moment reflection observable expands to 2^num_ancilla Pauli
    # terms; at this encoding's widths that would hang, not error. The
    # shared consumer must refuse immediately instead.
    import numpy as np

    from cudaq_algorithms import qubitization
    from cudaq_algorithms.sparse import SparseLCUEncoding

    rng = np.random.default_rng(7)
    dense = np.zeros((16, 16))
    for i, j in ((0, 3), (2, 9), (5, 5), (11, 14)):
        dense[i, j] = dense[j, i] = rng.uniform(-1.0, 1.0)
    encoding = SparseLCUEncoding(dense)
    assert encoding.num_ancilla > 20  # the pathological regime is real
    with pytest.raises(ValueError, match="walk_kernel"):
        qubitization.reflection_observable(encoding)
