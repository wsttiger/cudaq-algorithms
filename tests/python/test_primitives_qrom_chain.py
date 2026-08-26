# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for sequential-QROM chaining and bit packets (arXiv:2605.20334,
Secs. II.C and II.D, adapted to clean coherent ancillas).

Behavioral: a QROMChain (fused difference tables and the naive
load/unload reference) must be indistinguishable from ``m`` independent
QROMs when arbitrary caller operations act on the output between steps —
checked on superposed addresses with phase-carrying rotations
interleaved, full-state comparison. Bit-packet lookups (``alpha > 1``,
including ``alpha`` not dividing ``output_bits`` and non-power-of-two
table lengths) must read out exhaustively identically to the plain
variants and uncompute themselves on double application (pinned, not
assumed).

Resource: compiled ccx counts must equal the derived coherent formulas —
``(alpha + 1) W(A - log2 B, ceil(N/B)) + alpha W(log2 B, B) + b (B - 1)``
for packets, ``(m + 1) C`` vs ``2 m C`` for fused vs naive chains (the
margin ``(m - 1) C`` pinned exactly) — and auto pricing must match a
brute-force enumeration on every side of the ancilla-budget crossovers
(with unconstrained ancillas ``alpha = 1`` provably dominates; packets
win only under ``max_ancillas``).
"""

import numpy as np
import pytest

import cudaq

from cudaq_algorithms.primitives import QROM, QROMChain
from cudaq_algorithms.primitives._qrom import _select_copy_cost
from cudaq_algorithms.primitives._unary_iteration import _walk_toffoli_count

# ---------------------------------------------------------------------------
# Behavioral: bit packets (Sec. II.D)
# ---------------------------------------------------------------------------

# Register layout in the harnesses: address at [0, A), ladder at
# [A, A + L), output at [A + L, A + L + W).


def _basis(index: int, num_qubits: int) -> np.ndarray:
    ket = np.zeros(1 << num_qubits, dtype=np.complex128)
    ket[index] = 1.0
    return ket


def _qrom_readout_state(qrom: QROM, address: int) -> np.ndarray:
    lookup = qrom.kernel()
    num_address = qrom.num_address
    num_ladder = qrom.num_ladder
    num_output = qrom.num_output

    @cudaq.kernel
    def run():
        address_reg = cudaq.qvector(num_address)
        ladder = cudaq.qvector(num_ladder)
        output = cudaq.qvector(num_output)
        for b in range(num_address):
            if ((address >> b) & 1) == 1:
                x(address_reg[b])
        lookup(address_reg, ladder, output)

    return np.array(cudaq.get_state(run))


@pytest.mark.parametrize(
    "address_bits,num_entries,output_bits,block_size,"
    "alpha", [(3, 8, 2, 2, 2), (3, 5, 3, 2, 2), (3, 8, 3, 4, 3),
              (4, 13, 5, 4, 2), (4, 11, 5, 4, 3), (4, 16, 4, 8, 4),
              (4, 9, 4, 2, 4)])
def test_qrom_bit_packets_exhaustive_readout(address_bits, num_entries,
                                             output_bits, block_size, alpha):
    # alpha not dividing output_bits is included (5 bits over 2 and 3
    # packets): the slices are balanced, widest first, no padding.
    rng = np.random.default_rng(11 * address_bits + num_entries + alpha)
    data = [int(v) for v in rng.integers(0, 1 << output_bits, num_entries)]
    qrom = QROM(data,
                address_bits,
                output_bits,
                variant="select_copy",
                block_size=block_size,
                alpha=alpha)
    assert qrom.variant == "select_copy"
    assert qrom.alpha == alpha
    low_bits = block_size.bit_length() - 1
    packet = -(-output_bits // alpha)
    assert qrom.num_ladder == (max(address_bits - low_bits, low_bits) +
                               (block_size - 1) * packet)
    total = qrom.num_address + qrom.num_ladder + qrom.num_output
    shift = qrom.num_address + qrom.num_ladder
    for address in range(1 << address_bits):
        expected_word = data[address] if address < len(data) else 0
        state = _qrom_readout_state(qrom, address)
        expected = _basis(address + (expected_word << shift), total)
        np.testing.assert_allclose(state, expected, atol=1e-12)


def test_qrom_bit_packets_agree_with_select_on_superposed_addresses():
    data = [3, 0, 5, 6, 1, 7]
    address_bits, output_bits = 3, 3

    def joint_amplitudes(qrom: QROM) -> np.ndarray:
        lookup = qrom.kernel()
        num_ladder = qrom.num_ladder

        @cudaq.kernel
        def run():
            address_reg = cudaq.qvector(address_bits)
            ladder = cudaq.qvector(num_ladder)
            output = cudaq.qvector(output_bits)
            for b in range(address_bits):
                h(address_reg[b])
            lookup(address_reg, ladder, output)

        state = np.array(cudaq.get_state(run))
        tensor = state.reshape(1 << output_bits, 1 << num_ladder,
                               1 << address_bits)
        np.testing.assert_allclose(np.delete(tensor, 0, axis=1),
                                   0.0,
                                   atol=1e-12)
        return tensor[:, 0, :]

    reference = joint_amplitudes(
        QROM(data, address_bits, output_bits, variant="select"))
    for block_size in (2, 4):
        for alpha in (2, 3):
            packets = QROM(data,
                           address_bits,
                           output_bits,
                           variant="select_copy",
                           block_size=block_size,
                           alpha=alpha)
            np.testing.assert_allclose(reference,
                                       joint_amplitudes(packets),
                                       atol=1e-12)


@pytest.mark.parametrize("block_size,alpha", [(2, 2), (4, 3), (4, 2)])
def test_qrom_bit_packets_double_application_uncomputes(block_size, alpha):
    # Pinned, not assumed: from the contractual clean-ladder sector the
    # packet lookup XORs the same table twice, restoring any initial
    # output word on every superposed address branch (the global
    # involution of alpha = 1 is NOT claimed at alpha > 1 — see the
    # module docstring of _qrom).
    data = [5, 3, 0, 6, 7]
    qrom = QROM(data,
                address_bits=3,
                output_bits=3,
                variant="select_copy",
                block_size=block_size,
                alpha=alpha)
    lookup = qrom.kernel()
    num_ladder = qrom.num_ladder
    initial = 0b101

    @cudaq.kernel
    def run():
        address_reg = cudaq.qvector(3)
        ladder = cudaq.qvector(num_ladder)
        output = cudaq.qvector(3)
        for b in range(3):
            h(address_reg[b])
            if ((initial >> b) & 1) == 1:
                x(output[b])
        lookup(address_reg, ladder, output)
        lookup(address_reg, ladder, output)

    total = 3 + num_ladder + 3
    state = np.array(cudaq.get_state(run))
    expected = np.zeros(1 << total, dtype=np.complex128)
    for address in range(8):
        expected[address + (initial << (3 + num_ladder))] = 1.0 / np.sqrt(8.0)
    np.testing.assert_allclose(state, expected, atol=1e-12)


# ---------------------------------------------------------------------------
# Behavioral: the chain (Sec. II.C)
# ---------------------------------------------------------------------------


def _chain_tables():
    rng = np.random.default_rng(42)
    lengths = (8, 6, 7)  # non-power-of-two lengths included
    return [[int(v) for v in rng.integers(0, 8, n)] for n in lengths]


def _interleaved_state(kernel_list, num_ladder, thetas):
    """Full state of: prep superposed address, then kernel / caller-op
    alternation with per-step rotations + entangling CNOT on the output.

    ``kernel_list`` has 4 kernels (3 tables); the caller ops between
    them are phase-carrying, so any ordering or fusion error shows up in
    the amplitudes exactly.
    """
    k0, k1, k2, k3 = kernel_list
    t0, t1, t2 = thetas

    @cudaq.kernel
    def run():
        address_reg = cudaq.qvector(3)
        ladder = cudaq.qvector(num_ladder)
        output = cudaq.qvector(3)
        for b in range(3):
            h(address_reg[b])
            rz(0.3 + 0.2 * b, address_reg[b])
        k0(address_reg, ladder, output)
        ry(t0, output[0])
        cx(output[0], output[1])
        rz(t0, output[2])
        k1(address_reg, ladder, output)
        ry(t1, output[1])
        cx(output[1], output[2])
        rz(t1, output[0])
        k2(address_reg, ladder, output)
        ry(t2, output[2])
        cx(output[2], output[0])
        rz(t2, output[1])
        k3(address_reg, ladder, output)

    return np.array(cudaq.get_state(run))


# "auto" is deliberately absent: its dispatch decision is pinned by the
# factory tests, and each parametrization here retains ~1 GB of compiled
# kernels for the life of the process (CI runners have 7 GB).
@pytest.mark.parametrize("variant", ["select", "select_copy"])
@pytest.mark.parametrize("fused", [True, False])
def test_chain_matches_independent_qroms_with_interleaved_ops(fused, variant):
    # The semantics contract: chain fused == chain naive == m independent
    # QROMs (load, caller op, unload each), with arbitrary caller
    # operations on the output between the steps.
    tables = _chain_tables()
    thetas = (0.7, 1.3, 2.1)
    chain = QROMChain(tables, 3, 3, fused=fused, variant=variant)
    kernels = chain.kernels()
    assert len(kernels) == chain.num_tables + 1 == 4
    state = _interleaved_state(kernels, chain.num_ladder, thetas)

    # Reference: three independent lookups, each applied twice around
    # its caller op (load + self-inverse unload), on the same ladder
    # width (unused ladder lines are never touched).
    reference_qroms = [QROM(table, 3, 3, variant=variant) for table in tables]
    q0, q1, q2 = [q.kernel() for q in reference_qroms]
    t0, t1, t2 = thetas
    num_ladder = chain.num_ladder

    @cudaq.kernel
    def reference():
        address_reg = cudaq.qvector(3)
        ladder = cudaq.qvector(num_ladder)
        output = cudaq.qvector(3)
        for b in range(3):
            h(address_reg[b])
            rz(0.3 + 0.2 * b, address_reg[b])
        q0(address_reg, ladder, output)
        ry(t0, output[0])
        cx(output[0], output[1])
        rz(t0, output[2])
        q0(address_reg, ladder, output)
        q1(address_reg, ladder, output)
        ry(t1, output[1])
        cx(output[1], output[2])
        rz(t1, output[0])
        q1(address_reg, ladder, output)
        q2(address_reg, ladder, output)
        ry(t2, output[2])
        cx(output[2], output[0])
        rz(t2, output[1])
        q2(address_reg, ladder, output)

    expected = np.array(cudaq.get_state(reference))
    np.testing.assert_allclose(state, expected, atol=1e-12)


@pytest.mark.parametrize("fused", [True, False])
@pytest.mark.parametrize("variant,block_size", [("select", None),
                                                ("select_copy", 2)])
def test_chain_prefix_readout_holds_each_table(fused, variant, block_size):
    # After step kernel j the output must hold tables[j][k] (zero for
    # out-of-range addresses of a short table); after the final kernel
    # it must be restored to zero. Exhaustive over addresses.
    tables = _chain_tables()
    chain = QROMChain(tables,
                      3,
                      3,
                      fused=fused,
                      variant=variant,
                      block_size=block_size)
    kernels = chain.kernels()
    num_ladder = chain.num_ladder
    shift = 3 + num_ladder
    total = shift + 3
    # One harness per prefix length applying that many step kernels
    # (explicit unrolled calls — kernels cannot be looped over inside a
    # CUDA-Q kernel). The address is a RUNTIME ARGUMENT: a closure-baked
    # address would mint (and retain, compiled) a fresh kernel per
    # address — 8x the JIT memory for identical circuits.
    k0, k1, k2, k3 = kernels

    @cudaq.kernel
    def run1(address: int):
        address_reg = cudaq.qvector(3)
        ladder = cudaq.qvector(num_ladder)
        output = cudaq.qvector(3)
        for b in range(3):
            if ((address >> b) & 1) == 1:
                x(address_reg[b])
        k0(address_reg, ladder, output)

    @cudaq.kernel
    def run2(address: int):
        address_reg = cudaq.qvector(3)
        ladder = cudaq.qvector(num_ladder)
        output = cudaq.qvector(3)
        for b in range(3):
            if ((address >> b) & 1) == 1:
                x(address_reg[b])
        k0(address_reg, ladder, output)
        k1(address_reg, ladder, output)

    @cudaq.kernel
    def run3(address: int):
        address_reg = cudaq.qvector(3)
        ladder = cudaq.qvector(num_ladder)
        output = cudaq.qvector(3)
        for b in range(3):
            if ((address >> b) & 1) == 1:
                x(address_reg[b])
        k0(address_reg, ladder, output)
        k1(address_reg, ladder, output)
        k2(address_reg, ladder, output)

    @cudaq.kernel
    def run4(address: int):
        address_reg = cudaq.qvector(3)
        ladder = cudaq.qvector(num_ladder)
        output = cudaq.qvector(3)
        for b in range(3):
            if ((address >> b) & 1) == 1:
                x(address_reg[b])
        k0(address_reg, ladder, output)
        k1(address_reg, ladder, output)
        k2(address_reg, ladder, output)
        k3(address_reg, ladder, output)

    prefix_kernels = {1: run1, 2: run2, 3: run3, 4: run4}

    def prefix_state(address, prefix):
        return np.array(cudaq.get_state(prefix_kernels[prefix], address))

    for prefix in (1, 2, 3, 4):
        for address in range(8):
            if prefix <= 3:
                table = tables[prefix - 1]
                word = table[address] if address < len(table) else 0
            else:
                word = 0
            expected = _basis(address + (word << shift), total)
            np.testing.assert_allclose(prefix_state(address, prefix),
                                       expected,
                                       atol=1e-12)


def test_chain_step_kernels_self_inverse_on_clean_sector():
    # Each step kernel XORs a fixed table into the output and restores
    # the ladder, so applying it twice from a clean ladder is the
    # identity — pinned on superposed addresses with a non-zero initial
    # output word.
    tables = _chain_tables()
    chain = QROMChain(tables, 3, 3, fused=True)
    num_ladder = chain.num_ladder
    initial = 0b011
    total = 3 + num_ladder + 3
    expected = np.zeros(1 << total, dtype=np.complex128)
    for address in range(8):
        expected[address + (initial << (3 + num_ladder))] = 1.0 / np.sqrt(8.0)
    for step in chain.kernels():

        @cudaq.kernel
        def run():
            address_reg = cudaq.qvector(3)
            ladder = cudaq.qvector(num_ladder)
            output = cudaq.qvector(3)
            for b in range(3):
                h(address_reg[b])
                if ((initial >> b) & 1) == 1:
                    x(output[b])
            step(address_reg, ladder, output)
            step(address_reg, ladder, output)

        np.testing.assert_allclose(np.array(cudaq.get_state(run)),
                                   expected,
                                   atol=1e-12)


# ---------------------------------------------------------------------------
# Resource contracts
# ---------------------------------------------------------------------------

_HAS_RESOURCES = hasattr(cudaq, "estimate_resources")
needs_resources = pytest.mark.skipif(
    not _HAS_RESOURCES,
    reason="cudaq.estimate_resources is not available in this CUDA-Q")


def _compiled_toffolis(kernel, *widths) -> int:
    w0, w1, w2 = widths

    @cudaq.kernel
    def harness():
        r0 = cudaq.qvector(w0)
        r1 = cudaq.qvector(w1)
        r2 = cudaq.qvector(w2)
        kernel(r0, r1, r2)

    resources = cudaq.estimate_resources(harness)
    return resources.count_controls("x", 2)


@needs_resources
@pytest.mark.parametrize("block_size,alpha", [(2, 1), (2, 2), (4, 1), (4, 2),
                                              (4, 3), (8, 2), (8, 4)])
def test_qrom_packets_compiled_toffolis_match_derived_formula(
        block_size, alpha):
    # The derived coherent packet cost:
    #   (alpha + 1) W(A - log2 B, ceil(N/B)) + alpha W(log2 B, B)
    #   + b (B - 1)
    # (the copy term is alpha-independent: sum_s width_s = b), held
    # against both the emitter bookkeeping and the compiler.
    data = [3, 0, 5, 6, 1, 7, 2, 2, 4, 1, 6, 7, 0]
    address_bits, output_bits = 4, 4
    qrom = QROM(data,
                address_bits,
                output_bits,
                variant="select_copy",
                block_size=block_size,
                alpha=alpha)
    low_bits = block_size.bit_length() - 1
    num_blocks = -(-len(data) // block_size)
    expected = ((alpha + 1) *
                _walk_toffoli_count(address_bits - low_bits, num_blocks) +
                alpha * _walk_toffoli_count(low_bits, block_size) +
                output_bits * (block_size - 1))
    assert qrom.toffoli_count == expected
    assert expected == _select_copy_cost(address_bits, len(data), output_bits,
                                         block_size, alpha)
    compiled = _compiled_toffolis(qrom.kernel(), qrom.num_address,
                                  qrom.num_ladder, qrom.num_output)
    assert compiled == expected


def test_alpha_sweep_interpolates_the_walk_prefactor():
    # At fixed B each alpha increment adds exactly one block walk and
    # one copy walk — the (alpha + 1)-flavored prefactor interpolation —
    # so with unconstrained ancillas alpha = 1 dominates pointwise and
    # auto must never pick packets.
    data = [int(v) for v in np.arange(256) % 16]
    address_bits, output_bits = 8, 4
    for block_size in (4, 8, 16):
        low_bits = block_size.bit_length() - 1
        per_step = (
            _walk_toffoli_count(address_bits - low_bits, 256 // block_size) +
            _walk_toffoli_count(low_bits, block_size))
        base = QROM(data,
                    address_bits,
                    output_bits,
                    variant="select_copy",
                    block_size=block_size,
                    alpha=1)
        for alpha in (2, 3, 4):
            packets = QROM(data,
                           address_bits,
                           output_bits,
                           variant="select_copy",
                           block_size=block_size,
                           alpha=alpha)
            assert (packets.toffoli_count - base.toffoli_count == (alpha - 1) *
                    per_step)
            # Narrower registers: packets always need fewer ancillas.
            assert packets.num_ladder <= base.num_ladder
    auto = QROM(data, address_bits, output_bits)
    if auto.variant == "select_copy":
        assert auto.alpha == 1


def test_auto_picks_the_brute_force_optimum_under_every_ancilla_budget():
    # Sweep the budget across every crossover: the auto choice must
    # match a brute-force enumeration over (variant, B, alpha) at every
    # budget, and packets must actually win somewhere (the paper's
    # constrained regime, coherently).
    rng = np.random.default_rng(3)
    data = [int(v) for v in rng.integers(0, 16, 1024)]
    address_bits, output_bits = 10, 4
    candidates = [QROM(data, address_bits, output_bits, variant="select")]
    for block_size in (2, 4, 8, 16, 32, 64, 128, 256, 512):
        candidates.append(
            QROM(data,
                 address_bits,
                 output_bits,
                 variant="select_swap",
                 block_size=block_size))
        for alpha in (1, 2, 3, 4):
            candidates.append(
                QROM(data,
                     address_bits,
                     output_bits,
                     variant="select_copy",
                     block_size=block_size,
                     alpha=alpha))
    packet_winner_budgets = []
    for budget in (10, 12, 14, 16, 20, 24, 32, 48, 64, 96, 200):
        fitting = [q for q in candidates if q.num_ladder <= budget]
        auto = QROM(data, address_bits, output_bits, max_ancillas=budget)
        assert auto.num_ladder <= budget
        best = min(q.toffoli_count for q in fitting)
        assert auto.toffoli_count == best
        if auto.variant == "select_copy" and auto.alpha > 1:
            packet_winner_budgets.append(budget)
    # The constrained regime exists: some budgets are won by alpha > 1.
    assert packet_winner_budgets
    # Pinned concrete crossover (N = 1024, b = 4): at 24 clean ancillas
    # alpha = 4, B = 16 (591 Toffolis) beats the best alpha = 1 fit
    # (772); unconstrained, alpha = 1 at B = 32 wins outright (253).
    tight = QROM(data, address_bits, output_bits, max_ancillas=24)
    assert (tight.variant, tight.block_size, tight.alpha) == ("select_copy",
                                                              16, 4)
    assert tight.toffoli_count == 591
    free = QROM(data, address_bits, output_bits)
    assert (free.variant, free.block_size, free.alpha) == ("select_copy", 32,
                                                           1)
    assert free.toffoli_count == 253


@needs_resources
def test_chain_fused_costs_m_plus_1_lookups_and_naive_2m():
    # Equal-length tables: the per-lookup price C is shape-only, so
    # fused == (m + 1) C, naive == 2 m C, margin exactly (m - 1) C —
    # and the compiler agrees with the bookkeeping for every step
    # kernel of both modes.
    rng = np.random.default_rng(9)
    m = 4
    tables = [[int(v) for v in rng.integers(0, 8, 16)] for _ in range(m)]
    per_lookup = QROM(tables[0], 4, 3, variant="select").toffoli_count
    fused = QROMChain(tables, 4, 3, fused=True, variant="select")
    naive = QROMChain(tables, 4, 3, fused=False, variant="select")
    assert fused.toffoli_count == (m + 1) * per_lookup
    assert naive.toffoli_count == 2 * m * per_lookup
    assert naive.toffoli_count - fused.toffoli_count == (m - 1) * per_lookup
    assert fused.toffoli_count < naive.toffoli_count
    for chain in (fused, naive):
        assert len(chain.step_toffoli_counts) == m + 1
        assert sum(chain.step_toffoli_counts) == chain.toffoli_count
        for kernel, claimed in zip(chain.kernels(), chain.step_toffoli_counts):
            compiled = _compiled_toffolis(kernel, chain.num_address,
                                          chain.num_ladder, chain.num_output)
            assert compiled == claimed


def test_chain_fused_strictly_cheaper_for_blocked_variants_too():
    # The (m + 1) vs 2 m load count is variant-independent: it also
    # holds when the steps are select_copy lookups (same forced shape).
    rng = np.random.default_rng(17)
    m = 3
    tables = [[int(v) for v in rng.integers(0, 8, 32)] for _ in range(m)]
    per_lookup = QROM(tables[0], 5, 3, variant="select_copy",
                      block_size=4).toffoli_count
    fused = QROMChain(tables,
                      5,
                      3,
                      fused=True,
                      variant="select_copy",
                      block_size=4)
    naive = QROMChain(tables,
                      5,
                      3,
                      fused=False,
                      variant="select_copy",
                      block_size=4)
    assert fused.toffoli_count == (m + 1) * per_lookup
    assert naive.toffoli_count == 2 * m * per_lookup


def test_chain_and_packet_validation_raises():
    with pytest.raises(ValueError, match="alpha .* only valid with"):
        QROM([1, 2], 1, 2, alpha=2)
    with pytest.raises(ValueError, match="alpha .* only valid with"):
        QROM([1, 2], 2, 2, variant="select_swap", block_size=2, alpha=2)
    with pytest.raises(ValueError, match="alpha must be an integer in"):
        QROM([1, 2], 1, 2, variant="select_copy", alpha=3)
    with pytest.raises(ValueError, match="alpha must be an integer in"):
        QROM([1, 2], 1, 2, variant="select_copy", alpha=0)
    with pytest.raises(ValueError, match="max_ancillas must be a positive"):
        QROM([1, 2], 1, 2, max_ancillas=0)
    with pytest.raises(ValueError, match="fits max_ancillas"):
        QROM([1] * 16, 4, 1, max_ancillas=1)
    with pytest.raises(ValueError, match="fits max_ancillas"):
        QROM([1] * 16, 4, 4, variant="select_swap", max_ancillas=5)
    with pytest.raises(ValueError, match="sequence of lookup tables"):
        QROMChain(7, 2, 2)
    with pytest.raises(ValueError, match="at least one table"):
        QROMChain([], 2, 2)
    with pytest.raises(ValueError, match=r"tables\[1\] must be a sequence"):
        QROMChain([[1, 2], 5], 2, 2)
    # Range errors are located: the offending step is named.
    with pytest.raises(ValueError, match="table 1 of 2.*does not fit"):
        QROMChain([[1, 0], [1, 4]], 1, 2, fused=False)
    with pytest.raises(ValueError, match="fused step.*non-negative"):
        QROMChain([[1, 0], [1, -2]], 1, 2, fused=True)
    with pytest.raises(ValueError, match="addresses only 4"):
        QROMChain([[1] * 5], 2, 2)


def test_chain_repr_and_describe_reflect_the_choice():
    tables = _chain_tables()
    chain = QROMChain(tables, 3, 3, fused=True)
    assert "fused=True" in repr(chain)
    text = chain.describe()
    assert "fused unload tables[0] + load tables[1]" in text
    assert text.count("QROM(") == chain.num_tables + 1
    naive = QROMChain(tables, 3, 3, fused=False)
    assert "two lookups replayed back to back" in naive.describe()
    packets = QROM([int(v) for v in np.arange(16) % 8],
                   4,
                   3,
                   variant="select_copy",
                   block_size=4,
                   alpha=3)
    assert "alpha=3" in repr(packets)
    assert "# Toffoli" in packets.describe()
