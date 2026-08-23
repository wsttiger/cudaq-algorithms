# cudaq_algorithms.primitives: unary iteration + QROM (P0 promotion, SOTA)

Branch: `features/primitives_qrom_unary_iteration` off main. One PR.
Origin: post-0.1.0 P0 roadmap. The experimental sparse branch's M2
implementations are the SEED (design patterns, tests, CUDA-Q findings);
this PR promotes them into a package-blessed `cudaq_algorithms.primitives`
subpackage AND upgrades both circuits to the accepted state of the art.

## Scope

IN: `primitives/__init__.py`, `_unary_iteration.py`, `_qrom.py`, tests
(correctness + compiler-pinned resource contracts via
`cudaq.estimate_resources`, skipif-gated). OUT: arithmetic (next P1 PR),
alias sampling (P1, follow-up), the sparse encodings (P2, stay
experimental), measurement-based uncomputation (see Decisions), dirty-
ancilla QROAM borrowing (documented follow-up).

## State of the art targets

1. **Unary iteration — Babbush et al. 2018 (arXiv:1805.03662, Fig. 7)**:
   walk cost **N − 1 Toffolis** (plus Cliffords) for N iterated
   addresses, by reusing the ladder/AND ancilla across sibling subtrees
   (the sibling transition is a CNOT-and-continue, not an
   uncompute-recompute). Current seed costs 2N − 4; the resource tests
   must pin the improved count exactly and keep the emitter-vs-compiler
   equality check. Keep: trace-time body emitter (factory-time callback
   -> flat interpreter kernel over captured int lists — the only
   cudaq.control-compatible composition mechanism), controlled variant
   (external one-qubit view roots the tree), explicit inverses, partial
   trees (num_items < 2^bits never entered), the extended body
   vocabulary from the seed (free_x/free_cx, AND-ladder gadgets, work
   view, sign leaf) so the sparse branch can rebase onto it.
2. **QROM — two variants behind ONE surface**:
   - `QROM(data, address_bits, output_bits)` unary-iteration baseline
     (Babbush 2018): ~N Toffolis, X-only writes (self-inverse — pin the
     property).
   - **SELECT-SWAP / QROAM (Low–Kliuchnikov–Schaeffer 2018,
     arXiv:1812.00954; Berry et al. 2019 usage)**: lookup
     `ceil(N/lambda) + b*(lambda-1)` Toffolis with `lambda` swap blocks
     of `b`-bit outputs; `lambda` defaults to the cost-optimal power of
     two (~sqrt(N/b)), overridable. Clean swap ancillas only (counted
     and documented; dirty-borrowing is a named follow-up). The swap
     network's inverse is explicit. Selection: a `variant=` parameter or
     auto-dispatch on a size threshold — pick one, document it, and make
     the resource report expose the chosen variant and its ancilla/
     Toffoli price so callers can compare.
3. Resource contracts (tests): emitter bookkeeping == compiler ccx count
   (`Resources.count_controls("x", 2)` — `count("ccx")` is a known trap
   returning 0); unary iteration == N − 1 exactly on full trees;
   QROAM lookup count matches the formula and beats plain QROM above the
   crossover; per-doubling increments for linearity (ratio approaches
   the asymptotic from above for affine costs — assert increments).

## Decisions (made; revisit only with Scott)

- **Unitary throughout**: no measurement-based uncomputation even though
  the literature's headline T-counts assume it — primitives must stay
  statevector-testable and inverse-composable (house rules: no
  cudaq.adjoint, hand-written inverses, kernels unitary). Document the
  gap honestly where costs are stated: cite the papers' counts AND ours.
- **Namespace**: `cudaq_algorithms.primitives`. Package root __init__
  export deferred until the P1s join (users import the subpackage).
- API may deviate from the sparse seed where SOTA demands (the sparse
  branch rebases and adapts — it is P2 and downstream).

## House rules (unchanged)

CLAUDE.md landmines (no early returns, empty-list padding, no tuple
capture, no mixed qview+qubit controls, kernels in real .py files);
keep-alive kernel registry with subpackage-unique names; loud
ValueError at every factory boundary; dense-reference correctness tests
(column extraction / basis readout) + derived tolerances; yapf 0.43.0;
DCO-signed commits, no AI attribution; full suite green before done.

Environment: pip cudaq 0.15.1, `PYTHONPATH=$PWD/python python3.12 -m
pytest tests/python -q` (box has GPUs; qpp-cpu via conftest default).

## SelectCopy record (branch `features/qrom_select_copy`)

Source: Motlagh & Pocrnic, "Halving the cost of QROM", arXiv:2605.20334,
Sec. II.B only — construction re-derived from the paper's text; **no code
implementation was consulted** (the paper notes it ships in PennyLane;
that implementation was deliberately not read).

**The adapted construction (clean, fully unitary).** The select_swap
middle — route block `r` to slot 0, copy slot 0 to the output, route
back — *is* one multiplexed copy `|r>|y>(x)_j|phi_j> ->
|r>|y XOR phi_r>(x)_j|phi_j>`. Implemented as `variant="select_copy"` on
the existing `QROM` surface (`(address, ladder, output)`, block
registers inside the ladder view): the block-index walk (Sel1) writes
entry `(q, 0)` straight into the output (leaf-controlled CNOTs, free)
and `data[q,r] XOR data[q,0]` into block register `r` for
`r = 1..B-1`; one unary-iteration walk over the low `log2(B)` address
bits Toffoli-copies register `r` into the output (empty body at
`r = 0`, opcode 21 `ccx(ladder, ladder -> target)` added to the
interpreter); a second block-index walk (Sel2) unwrites the registers
only. X-only sandwich `W1 C W2`: exactly self-inverse
(`W2 W1 = D` = the direct-write walk, `C` and `D` commute, all four are
involutions, `U^2 = W1 C D C W2 = W1 D W2 = I`) — pinned by tests.

**Derived Toffoli formula** (W = the fused coherent walk count,
`3B/2 - 5` full/uncontrolled, `B >= 8`):

    select_copy(N, b, B) = 2 W(A - log2(B), ceil(N/B))
                           + W(log2(B), B) + b (B - 1)

vs `select_swap = 2 W(A - log2(B), ceil(N/B)) + 2 b (B - 1)`. Ancillas:
`max(A - log2(B), log2(B))` shared walk lines + `(B - 1) b` block
registers (one register fewer than select_swap; the low walk reuses the
block walk's line pool). Verified: exhaustive classical truth-table
replay over 108 shapes, cudaq exhaustive readout, and
`cudaq.estimate_resources` ccx == formula (compiler-pinned in
`test_primitives_resources.py`).

**Reconciliation with the paper's half-cost claim.** The claim is about
the routing term: two multiplexed swaps cost `2 b (B - 1)` Toffolis
(+ a `4 b (B - 1)` CNOT storm); the one multiplexed copy costs
`b (B - 1)` Toffolis + its iterator, no routing CNOTs — the per-bit term
is exactly halved. In the paper's measured-uncompute accounting the
iterator is `B - 3`, so copy < 2 swaps for every `b >= 1`. In our
coherent accounting the iterator is `W(log2(B), B) = 3B/2 - 5`, so the
margin at equal blocks is `b (B - 1) - W(log2(B), B)`: positive for all
`b >= 2`, and for `b = 1` only below `B = 8` (tie at 8, negative above).
`auto` therefore prices all three variants: it picks select for small
tables/wide words, select_swap for the `b = 1` large-block corner, and
select_copy everywhere else (savings over select_swap grow to ~2x as
`b` grows; e.g. N=4096, b=16: 913 vs 1238 at optimal blocks).
Optimal-block heuristics: `B* ~ sqrt(3N/(2b))` (swap) vs
`B* ~ sqrt(6N/(2b + 3))` (copy) — the enumeration over powers of two
remains the source of truth.

**Deferred (out of scope on house rules — clean ancillas, no
measurement):**
- Dirty-ancilla borrowing and the measured Restore operation (paper
  Fig. 3/App. B) — the whole `2 N/B` dirty-sandwich machinery.
- Sec. II.C sequential-QROM fusion ((m+1) loads for m lookups): worth
  revisiting for THC/DF SELECTs even in the clean setting, where
  back-to-back lookups can share the unwrite walk.
- Sec. II.D bit packets (alpha sequential b/alpha-bit lookups): at
  `alpha = b` this gives the `(1 + 1/b) N/B` prefactor — the probable
  literature source of any quoted `1.25 N` figure (`b = 4`). Depends on
  the sequential-QROM fusion and the dirty-register accounting; a clean
  coherent analog would need its own cost derivation.
