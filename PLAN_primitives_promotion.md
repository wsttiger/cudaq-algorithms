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

## Chaining + bit packets record (branch `features/qrom_chain_bit_packets`)

Source: Motlagh & Pocrnic, "Halving the cost of QROM", arXiv:2605.20334,
Secs. II.C (sequential QROMs) and II.D (bit packets) — both re-derived
from the paper's text into this library's clean/coherent setting; **no
code implementation was consulted** (explicitly not PennyLane's, which
the paper says ships the construction).

**Sequential chaining (Sec. II.C) — `QROMChain`, the THC-facing
surface.** `QROMChain(tables, address_bits, output_bits, fused=True)`
mints `m + 1` step kernels for `m` tables sharing one address register,
each with the uniform QROM signature `(address, ladder, output)` and a
shared `num_ladder = max` ladder (wider views are safe — unused lines
untouched). Use pattern (the design commitment THC circuits consume):
`ks[0]; caller op 0; ks[1]; ...; ks[m-1]; caller op m-1; ks[m]` — after
`ks[j]` the output holds `tables[j][k]`, after `ks[m]` it is XOR-clean.
The chain cannot be one closed kernel (caller ops are arbitrary), so it
mints per-step kernels with the paper's XOR trick fused *into the
tables*: step 0 = `QROM(T_0)`, step `j` = `QROM(T_{j-1} XOR T_j)` (one
lookup does unload + load, since lookups are X-only XORs), step `m` =
`QROM(T_{m-1})`. `fused=False` is the behavioral reference: transition
kernels replay `QROM(T_{j-1})` then `QROM(T_j)` back-to-back in one
flat interpreter. Per-step lookups are independently auto-priced;
`variant`/`block_size`/`alpha`/`max_ancillas` forward. Table lengths
may differ (zero-padded XOR, consistent with out-of-range-reads-zero).

**Chain cost (coherent, pinned).** Every step is a full lookup at the
exact per-variant price `C`; equal-length tables give
`fused = (m + 1) C` vs `naive = 2 m C` — the paper's `(m + 1)`-vs-`2 m`
load count carries into the coherent setting *intact* (it is a count of
lookup flavors, not of measured loads); margin exactly `(m - 1) C`,
equal at `m = 1`. Compiler-pinned per step kernel in
`test_primitives_qrom_chain.py`. Deliberately NOT taken (documented in
`_qrom_chain.py`): a step's trailing register-unwrite walk commutes
with the caller's output-only op and could migrate into the next
kernel and fuse with its write walk — `m + 2` block walks instead of
`2 (m + 1)` for select_copy steps, saving one `W(A - log2 B, N/B)` per
boundary — at the price of step kernels that are no longer
self-contained QROMs. Revisit if THC's Toffoli budget demands it.

**Bit packets (Sec. II.D) — `alpha=` on `QROM`, select_copy only.**
The `b`-bit table is sliced into `alpha` balanced packets (widest
first; `alpha` need not divide `b`, no padding) written to disjoint
output slices, sharing one `(B - 1) * ceil(b/alpha)`-bit register
bank; adjacent unwrite/write block walks fuse (Sec. II.C applied
internally, no caller ops between packets). Derived coherent cost and
ancillas (compiler-pinned):

    packets(N, b, B, alpha) = (alpha + 1) W(A - log2 B, ceil(N/B))
                              + alpha W(log2 B, B) + b (B - 1)
    ancillas = max(A - log2 B, log2 B) + (B - 1) ceil(b / alpha)

`alpha = 1` reproduces select_copy exactly (same op stream). The copy
term is alpha-independent (`sum_s width_s = b`). Self-inverse: at
`alpha = 1` the global involution proof holds; at `alpha > 1` the copy
walks read register bits the transition walks rewrite, so only the
contractual clean-ladder-sector inverse is claimed (double application
= identity from |0> ancillas — pinned, not assumed).

**Reconciliation with the paper's halving claim.** At fixed `B` the
packet cost is strictly increasing in `alpha` (each increment adds one
block walk + one copy walk), so with UNCONSTRAINED ancillas
`alpha = 1` dominates pointwise and auto never picks packets — true in
the paper's accounting too (its claim is explicitly "when constrained
by the number of dirty qubits"). The halving lives at a FIXED ancilla
budget: `alpha`-times-narrower registers let `B` grow `alpha`-fold, so
at budget `~ b*lambda` the dominant term drops from `2 * (3/2) N /
lambda` (alpha = 1) to `(1 + 1/b) * (3/2) N / lambda` (alpha = b). The
coherent adaptation PRESERVES the halving headline exactly: the 3/2
coherent-walk overhead multiplies both sides and cancels in the ratio,
verified numerically at N = 2^26, matched budgets — cost ratio 0.7500 /
0.6252 / 0.5633 / 0.5344 at b = 2/4/8/16 vs the ideal (1 + 1/b)/2 =
0.75 / 0.625 / 0.5625 / 0.5312. In ABSOLUTE units our alpha = b
dominant term is `(3/2)(1 + 1/b) N/lambda`, not the paper's
`(1 + 1/b) N/lambda`: the measured-uncompute figure `1.25 N/lambda` at
`b = 4` (the probable literature source of quoted "1.25 N" numbers) is
coherently `1.875 N/lambda` here — the halving survives, the measured
prefactor does not. Subleading terms also match in shape: paper
`2(b+1)(b lambda - 2) ~ 2 b^2 lambda`, ours
`alpha W(log2 B, B) + b(B-1) ~ (5/2) b^2 lambda`; both need
`N >> b^2 lambda^2`.

**Auto pricing over the enlarged space.** `QROM` now enumerates
`(variant, block_size, alpha)` — alpha in `[1, b]` for select_copy —
and accepts `max_ancillas` (bound on `num_ladder`; candidates filtered,
loud ValueError when nothing fits). Ties: select, then select_copy over
select_swap, then smaller alpha, then smaller blocks. Pinned: budgeted
auto == brute-force enumeration at every budget across all crossovers
(e.g. N = 1024, b = 4: budget 24 -> select_copy B=16 alpha=4 at 591
Toffolis vs 772 best alpha=1 fit; unconstrained -> B=32 alpha=1 at
253). `repr` shows variant/block/alpha/ancillas/toffolis; new
`QROM.describe()` decodes the full instruction tape;
`QROMChain.describe()` labels each step's role.

**Cost table** (coherent Toffolis at unconstrained-optimal parameters;
packets never beat alpha = 1 unconstrained — shown to make that
honest):

    N      b   select   select_swap        select_copy(a=1)
    256    1   379      68  (B=16)         72   (B=16)
    256    4   379      142 (B=8)          117  (B=16)
    256    8   379      198 (B=8)          149  (B=8)
    1024   4   1531     302 (B=16)         253  (B=32)
    1024   16  1531     598 (B=8)          441  (B=16)
    4096   4   6139     622 (B=32)         525  (B=64)
    4096   8   6139     870 (B=32)         665  (B=32)
    4096   16  6139     1238 (B=16)        913  (B=32)
    16384  8   24571    1766 (B=64)        1353 (B=64)

    Budgeted (N = 4096, b = 8, A = 12), best (variant, B, alpha):
    budget 24 -> 3683 (B=16, a=8) | 32 -> 3129 (B=8, a=3)
    | 48 -> 2091 (B=16, a=4) | 64 -> 1693 (B=16, a=3)
    | 96 -> 1295 (B=16, a=2) | 128 -> 897 (B=16, a=1)
    | 256 -> 665 (B=32, a=1, the unconstrained optimum)

**Deferred (house rules — clean ancillas, no measurement):** dirty-
qubit borrowing and the measured "Restore" clean-up (paper Fig. 3 /
Appendix B) — the packet construction here uses clean registers and
unitary unwrite walks instead; the paper's caching register for the
re-writing chain variant (its Eq. (4) `(m + 2)` fix-up flavor) is moot
in the clean setting (every step restores its own ancillas); the
cross-boundary walk fusion noted above (`m + 2` block walks).
