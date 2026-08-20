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
