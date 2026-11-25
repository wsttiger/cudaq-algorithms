# Implementing a paper with explicit fidelity requirements

## Workflow

Establish the paper version, requested sections, exclusions, checkout revision,
runtime import path and execution target. Before changing code, keep one row per
requested construction: equation/figure, input/output contract, current public
API, needed change, independent test, precision budget and resource assumptions.
Inspect existing APIs and their composition tests before declaring a gap.

Implement in dependency order. Host compilation of tables is appropriate;
executed circuits must consume the selected quantum data. Keep numerical
references independent of the scheduler, lookup and rotation code being tested.
Check joint outputs and promised clean workspace, including coherent controls
and physically consequential phases. An inverse roundtrip alone does not
establish the forward operation. Resource checks must expose a bypassed lookup,
unused word, absent arithmetic or different query schedule.

Keep these classifications separate from scientific correctness:

| Classification | Evidence needed |
| --- | --- |
| paper construction | requested mechanism implemented and independently executed at a stated scale |
| demonstrated structural equivalent | same relevant mechanism with derived precision and resource behavior; identify deviations |
| functional substitute | correct mathematical action through a materially different construction |
| partial | implemented portion and exact unverified/missing portion identified |
| missing | no implementation of the requested mechanism |

## Verification

Record expected and observed results, exact commands, tolerances, input hashes,
precision settings and failures. Distinguish constructed, frontend compiled and
executed. Preflight memory, query count and depth; a simulator limit may justify
a smaller independent component test, but does not turn it into a molecular run.
Keep baseline examples working. Preserve original fixtures and evaluation files.

For each approximation separate model/factorization error, coefficient and angle
rounding, phase resolution, shots and initial overlap. Exact energies may check
an estimate but must not choose its bin. Count actual lookup, erase, arithmetic,
controls and scratch independently; preparation and rotation synthesis are
explicit costs. A quoted paper formula, simulator multi-control instruction or
passing projected block is insufficient evidence of a fault-tolerant price.

Finish with the construction table and remaining limitations. Re-score only
against the original rubric. Skill routing checks establish navigation, not
scientific validation or behavioral uplift; those need separate executed tests.

THC-specific contracts and acceptance checks are in the
[THC fidelity checklist](thc/thc-fidelity.md).
