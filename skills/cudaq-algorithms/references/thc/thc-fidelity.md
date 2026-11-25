# Fidelity checklist for arXiv:2011.03494v3

Scope: main nonorthogonal THC construction, Sections II-III and Appendices F-G;
honor the user's actual exclusions. Apply the [paper workflow](../paper-implementation.md).

## Workflow

| Paper construction | Reuse / implementation boundary | Independent acceptance |
| --- | --- | --- |
| II, THC representation and modified one-body operator | application fitter/contractions; retain DF, additional THC, total errors, shift and lambda | conventional CAR Hamiltonian, approximate factor case, signed overlapping factors and gauge normalization |
| III B, Figs 3-4, valid pairs / alias | [THCPairDomain](thc-domain.md), [index](thc-index.md), [pair PREPARE](thc-pair-prepare.md) | amplitudes/history, sentinel, signs, nonpower domain, inverse and clean scratch |
| III C, data lookup, Apps F-G | [QROM](../primitives/primitives-qrom.md), unary iteration; explicit variants/block sizes | coherent address tests, padding and compiler-priced compute/erase |
| III C, programmable Givens | existing [planning](../basis-change/basis-change-plan.md) and [word consumer](../basis-change/basis-change-programmable.md) | changing quantum words with fixed host schedule changes full-Fock operation; three-orbital determinant reference |
| III C, phase-gradient rotations | [arithmetic backend](../basis-change/basis-change-phase-gradient.md) and catalyst | rounded-target joint state, phases, restoration, bit-width sweep and actual cost |
| III C, self-inverse SELECT and signed walk | application half-SELECT/order construction | noncommuting directions, both spins, one-body identity, controlled relative phases and multiple signed energies |
| III A, Figs 1-2, optimized phase control | [unary schedule](../qubitization/qubitization-unary-phase-estimation.md), distinct from [textbook QPE](../qubitization/qubitization-phase-estimation.md) | derive selected powers, sine window, query count, padding and decoder; compare independent Fourier distribution |
| Apps F-G, erasure | [measured QROM eraser](../primitives/primitives-qrom-erasure.md); coherent inverse reference | every small measurement branch, entangled reference and actual application use |

## Verification

Record source revision, commands, numerical tolerances and precision/cost limits
for every row. Preserve the baseline H2 path; separately label full paper H2 as
constructed, compiled or executed. A feasible restricted-sector demonstration
is not a full molecular validation. The current CDKM phase backend, coherent
unary ladder, analytic amplification rotation and sine-tree preparation do not
establish all paper headline fault-tolerant prefactors. Do not report exact
paper resource reproduction until their decomposition and synthesis are priced.
