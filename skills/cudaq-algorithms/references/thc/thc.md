# THC pair preparation

## Selectable operation contracts

| Operation + object | Public entry point | Kind / layer | Capabilities | Record |
| --- | --- | --- | --- | --- |
| prepare / uniform valid triangular and sentinel pairs | `cudaq_algorithms.thc_prepare.THCPairDomain` | host + device kernels | inequalities, analytic amplitude amplification, clean scratch | [valid pair domain](thc-domain.md) |
| compute / reversible contiguous pair index | `cudaq_algorithms.thc_prepare.TriangularIndex` | host + device kernel | XOR arithmetic, clean scratch | [pair indexing](thc-index.md) |
| prepare / finite-bit signed pair distribution | `cudaq_algorithms.thc_prepare.THCPairPrepare` | host tables + device kernels | valid d-bin alias, retained histories, inverse | [pair alias](thc-pair-prepare.md) |

These are development APIs, not a packaged chemistry fitter or complete THC
block encoding. For arXiv:2011.03494 use the [fidelity checklist](thc-fidelity.md).

## Workflow

Choose an explicit one-body contraction policy and derive signed row weights.
Use the same pair indexing, diagonal multiplicities and one-body sentinel in
PREPARE and SELECT. The generic power-of-two alias sampler has a different
source-domain contract.

## Verification

`tests/python/test_thc_prepare.py` checks domain amplitudes, index XOR and alias
histories against independent formulas; application SELECT/walk validation is
still required. Source: `python/cudaq_algorithms/thc_prepare.py`.
