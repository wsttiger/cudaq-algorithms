# cudaq-algorithms — QA wheel validation handoff

## What you are validating

One wheel: `cudaq_algorithms-<version>-py3-none-any.whl`. It is pure Python
and CUDA-agnostic — the same file installs everywhere. Its `cudaq`
dependency is a metapackage that selects the right `cuda-quantum-cuNN`
(cu12 or cu13) for the machine at install time, so there are no per-CUDA
wheel variants to track.

## What you need

1. **The wheel** (drop it in a directory; default expected location is
   `/root/wheels`).
2. **A checkout of the repository at the release tag.** Tests and examples
   are not packaged inside the wheel; the validation script must run from
   the repo root.
3. **conda** (miniconda is fine) and network access to pypi.org and
   pypi.nvidia.com.

## How to run

From the repository root:

```bash
bash scripts/validation/validate_wheels.sh --wheels-dir /path/to/wheels
```

That covers everything: for each Python version (3.11, 3.12, 3.13) it
creates a fresh conda environment, installs the wheel (resolving a released
cuda-quantum from pypi.nvidia.com), runs the full pytest suite once per
simulator, and executes every example. Failures are collected and
summarized at the end; exit code 0 means everything passed.

Useful options:

```bash
--python-versions "3.12"          # limit the Python loop (e.g. a quick leg)
--cuda-version 12.6.0             # conda CUDA toolkit version for GPU legs
--cudaq-wheels <dir> --cudaq-version 0.15.99
                                  # test against UNRELEASED cuda-quantum
                                  # wheels instead of the released ones
```

GPU coverage is automatic: if `nvidia-smi` works, the `nvidia-fp64`
simulator is added to the test matrix and the conda environment gets a CUDA
toolkit (the cu13 cuda-quantum wheel pulls CuPy, which needs one).

## What a passing run looks like

Reference numbers from the release-engineering dry run (x86_64 + GPU,
wheel built from main, released cuda-quantum-cu12 0.15.1, the full
default three-Python invocation):

- pytest: **278 passed, 3 skipped** on every leg — six suite runs total
  (Python 3.11/3.12/3.13, each on `qpp-cpu` and `nvidia-fp64`,
  ~3 minutes per run).
- All **8 examples** pass their self-checks on every Python (24 runs).
- Final line: `Validation completed successfully on <arch>!`, exit 0.
- The full three-Python run took roughly an hour including downloads.
- On Python 3.13 the suite additionally prints ~586 deprecation warnings
  from cuda-quantum itself (`ast_bridge.py: ast.Assign() ... will become
  an error in Python 3.15`) — upstream cuda-quantum, harmless today, not
  a cudaq-algorithms issue.

## Expected skips and exclusions (not bugs)

- **3 pytest skips**: the Psi4 integration tests skip when `psi4` is not
  installed. Psi4 is intentionally not part of the validation environment
  (it is not pip-installable in a portable way); the `from_psi4` path is
  covered by unit tests against recorded data elsewhere in the suite.
- **The fp32 `nvidia` target is deliberately never run against the test
  suite.** The suite validates against dense references at 1e-10..1e-12
  tolerances, which require fp64 arithmetic. On a GPU box, running pytest
  with the default `nvidia` target WILL fail with ~1e-6 discrepancies —
  that is a precision-budget mismatch by design, not a library bug. GPU
  coverage is via `nvidia-fp64`. (The same applies if you run examples
  manually on a GPU machine: use `CUDAQ_DEFAULT_SIMULATOR=nvidia-fp64` or
  `qpp-cpu`.)
- **Two examples need optional packages** the script installs for you
  (`pyscf`, `qsppack`). Without them those examples exit with a one-line
  "pip install ..." message rather than a traceback — by design.
- **Large example configurations skip circuit execution on purpose**: the
  chemistry demo's bigger molecules print the statevector cost
  (e.g. "26 system + 14 ancilla = 40 qubits") and run only the classical
  analysis. The printed explanation is the expected behavior.

## Known environment notes

- The script exports `CONDA_PLUGINS_AUTO_ACCEPT_TOS=true` because recent
  conda (≥ 24.x ToS plugin; verified on 26.5.3) otherwise refuses
  non-interactive channel use with `CondaToSNonInteractiveError`. This
  accepts Anaconda's channel Terms of Service on the runner's behalf —
  remove that line if your policy requires explicit acceptance.
- `OMP_NUM_THREADS=8` is set to keep OpenBLAS from oversubscribing.
- Wall-clock numbers above assume an otherwise idle machine; the CPU
  simulator parallelizes heavily and slows under contention.

## What to report

- The script's end-of-run summary (it lists every failure as
  `py<version>: <what>`), plus the full log.
- For pytest failures: the machine's CUDA driver/GPU model and which
  simulator leg failed — a failure on `nvidia-fp64` but not `qpp-cpu`
  is simulator-specific signal we especially want.
- Anything that failed at *install* time (resolver errors from the
  `cudaq` metapackage) — please include `pip debug --verbose`'s platform
  tags and the machine's CUDA version.

## Matrix we validated before handoff

| leg | status |
|---|---|
| x86_64 + GPU, py3.11 + 3.12 + 3.13, released cuda-quantum (PyPI mode) | full pass (dry run above) |
| {x86_64, arm64} × {CPU, GPU}, py3.11–3.13, wheel install + suite | green in CI (wheel test matrix) |
| Custom mode (unreleased cuda-quantum, `--no-deps` path) | green in CI |

The one axis QA adds beyond CI: released-artifact installs on your own
fleet's OS/driver variety, macOS if applicable, and the three-Python loop
on real machines.
