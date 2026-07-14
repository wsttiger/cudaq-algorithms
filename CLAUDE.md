# cudaq-algorithms — working notes for coding agents (and new contributors)

Primitive-first FTQC algorithms library on CUDA-Q: block encodings
(`PauliLCU`), qubitization (`Walk`), `QSVT`, double factorization, and
fermion-to-qubit transforms. Python package in `python/cudaq_algorithms/`,
optional compiled extension in `lib/` + `python/bindings/`, tests in
`tests/python/`, runnable examples in `examples/`, docs in `docs/`.

Physics and numerical conventions (qubit ordering, integral tensors,
block-encoding normalization) live in `docs/conventions.md` — read it
before writing or validating any numerics. Most real bugs here have been
convention bugs.

## Build and test

Fast path — no build required. The compiled extension is optional by
design; the pure-Python modules import and work without it:

```bash
PYTHONPATH=python python3 -m pytest tests/python -q
```

Tests select their simulator via `conftest.py`: `qpp-cpu` by default
(fp64 — required for the 1e-10..1e-12 tolerances used throughout;
the default `nvidia` target is fp32 and will fail them). Override with
`CUDAQ_DEFAULT_SIMULATOR`.

With the compiled extension (needed only for the C++-backed APIs):

```bash
cmake -B build -DCUDAQ_DIR=<cudaq install>/lib/cmake/cudaq
cmake --build build -j
PYTHONPATH=build/python python3 -m pytest tests/python -q
```

The build copies `python/cudaq_algorithms/` into `build/python/`; after
editing Python sources, `rm -rf build/python/cudaq_algorithms` and
rebuild, or the stale copy is what you'll import.

## CUDA-Q kernel-language landmines

These are silent or misleading failure modes of `@cudaq.kernel` Python
kernels. The library's code shapes exist because of them — follow the
established patterns:

- **`return` inside a kernel is silently ignored**
  ([cuda-quantum#4845](https://github.com/NVIDIA/cuda-quantum/issues/4845)).
  Never use early-return guards; write positive `if` blocks. A "guarded"
  kernel that falls through can segfault or corrupt results silently.
- **Empty lists cannot cross the kernel boundary**
  ([cuda-quantum#4847](https://github.com/NVIDIA/cuda-quantum/issues/4847)).
  Flattened list arguments are padded with one dummy element when they
  could be empty (see `pauli_lcu.py`, `df_encoding.py`); the pad is never
  dereferenced because the matching count is 0.
- **`cudaq.adjoint` is broken** — it silently mis-replays loop-carried
  classical updates
  ([#4897](https://github.com/NVIDIA/cuda-quantum/issues/4897)) and
  rejects some valid unitary kernels
  ([#4898](https://github.com/NVIDIA/cuda-quantum/issues/4898)). Inverses
  are hand-written (e.g. `unprepare`) with tests pinning the inverse
  property. Do not introduce `cudaq.adjoint` calls.
- **`exp_pauli` does not accept individual qubit operands** — only
  runtime-contiguous slices (`qubits[a:a + 3]`). Design multi-qubit
  rotations so their support is contiguous.
- **Tuples cannot be closure-captured into kernels.** Unpack into scalar
  and list locals before defining the kernel (see the factory methods in
  `df_encoding.py`).
- **Kernels can call closure-captured kernels** with fixed, data-free
  signatures — this is the dependency-injection mechanism behind the
  `BlockEncoding` protocol and `state_prep` injection. Data crosses the
  boundary only at factory time, captured inside the returned kernel.
- **Kernel source must live in a real `.py` file** (CUDA-Q uses
  `inspect`); kernels cannot be defined in `exec`'d strings or REPL-only
  contexts.
- **Beware PEP 263**: a comment matching `coding: <name>` in a script's
  first two lines is parsed as an encoding declaration and breaks the
  file. Don't start scripts with comments like `# ForceEncoding: ...`.

## Validation culture

- Every quantum primitive is validated against an **independent dense
  reference** built in NumPy (dense Pauli/ladder matrices, exact
  eigendecompositions) — not against another circuit. See
  `tests/python/dense_references.py` and the ladder builders in
  `test_df_encoding.py` / `test_fermion_compilers.py`.
- Validation failures at the factory boundary must raise loudly
  (`ValueError` with a specific message); nothing may fail silently
  inside a circuit.
- New tests should run without the compiled extension when possible;
  extension-dependent tests use
  `pytest.mark.skipif(importlib.util.find_spec("cudaq_algorithms._pycudaq_algorithms") is None, ...)`.
- When touching anything numeric, prefer adding a small-system
  dense-reference test over a smoke test. Spectrum comparison
  (`np.linalg.eigvalsh` on `op.to_matrix()`) catches convention errors
  that circuit-runs-without-crashing tests cannot. Use `eigvalsh` only
  on operators you know are Hermitian.

## Formatting

```bash
bash scripts/run_yapf_format.sh   # yapf pinned to 0.43.0 in CI
```

**The script formats tracked files only.** Newly created (untracked)
files are skipped — run `yapf -i <file>` on them directly before
committing, or CI's formatting check will fail after the commit.
clang-format 18 governs C++ (`scripts/run_clang_format.sh`).

## API idioms

- Constructor holds the problem, method calls hold the choices:
  `Walk(encoding).kernel(power)`, `QSVT(encoding).kernel(sequence)`,
  `Trotter(H).kernel(time, steps, order)`.
- Consumers are generic over the encoding via the structural
  `BlockEncoding` protocol (`block_encoding.py`) — no inheritance;
  implement the documented factory surface and any encoding plugs into
  `Walk`/`QSVT`.
- Kernel factories return either `@cudaq.kernel(state)` circuits
  (simulation path) or, with `state_prep=`, zero-argument
  hardware-shaped circuits.
- Package root exports the object surface only; composable device
  kernels stay in module namespaces (`pauli_lcu.prepare`, ...) because
  their names are too generic to re-export.
