# FTQC benchmarks

Three end-to-end benchmark applications that exercise the whole
cudaq-algorithms stack, each from frozen chemist integrals to a checked
physical answer. Together they touch every layer: the chemistry bridge,
the Jordan-Wigner fermion transform, the `PauliLCU` block encoding, all
three primitives (`Walk`, `QSVT`, `Trotter`), state preparation, and
both measurement conventions.

| benchmark | stack path | target | pass criterion |
| --- | --- | --- | --- |
| `bench_hamiltonian_simulation_qsvt.py` | integrals → JW → PauliLCU → QSPPACK phases → QSVT cos/sin → recombination | `exp(-iHt)\|HF>` | max amplitude error vs dense `expm` |
| `bench_hamiltonian_simulation_trotter.py` | integrals → JW → Trotter orders 1/2/4 | `exp(-iHt)\|HF>` | error tolerance + measured convergence slopes |
| `bench_quantum_exact_lanczos.py` | integrals → JW → PauliLCU → Walk Chebyshev moments → Krylov eigensolve | ground-state energy | `\|E_QEL − E_FCI\|` vs frozen FCI |

The two Hamiltonian-simulation benchmarks compute the *same* state by
completely independent circuit constructions, so they cross-validate
each other; QEL checks the spectral (moment/observable) path the other
two never touch.

## Running

```bash
export LD_LIBRARY_PATH=/usr/local/cudaq-v0.15/lib          # or your CUDA-Q
export PYTHONPATH=/usr/local/cudaq-v0.15:<repo>/python
cd ftqc-benchmarks
python3 bench_hamiltonian_simulation_qsvt.py      # needs qsppack + scipy
python3 bench_hamiltonian_simulation_trotter.py
python3 bench_quantum_exact_lanczos.py
```

Every script exits 0 on PASS and 1 on FAIL, so the suite doubles as a
stack-level integration gate. `CUDAQ_DEFAULT_SIMULATOR` overrides the
target (default `qpp-cpu`, fp64).

## Molecules

Molecules live in `_common.py` as frozen inline literals (chemist
`(pq|rs)` integrals, nuclear repulsion, FCI reference) — no data files,
no runtime pyscf dependency. Current registry:

- `h2` — H2/STO-3G at 0.7414 A: 4 qubits, 2 electrons, 15 Pauli terms.

Growing the suite = adding one `Molecule` literal (freeze the integrals
from a converged mean field the way
`tests/python/test_fermion_compilers.py` froze H2) and passing
`--molecule <name>`. Candidates in size order: H4 chain (8 qubits),
LiH (12 qubits), H2O (14 qubits).

## Reference results (H2, qpp-cpu fp64, defaults)

- QSVT (degree 16, t = 0.8, tau = alpha·t ≈ 2.15): max amplitude error
  8.5e-15, fidelity 1 − O(1e-15).
- Trotter (t = 0.8): order-4 error 6.8e-10 at 32 steps; measured slopes
  match orders 1/2/4 to ~1e-3.
- QEL (Krylov dimension 4, 8 moments): |E_QEL − E_FCI| ≈ 5e-15 Ha
  (the Krylov space contains the exact ground state for H2's
  2-dimensional HF-connected sector, so QEL is exact here, not merely
  converged).

## Notes

- The electronic Hamiltonian is built with `scalar_offset=0`; nuclear
  repulsion is added classically so the block-encoding `alpha` is not
  inflated by a constant.
- `Trotter` receives the Hamiltonian as a word dict
  (`_common.hamiltonian_dict`): under CUDA-Q v0.15 the SpinOperator
  ingestion path raises on identity terms (`term.max_degree` —
  "operator is not acting on any degrees"); the dict path handles the
  identity word and `sim.evolve` reintroduces its phase.
