# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""Shared molecule data and harness helpers for the FTQC benchmarks.

The molecules are frozen inline (integrals as literals, no data files):
each entry pins the spatial chemist integrals, the nuclear repulsion,
and the FCI energy the benchmarks must reproduce. H2/STO-3G is the
starter; growing the suite means adding one ``Molecule`` literal here
(freeze the integrals from a converged mean field the way
``tests/python/test_fermion_compilers.py`` does) — the benchmark
scripts pick it up by name.

Everything quantum flows through the package under test:
``chemistry.qubit_hamiltonian`` (spin expansion + Jordan-Wigner),
``PauliLCU`` / ``Walk`` / ``QSVT`` / ``Trotter``. The only classical
reference is the dense matrix of the qubit Hamiltonian.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import cudaq
import numpy as np

from cudaq_algorithms import chemistry

# ----------------------------------------------------------------------------
# Frozen molecules (spatial chemist integrals; see module docstring)
# ----------------------------------------------------------------------------


@dataclass(frozen=True)
class Molecule:
    """A frozen benchmark molecule in its mean-field orbital basis."""

    name: str
    description: str
    num_electrons: int
    one_body_spatial: tuple  # (n, n) nested tuples, chemist convention
    eri_spatial: tuple  # (n, n, n, n) nested tuples, chemist convention
    nuclear_repulsion: float
    fci_energy: float  # frozen reference; benchmarks must reproduce it

    @property
    def num_spatial_orbitals(self) -> int:
        return len(self.one_body_spatial)

    @property
    def num_qubits(self) -> int:
        return 2 * self.num_spatial_orbitals


# Frozen from a converged pyscf RHF/FCI run (same literals as
# tests/python/test_fermion_compilers.py); H2 at 0.7414 A, STO-3G.
H2 = Molecule(
    name="h2",
    description="H2 / STO-3G at 0.7414 A (4 qubits, 2 electrons)",
    num_electrons=2,
    one_body_spatial=(
        (-1.2488468037963385, 8.867226166437188e-18),
        (9.4085812113517e-17, -0.4796778131338564),
    ),
    # Chemist (pq|rs) convention (what chemistry.qubit_hamiltonian takes);
    # sanity: (00|00)=0.6733, (11|11)=0.6963, (00|11)=0.6624, (01|01)=0.1816.
    eri_spatial=(
        (((0.6733439450064822, 0.0), (0.0, 0.6624272943269697)),
         ((-2.0816681711721685e-17, 0.18162533147656484),
          (0.18162533147656507, 8.326672684688674e-17))),
        (((2.0816681711721685e-17, 0.18162533147656484),
          (0.18162533147656523, -8.326672684688674e-17)),
         ((0.6624272943269696, 0.0), (-5.551115123125783e-17,
                                      0.6962915699872075))),
    ),
    nuclear_repulsion=0.7080240981000804,
    fci_energy=-1.1371757102406845,
)

MOLECULES = {H2.name: H2}

# ----------------------------------------------------------------------------
# Hamiltonian construction and dense references
# ----------------------------------------------------------------------------


def electronic_hamiltonian(molecule: Molecule):
    """Electronic ``cudaq.SpinOperator`` (no nuclear repulsion).

    The nuclear repulsion is kept classical and added to reported
    energies, so the block-encoding normalization ``alpha`` is not
    inflated by a constant shift.
    """
    return chemistry.qubit_hamiltonian(np.array(molecule.one_body_spatial),
                                       np.array(molecule.eri_spatial),
                                       scalar_offset=0.0,
                                       tolerance=1e-12)


def hamiltonian_dict(operator, num_qubits: int) -> dict:
    """``{pauli_word: real_coefficient}`` from a SpinOperator.

    Workaround for feeding ``Trotter`` a SpinOperator that contains an
    identity term: under CUDA-Q v0.15 the SpinOperator ingestion path
    trips on ``term.max_degree`` (raises "operator is not acting on any
    degrees" for identity terms), while the dict path handles the
    identity word fine.
    """
    terms: dict = {}
    for term in operator:
        word = str(term.get_pauli_word(num_qubits))
        coefficient = complex(term.evaluate_coefficient())
        terms[word] = terms.get(word, 0.0) + coefficient.real
    return terms


def dense_matrix(operator, num_qubits: int) -> np.ndarray:
    """Dense matrix of a SpinOperator (the classical reference)."""
    matrix = np.asarray(operator.to_matrix())
    expected = 1 << num_qubits
    if matrix.shape != (expected, expected):
        raise ValueError(f"operator acts on {matrix.shape}, "
                         f"expected {expected} (padding mismatch?)")
    return matrix


def hartree_fock_ket(molecule: Molecule) -> np.ndarray:
    """The HF determinant statevector (interleaved spin orbitals)."""
    ket = np.zeros(1 << molecule.num_qubits, dtype=np.complex128)
    ket[(1 << molecule.num_electrons) - 1] = 1.0
    return ket


def hartree_fock_prep(molecule: Molecule):
    """``(qubits: qview)`` kernel preparing the HF determinant.

    The injectable form of ``stateprep.hartree_fock`` for this
    molecule's electron count (closed shell, interleaved layout).
    """
    num_electrons = molecule.num_electrons

    @cudaq.kernel
    def prepare_hartree_fock(qubits: cudaq.qview):
        for index in range(num_electrons):
            x(qubits[index])

    return prepare_hartree_fock


# ----------------------------------------------------------------------------
# Reporting
# ----------------------------------------------------------------------------


class stopwatch:
    """Context manager: ``with stopwatch() as t: ...; t.seconds``."""

    def __enter__(self):
        self._start = time.perf_counter()
        return self

    def __exit__(self, *exc):
        self.seconds = time.perf_counter() - self._start
        return False


@dataclass
class Report:
    """Collects benchmark lines and enforces the pass criterion."""

    title: str
    lines: list = field(default_factory=list)
    failures: list = field(default_factory=list)

    def add(self, label: str, value) -> None:
        self.lines.append((label, value))

    def check(self, label: str, error: float, tolerance: float) -> None:
        status = "PASS" if error <= tolerance else "FAIL"
        self.add(label, f"{error:.3e} (tolerance {tolerance:.1e}) {status}")
        if error > tolerance:
            self.failures.append(label)

    def render(self) -> int:
        width = max(len(label) for label, _ in self.lines)
        print("=" * 72)
        print(self.title)
        print("=" * 72)
        for label, value in self.lines:
            print(f"  {label:<{width}}  {value}")
        verdict = "PASS" if not self.failures else "FAIL"
        print("-" * 72)
        print(f"  {verdict}: {self.title}")
        return 0 if not self.failures else 1
