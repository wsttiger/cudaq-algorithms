# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Molecular reference cases for the double-factorization benchmarks.

Two families, each returned as ``ReferenceCase(eri, one_body, core_energy,
metadata)`` with ``eri`` the chemist-notation
``(pq|rs)`` tensor the ``double_factorization`` module consumes:

* Linear hydrogen chains H_N in STO-6G at 1.4 bohr spacing (and a stretched
  H10 at 2.8 bohr), RHF orbitals, N orbitals / N electrons. PySCF builds the
  mean field and writes an FCIDUMP (``H{N}_sto6g_d{d}.fcidump``); the
  integrals are then read back through ``chemistry.from_fcidump`` so the
  benchmark exercises the package's own reader. Files are cached in
  ``--cache-dir`` (default ``./df_reference_data``).
* FeMoco, Reiher et al., PNAS 114, 7555 (2017): the 54-orbital / 54-electron
  active-space integrals shipped with OpenFermion as
  ``resource_estimates/integrals/eri_reiher.h5`` (keys ``eri``, ``h0``,
  ``ecore``). Read directly with h5py; OpenFermion's
  ``load_casfile_to_pyscf`` is used as a cross-check when it is importable
  (it needs OpenFermion's resource-estimate extras, including jax).

    PYTHONPATH=python python3 benchmarks/double_factorization/reference_cases.py --build
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import pathlib
import platform
import sys
import time

import numpy as np

from cudaq_algorithms import chemistry

CHAIN_LENGTHS = (8, 10, 12, 16, 20)
CHAIN_SPACING_BOHR = 1.4
STRETCHED_SPACING_BOHR = 2.8
DEFAULT_CACHE_DIR = pathlib.Path("df_reference_data")


@dataclasses.dataclass
class ReferenceCase:
    name: str
    eri: np.ndarray
    one_body: np.ndarray
    core_energy: float
    metadata: dict

    @property
    def num_orbitals(self) -> int:
        return self.eri.shape[0]

    def centered_one_body_eigenvalues(self, factorization) -> np.ndarray:
        """Eigenvalues of the centered one-body matrix of ``factorization``
        (what ``double_factorization_one_norm`` expects); depends on the
        fit, so it is computed per factorization."""
        from cudaq_algorithms import double_factorization as df
        return np.linalg.eigvalsh(
            df.centered_one_body(self.one_body, factorization))


# --------------------------------------------------------------------------- #
# Hydrogen chains
# --------------------------------------------------------------------------- #


def chain_fcidump_name(n: int, d: float = CHAIN_SPACING_BOHR) -> str:
    return f"H{n}_sto6g_d{d}.fcidump"


def h_chain(n: int,
            d: float = CHAIN_SPACING_BOHR,
            cache_dir: pathlib.Path = DEFAULT_CACHE_DIR):
    """RHF/STO-6G linear H_n at spacing ``d`` bohr; writes the FCIDUMP into
    ``cache_dir`` and returns ``(mean_field, path)``."""
    from pyscf import gto, scf, tools
    mol = gto.M(atom=[('H', (0, 0, i * d)) for i in range(n)],
                basis='sto-6g',
                unit='Bohr',
                verbose=0)
    mf = scf.RHF(mol).run()
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / chain_fcidump_name(n, d)
    tools.fcidump.from_scf(mf, str(path))
    return mf, path


def hydrogen_chain(n: int,
                   d: float = CHAIN_SPACING_BOHR,
                   cache_dir: pathlib.Path = DEFAULT_CACHE_DIR,
                   rebuild: bool = False) -> ReferenceCase:
    """Load (building and caching if needed) the H_n/STO-6G reference case."""
    import pyscf
    path = cache_dir / chain_fcidump_name(n, d)
    metadata = {
        "family": "hydrogen_chain",
        "num_atoms": n,
        "spacing_bohr": d,
        "basis": "sto-6g",
        "orbitals": "RHF canonical",
        "pyscf": pyscf.__version__,
        "fcidump": str(path),
    }
    if rebuild or not path.exists():
        start = time.perf_counter()
        mf, path = h_chain(n, d, cache_dir)
        metadata.update({
            "scf_converged": bool(mf.converged),
            "rhf_energy": float(mf.e_tot),
            "build_seconds": time.perf_counter() - start,
        })
        (path.with_suffix(".json")).write_text(json.dumps(metadata, indent=1))
    elif path.with_suffix(".json").exists():
        metadata.update(json.loads(path.with_suffix(".json").read_text()))
    one_body, eri, core = chemistry.from_fcidump(path.read_text())
    metadata["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()[:16]
    return ReferenceCase(
        f"H{n}-sto6g" + ("-stretched" if d != CHAIN_SPACING_BOHR else ""), eri,
        one_body, core, metadata)


# --------------------------------------------------------------------------- #
# FeMoco (Reiher)
# --------------------------------------------------------------------------- #


def reiher_h5_path() -> pathlib.Path:
    import openfermion
    return (pathlib.Path(openfermion.__file__).parent / "resource_estimates" /
            "integrals" / "eri_reiher.h5")


def femoco_reiher(cross_check: bool = True) -> ReferenceCase:
    """The Reiher FeMoco CAS(54e, 54o) integrals, read from OpenFermion's
    ``eri_reiher.h5``. ``h0`` is the one-body (core) Hamiltonian in the
    active space and ``ecore`` the constant (frozen core + nuclear) energy."""
    import h5py
    import openfermion
    path = reiher_h5_path()
    with h5py.File(path, "r") as f:
        eri = np.asarray(f["eri"][()], dtype=float)
        one_body = np.asarray(f["h0"][()], dtype=float)
        core = float(f["ecore"][()])
    metadata = {
        "family": "femoco",
        "reference": "Reiher et al., PNAS 114, 7555 (2017)",
        "active_space": "54 electrons (27 alpha, 27 beta) in 54 orbitals",
        "openfermion": openfermion.__version__,
        "file": str(path),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest()[:16],
        "cross_checked_with_openfermion_loader": False,
    }
    if cross_check:
        try:
            from openfermion.resource_estimates.molecule import (
                load_casfile_to_pyscf)
        except ImportError:  # resource-estimate extras (jax) absent
            load_casfile_to_pyscf = None
        if load_casfile_to_pyscf is not None:
            _, mf = load_casfile_to_pyscf(str(path), num_alpha=27, num_beta=27)
            from pyscf import ao2mo
            n = one_body.shape[0]
            eri_of = np.asarray(ao2mo.restore("s1", mf._eri, n))
            if not (np.allclose(eri_of, eri, atol=1.0e-12)
                    and np.allclose(mf.get_hcore(), one_body, atol=1.0e-12)):
                raise RuntimeError(
                    "eri_reiher.h5 disagrees with OpenFermion's "
                    "load_casfile_to_pyscf")
            metadata["cross_checked_with_openfermion_loader"] = True
    return ReferenceCase("FeMoco-reiher", eri, one_body, core, metadata)


# --------------------------------------------------------------------------- #

# name -> (loader(cache_dir), proposed leaf counts). Leaf counts span
# compressed to overcomplete (n/2, n, 2n); FeMoco's bracket the DF ranks
# used for this Hamiltonian in the literature.
REFERENCE_CASES = {
    **{
        f"H{n}-sto6g": ((lambda cache_dir, n=n: hydrogen_chain(
            n, CHAIN_SPACING_BOHR, cache_dir)), [n // 2, n, 2 * n])
        for n in CHAIN_LENGTHS
    },
    "H10-sto6g-stretched":
    ((lambda cache_dir: hydrogen_chain(10, STRETCHED_SPACING_BOHR, cache_dir)),
     [5, 10, 20]),
    "FeMoco-reiher": ((lambda cache_dir: femoco_reiher()), [54, 108, 216]),
}


def validate(case: ReferenceCase) -> dict:
    """Convention checks: 8-fold symmetry and a positive semidefinite ERI
    supermatrix (chemist notation). Returns a small report."""
    eri = case.eri
    n = case.num_orbitals
    sym = max(float(np.abs(eri - eri.transpose(1, 0, 2, 3)).max()),
              float(np.abs(eri - eri.transpose(0, 1, 3, 2)).max()),
              float(np.abs(eri - eri.transpose(2, 3, 0, 1)).max()))
    supermatrix = eri.reshape(n * n, n * n)
    eigenvalues = np.linalg.eigvalsh(0.5 * (supermatrix + supermatrix.T))
    return {
        "n": n,
        "eri_norm": float(np.linalg.norm(eri)),
        "max_symmetry_violation": sym,
        "supermatrix_min_eigenvalue": float(eigenvalues.min()),
        "supermatrix_rank_1e-8": int((eigenvalues > 1.0e-8).sum()),
        "supermatrix_dim": n * n,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build",
                        action="store_true",
                        help="build/cache every case and print a report")
    parser.add_argument("--cases", default=",".join(REFERENCE_CASES))
    parser.add_argument("--cache-dir", default=str(DEFAULT_CACHE_DIR))
    args = parser.parse_args()
    cache_dir = pathlib.Path(args.cache_dir)
    print(f"python {platform.python_version()}  numpy {np.__version__}  "
          f"cache {cache_dir.resolve()}")
    for name in args.cases.split(","):
        loader, leaf_counts = REFERENCE_CASES[name.strip()]
        start = time.perf_counter()
        try:
            case = loader(cache_dir)
        except ImportError as error:
            print(f"{name}: SKIPPED ({error})")
            continue
        elapsed = time.perf_counter() - start
        report = validate(case)
        print(
            f"{case.name}: n={report['n']} leaves={leaf_counts} "
            f"||eri||_F={report['eri_norm']:.4f} "
            f"sym={report['max_symmetry_violation']:.1e} "
            f"min eig={report['supermatrix_min_eigenvalue']:.1e} "
            f"rank={report['supermatrix_rank_1e-8']}/{report['supermatrix_dim']} "
            f"core={case.core_energy:.6f} ({elapsed:.1f} s)")
        if args.build:
            print("   " + json.dumps(case.metadata))


if __name__ == "__main__":
    main()
