# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# [Begin Documentation]
"""Explicit (X-DF) and compressed (C-DF) double factorization of the two-electron
integrals, following Cohn, Motta, and Parrish, PRX Quantum 2, 040352 (2021).

Generates restricted Hartree-Fock molecular-orbital integrals for H2O/STO-3G with
PySCF, then double-factorizes the ERI tensor two ways:

  * X-DF: the exact factorization (rank-one cores) via pivoted Cholesky of
    the ERI supermatrix (the default), truncated by residual pivot.
  * C-DF: a least-squares-optimized factorization that reaches comparable
    accuracy with far fewer leaves, with and without RC-DF regularization,
    reported together with its convergence diagnostics: the outer
    optimizer's termination, the final inner core solve's verified residual
    against its target, and the in-loop solve summary. The reconstruction
    error and one-norm are labeled unconverged when the fit is.

Heavy linear algebra uses the NVIDIA math libraries (cuSOLVER/cuBLAS via CuPy)
when a GPU is available; otherwise it falls back to NumPy/SciPy.
"""
from __future__ import annotations

import numpy as np

import cudaq_algorithms as algorithms

df = algorithms.double_factorization


def water_mo_integrals():
    """RHF molecular-orbital integrals of H2O/STO-3G: the chemist-notation
    ERI tensor and the one-body (core Hamiltonian) matrix."""
    from pyscf import ao2mo, gto, scf
    mol = gto.M(atom="O 0 0 0; H 0 0 0.957; H 0 0.926 -0.24",
                basis="sto-3g",
                verbose=0)
    mf = scf.RHF(mol).run()
    n = mf.mo_coeff.shape[1]
    eri = np.asarray(ao2mo.restore("s1", ao2mo.kernel(mol, mf.mo_coeff), n))
    one_body = mf.mo_coeff.T @ mf.get_hcore() @ mf.mo_coeff
    return eri, one_body


def report(label, factorization, eri, one_body, regularization):
    """Print the fit together with the evidence that it converged.

    A small one-norm only means something for a converged fit: the outer
    optimizer must have stopped at a stationary point and the final inner
    core solve must have met its verified residual threshold. Both are
    checked from the public result fields, and an unconverged fit is
    labeled as such rather than presented as a validated estimate."""
    inner = factorization.inner_solve
    summary = factorization.inner_solve_summary
    rel = df.factorization_error(eri, factorization) / np.linalg.norm(eri)
    # The LCU one-norm takes the eigenvalues of the *centered* one-body
    # matrix of this factorization (bare one-body plus the exchange
    # correction of the factorized tensor and the leaf-centering remainder),
    # which depends on the fit, so it is recomputed per factorization.
    kappa = df.centered_one_body(one_body, factorization)
    one_norm = df.double_factorization_one_norm(factorization,
                                                np.linalg.eigvalsh(kappa),
                                                "lcu")
    status = "converged" if factorization.converged else "UNCONVERGED"
    print(f"  {label}: {status}")
    print(f"    outer L-BFGS-B : success={factorization.optimizer_success} "
          f"nit={factorization.optimizer_nit} "
          f"({factorization.optimizer_message})")
    print(f"    final inner    : {inner.solver}/{inner.preconditioner} "
          f"{inner.termination} after {inner.iterations} iterations, "
          f"verified residual {inner.residual_norm:.2e} "
          f"<= target {inner.threshold:.2e}: {inner.converged}")
    print(
        f"    in-loop inner  : {summary.num_solves - summary.num_unconverged}"
        f"/{summary.num_solves} solves met the loop tolerance, "
        f"{summary.total_iterations} CG iterations in total")
    print(f"    objective={factorization.objective:.3e}  "
          f"rel reconstruction error={rel:.3e}  one-norm (lcu)={one_norm:.4f}"
          f"{'' if factorization.converged else '  (unconverged fit)'}")
    if regularization == 0.0:
        # Without a ridge term the inner system is singular: the cores are
        # fixed only up to its null space, and the one-norm (unlike the
        # reconstruction) depends on which representative the solver
        # returns. Compare one-norms at rho > 0.
        print("    (rho = 0: one-norm is representation dependent)")
    return rel


def main():
    # "auto" is size-aware: it stays on NumPy for a problem this small and
    # moves to CuPy (cuSOLVER / cuBLAS) once the orbital count is large
    # enough to amortize the GPU launch overhead.
    backend = "auto"
    eri, one_body = water_mo_integrals()
    norm = np.linalg.norm(eri)
    print("Double factorization of H2O/STO-3G two-electron integrals")
    print(f"  backend: {backend} (GPU usable: {df.cupy_gpu_available()})")
    print(f"  orbitals: {eri.shape[0]}   ||eri||_F: {norm:.6f}")

    full = df.explicit_double_factorization(eri,
                                            threshold=0.0,
                                            backend=backend)
    print("\nX-DF (explicit):")
    print(f"  full-rank leaves: {full.num_leaves}   "
          f"reconstruction error: {df.factorization_error(eri, full):.3e}")
    for threshold in (1.0e-2, 1.0e-3, 1.0e-4):
        truncated = df.explicit_double_factorization(eri,
                                                     threshold=threshold,
                                                     backend=backend)
        rel = df.factorization_error(eri, truncated) / norm
        print(
            f"  threshold {threshold:.0e}: leaves={truncated.num_leaves:2d}  "
            f"rel error={rel:.3e}")

    # C-DF with the matrix-free CG inner solve (the path that scales to
    # large orbital counts). Solver settings are stated up front; every
    # inner solve is judged by its verified residual, and the result
    # carries the outcome. Truncated molecular C-DF can need a few
    # thousand L-BFGS steps, so the outer budget is raised above the
    # library default (2000); if it is still not enough the fit comes
    # back marked unconverged, and the report below says so.
    settings = dict(max_iterations=5000,
                    inner_solver="cg",
                    cg_tolerance=1.0e-10,
                    cg_preconditioner="pairwise",
                    backend=backend)
    print("\nC-DF (compressed) vs X-DF at equal leaf count:")
    print("  settings: " + ", ".join(f"{k}={v}" for k, v in settings.items()))
    for num_leaves in (2, 4):
        explicit = df.explicit_double_factorization(eri,
                                                    threshold=0.0,
                                                    max_num_leaves=num_leaves,
                                                    backend=backend)
        x_rel = df.factorization_error(eri, explicit) / norm
        print(f"\n  leaves={num_leaves}:  X-DF rel error={x_rel:.3e}")
        compressed = df.compressed_double_factorization(eri,
                                                        num_leaves=num_leaves,
                                                        **settings)
        c_rel = report("C-DF  (rho=0)", compressed, eri, one_body, 0.0)
        assert c_rel <= x_rel + 1.0e-6
        # RC-DF: the ridge term shrinks the cores (lower one-norm) and
        # conditions the inner system, at a small cost in reconstruction.
        regularized = df.compressed_double_factorization(eri,
                                                         num_leaves=num_leaves,
                                                         regularization=1.0e-3,
                                                         **settings)
        report("RC-DF (rho=1e-3)", regularized, eri, one_body, 1.0e-3)

    # The example asserts what it prints: the final inner solves met their
    # verified thresholds, so the reported residuals are trustworthy.
    assert compressed.inner_solve.converged
    assert regularized.inner_solve.converged


if __name__ == "__main__":
    main()
