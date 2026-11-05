# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Reliability and timing of the C-DF inner core solve and of complete fits.

Two experiment families, kept separate so inner-solver behavior is not
confused with changes in the outer optimization path:

  --inner   fixed-rotation inner solves ``(A + 2 rho I) Z = R`` at the X-DF
            warm-start rotations: plain CG, pairwise-preconditioned CG, and
            (where the design matrix fits) the explicit lstsq reference, each
            cold and warm started, at several regularization strengths.
  --fits    complete C-DF / RC-DF fits with the same inner configurations.

Every solve is judged by the same verified criterion, the true residual
``||R - (A + 2 rho I) Z||_F <= tol * ||R||_F`` recomputed from the operator,
and every timing is the complete call: operator and preconditioner setup,
host-to-device transfers, and a device synchronization at the end, after one
untimed warm-up call on the backend. The hardware and library versions are
recorded in the output. CuPy is run first when a GPU is usable; small NumPy
and lstsq cases remain as references.

Workloads (``--cases``): synthetic exact-rank and compressed ERIs; hydrogen
chains H4..H10 and H2O in STO-3G, 6-31G and 6-311G from PySCF when it is
installed. All arithmetic is float64.

    PYTHONPATH=python python3 \\
        benchmarks/double_factorization/bench_inner_solve_reliability.py \\
        --inner --fits --json results.jsonl
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
import warnings

import numpy as np
import scipy

import cudaq_algorithms as algorithms

df = algorithms.double_factorization
from cudaq_algorithms.double_factorization import _factorization as F
from cudaq_algorithms.double_factorization._backend import (resolve_backend,
                                                            to_device)

# --------------------------------------------------------------------------- #
# Environment
# --------------------------------------------------------------------------- #


def _sync(xp):
    if xp.__name__ == "cupy":
        xp.cuda.runtime.deviceSynchronize()


def environment():
    info = {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "omp_num_threads": os.environ.get("OMP_NUM_THREADS", "(unset)"),
        "cpu": platform.processor() or platform.machine(),
        "cupy": None,
        "cuda_runtime": None,
        "cuda_driver": None,
        "gpu": None,
    }
    if df.cupy_gpu_available():
        import cupy
        info["cupy"] = cupy.__version__
        info["cuda_runtime"] = cupy.cuda.runtime.runtimeGetVersion()
        info["cuda_driver"] = cupy.cuda.runtime.driverGetVersion()
        props = cupy.cuda.runtime.getDeviceProperties(
            cupy.cuda.runtime.getDevice())
        info["gpu"] = props["name"].decode() if isinstance(
            props["name"], bytes) else str(props["name"])
    return info


def backends(requested):
    available = ["cupy", "numpy"] if df.cupy_gpu_available() else ["numpy"]
    if requested == "all":
        return available
    chosen = [b.strip() for b in requested.split(",")]
    missing = [b for b in chosen if b not in available]
    if missing:
        raise SystemExit(f"backend(s) {missing} not usable here")
    return chosen


# --------------------------------------------------------------------------- #
# Workloads
# --------------------------------------------------------------------------- #


def synthetic_eri(n, num_vectors, seed):
    rng = np.random.default_rng(seed)
    leaves = rng.standard_normal((num_vectors, n, n))
    leaves = 0.5 * (leaves + leaves.transpose(0, 2, 1))
    return np.einsum("xpq,xrs->pqrs", leaves, leaves)


def molecule(atom, basis):
    from pyscf import ao2mo, gto, scf
    mol = gto.M(atom=atom, basis=basis, verbose=0)
    mf = scf.RHF(mol).run()
    n = mf.mo_coeff.shape[1]
    eri = np.asarray(ao2mo.restore("s1", ao2mo.kernel(mol, mf.mo_coeff), n))
    one_body = mf.mo_coeff.T @ mf.get_hcore() @ mf.mo_coeff
    return eri, np.linalg.eigvalsh(one_body)


def hydrogen_chain(count, basis, spacing=1.0):
    atom = "; ".join(f"H 0 0 {spacing * i:.3f}" for i in range(count))
    return molecule(atom, basis)


_WATER = "O 0 0 0; H 0 0 0.957; H 0 0.926 -0.24"

# name -> (loader, leaf counts). Leaf counts below the X-DF rank are genuinely
# compressed fits; the synthetic exact-rank case is the only exact one.
CASES = {
    "synthetic-exact-n4": (lambda:
                           (synthetic_eri(4, 3, 11), np.zeros(4)), [3]),
    "synthetic-compressed-n8": (lambda:
                                (synthetic_eri(8, 6, 5), np.zeros(8)), [3, 6]),
    "H4-sto3g": (lambda: hydrogen_chain(4, "sto-3g"), [2, 4]),
    "H6-sto3g": (lambda: hydrogen_chain(6, "sto-3g"), [3, 6]),
    "H8-sto3g": (lambda: hydrogen_chain(8, "sto-3g"), [4, 8]),
    "H10-sto3g": (lambda: hydrogen_chain(10, "sto-3g"), [5, 10]),
    "H2O-sto3g": (lambda: molecule(_WATER, "sto-3g"), [4, 7]),
    "H2O-631g": (lambda: molecule(_WATER, "6-31g"), [8, 13]),
    "H2O-6311g": (lambda: molecule(_WATER, "6-311g"), [20]),
}
DEFAULT_CASES = ("synthetic-exact-n4,synthetic-compressed-n8,H4-sto3g,"
                 "H6-sto3g,H8-sto3g,H2O-sto3g,H2O-631g")


def load_case(name):
    loader, leaf_counts = CASES[name]
    try:
        eri, one_body_eigenvalues = loader()
    except ImportError as error:
        print(f"  {name}: SKIPPED ({error})")
        return None
    return eri, one_body_eigenvalues, leaf_counts


# --------------------------------------------------------------------------- #
# Inner-solve experiments
# --------------------------------------------------------------------------- #

INNER_CONFIGS = [("cg", "none"), ("cg", "pairwise"), ("lstsq", "none")]


def xdf_rotations(eri, num_leaves):
    xdf = df.explicit_double_factorization(eri,
                                           threshold=0.0,
                                           max_num_leaves=num_leaves,
                                           backend="numpy")
    rotations = [np.asarray(r, dtype=float) for r in xdf.leaf_rotations]
    n = eri.shape[0]
    while len(rotations) < num_leaves:
        rotations.append(np.eye(n))
    return rotations[:num_leaves]


def perturbed_rotations(rotations, seed, scale=1.0e-2):
    """Rotations a short L-BFGS step away: ``U exp(delta)`` with a small
    random antisymmetric ``delta``. The cores at ``rotations`` then serve as
    a realistic warm start for the solve at these."""
    rng = np.random.default_rng(seed)
    out = []
    for u in rotations:
        delta = rng.standard_normal(u.shape) * scale
        delta = delta - delta.T
        out.append(u @ scipy.linalg.expm(delta))
    return out


def timed_inner_solve(eri, rotations, xp, solver, preconditioner, rho, tol,
                      guess, max_iterations):
    """Complete solve time: transfers + operator/preconditioner setup +
    iterations + final verification + device sync."""
    _sync(xp)
    start = time.perf_counter()
    eri_dev = to_device(eri, xp)
    rotations_dev = [to_device(u, xp) for u in rotations]
    guess_dev = None if guess is None else to_device(guess, xp)
    if solver == "lstsq":
        cores, diag = F._solve_inner_cores_lstsq(eri_dev, rotations_dev, xp,
                                                 rho, tol)
    else:
        cores, diag = F._solve_inner_cores_cg(eri_dev,
                                              rotations_dev,
                                              xp,
                                              rho,
                                              tol,
                                              max_iterations,
                                              guess_dev,
                                              preconditioner=preconditioner)
    _sync(xp)
    total = time.perf_counter() - start
    return cores, diag, total


def timed_setup(eri, rotations, xp, preconditioner, rho):
    """Operator (+ preconditioner) construction alone, transfers included."""
    _sync(xp)
    start = time.perf_counter()
    eri_dev = to_device(eri, xp)
    rotations_dev = [to_device(u, xp) for u in rotations]
    op = F._InnerOperator(eri_dev, rotations_dev, xp, rho)
    if preconditioner == "pairwise":
        F._PairwisePreconditioner(op)
    _sync(xp)
    return time.perf_counter() - start


def run_inner(case_name, eri, leaf_counts, backend_names, rhos, tol, repeats,
              max_iterations, lstsq_limit_bytes, emit):
    n = eri.shape[0]
    print(f"\n### inner solves: {case_name}  n={n}  tol={tol:g}")
    print(f"  {'backend':7} {'T':>3} {'rho':>7} {'start':5} {'solver':6} "
          f"{'precond':8} | {'conv':4} {'term':16} {'iters':>5} "
          f"{'resid/thr':>9} | {'setup ms':>9} {'total ms':>9}")
    for num_leaves in leaf_counts:
        rotations = xdf_rotations(eri, num_leaves)
        design_bytes = F.lstsq_design_bytes(n, num_leaves)
        for backend in backend_names:
            xp, _ = resolve_backend(backend)
            for rho in rhos:
                # Warm start: the plain-CG solution at nearby rotations.
                near = perturbed_rotations(rotations, seed=3)
                warm_cores, _, _ = timed_inner_solve(eri, near, xp, "cg",
                                                     "none", rho, tol, None,
                                                     max_iterations)
                warm = np.stack([
                    np.asarray(xp.asnumpy(c))
                    if backend == "cupy" else np.asarray(c) for c in warm_cores
                ])
                for solver, preconditioner in INNER_CONFIGS:
                    if solver == "lstsq" and design_bytes > lstsq_limit_bytes:
                        continue
                    starts = [("cold", None)]
                    if solver == "cg":
                        starts.append(("warm", warm))
                    for start_name, guess in starts:
                        # Untimed warm-up on this backend/config.
                        timed_inner_solve(eri, rotations, xp, solver,
                                          preconditioner, rho, tol, guess,
                                          max_iterations)
                        best_total, best_setup, diag = np.inf, np.inf, None
                        for _ in range(repeats):
                            _, diag, total = timed_inner_solve(
                                eri, rotations, xp, solver, preconditioner,
                                rho, tol, guess, max_iterations)
                            best_total = min(best_total, total)
                            best_setup = min(
                                best_setup,
                                timed_setup(eri, rotations, xp, preconditioner,
                                            rho))
                        ratio = (diag.residual_norm /
                                 diag.threshold if diag.threshold > 0 else 0.0)
                        print(f"  {backend:7} {num_leaves:>3} {rho:>7g} "
                              f"{start_name:5} {solver:6} {preconditioner:8} "
                              f"| {str(diag.converged):4} "
                              f"{diag.termination:16} {diag.iterations:>5} "
                              f"{ratio:>9.2e} | {best_setup*1e3:>9.2f} "
                              f"{best_total*1e3:>9.2f}")
                        emit({
                            "experiment": "inner",
                            "case": case_name,
                            "n": n,
                            "num_leaves": num_leaves,
                            "backend": backend,
                            "rho": rho,
                            "start": start_name,
                            "solver": solver,
                            "preconditioner": preconditioner,
                            "tolerance": tol,
                            "max_iterations": max_iterations,
                            "converged": diag.converged,
                            "termination": diag.termination,
                            "iterations": diag.iterations,
                            "residual_norm": diag.residual_norm,
                            "threshold": diag.threshold,
                            "rhs_norm": diag.rhs_norm,
                            "setup_s": best_setup,
                            "total_s": best_total,
                        })


# --------------------------------------------------------------------------- #
# Complete fits
# --------------------------------------------------------------------------- #

FIT_CONFIGS = [("cg", "none", "none"), ("cg", "pairwise", "none"),
               ("cg", "none", "lstsq"), ("lstsq", "none", "none")]


def run_fits(case_name, eri, one_body_eigenvalues, leaf_counts, backend_names,
             rhos, tol, max_iterations, cg_max_iterations, lstsq_limit_bytes,
             emit):
    n = eri.shape[0]
    norm = float(np.linalg.norm(eri))
    print(f"\n### complete fits: {case_name}  n={n}  max_iterations="
          f"{max_iterations}  cg_tolerance={tol:g}")
    print(f"  {'backend':7} {'T':>3} {'rho':>7} {'solver':6} {'precond':8} "
          f"{'fallbk':6} | {'outer':5} {'nit':>5} {'final':5} {'in-loop':>14} "
          f"| {'objective':>11} {'rel err':>9} {'lambda':>8} {'time s':>8}")
    for num_leaves in leaf_counts:
        design_bytes = F.lstsq_design_bytes(n, num_leaves)
        for backend in backend_names:
            xp, _ = resolve_backend(backend)
            for rho in rhos:
                for solver, preconditioner, fallback in FIT_CONFIGS:
                    if solver == "lstsq" and design_bytes > lstsq_limit_bytes:
                        continue
                    kwargs = dict(num_leaves=num_leaves,
                                  max_iterations=max_iterations,
                                  regularization=rho,
                                  inner_solver=solver,
                                  cg_tolerance=tol,
                                  cg_max_iterations=cg_max_iterations,
                                  cg_preconditioner=preconditioner,
                                  inner_fallback=fallback,
                                  backend=backend)
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", RuntimeWarning)
                        # Untimed warm-up (tiny budget) on this backend.
                        df.compressed_double_factorization(
                            eri, **{
                                **kwargs, "max_iterations": 1
                            })
                        _sync(xp)
                        start = time.perf_counter()
                        fit = df.compressed_double_factorization(eri, **kwargs)
                        _sync(xp)
                        elapsed = time.perf_counter() - start
                    rel_error = df.factorization_error(eri, fit) / norm
                    lam = df.double_factorization_one_norm(
                        fit, one_body_eigenvalues, "lcu")
                    summary = fit.inner_solve_summary
                    inner = fit.inner_solve
                    loop = (f"{summary.num_solves - summary.num_unconverged}"
                            f"/{summary.num_solves} ok")
                    print(f"  {backend:7} {num_leaves:>3} {rho:>7g} "
                          f"{solver:6} {preconditioner:8} {fallback:6} | "
                          f"{str(fit.optimizer_success):5} "
                          f"{fit.optimizer_nit:>5} {str(inner.converged):5} "
                          f"{loop:>14} | {fit.objective:>11.4e} "
                          f"{rel_error:>9.2e} {lam:>8.3f} {elapsed:>8.2f}")
                    emit({
                        "experiment": "fit",
                        "case": case_name,
                        "n": n,
                        "num_leaves": num_leaves,
                        "backend": backend,
                        "rho": rho,
                        "solver": solver,
                        "preconditioner": preconditioner,
                        "fallback": fallback,
                        "max_iterations": max_iterations,
                        "cg_tolerance": tol,
                        "optimizer_success": fit.optimizer_success,
                        "optimizer_nit": fit.optimizer_nit,
                        "optimizer_status": fit.optimizer_status,
                        "optimizer_grad_norm": fit.optimizer_grad_norm,
                        "final_converged": inner.converged,
                        "final_termination": inner.termination,
                        "final_iterations": inner.iterations,
                        "final_residual_norm": inner.residual_norm,
                        "final_threshold": inner.threshold,
                        "final_fallback": inner.fallback,
                        "loop_solves": summary.num_solves,
                        "loop_unconverged": summary.num_unconverged,
                        "loop_total_iterations": summary.total_iterations,
                        "loop_worst_residual_to_threshold":
                        summary.worst_residual_to_threshold,
                        "objective": fit.objective,
                        "relative_error": rel_error,
                        "one_norm_lcu": lam,
                        "converged": fit.converged,
                        "time_s": elapsed,
                    })


# --------------------------------------------------------------------------- #


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--inner", action="store_true")
    parser.add_argument("--fits", action="store_true")
    parser.add_argument("--cases", default=DEFAULT_CASES)
    parser.add_argument("--backends",
                        default="all",
                        help="comma list or 'all' (CuPy first when usable)")
    parser.add_argument("--rhos", default="0,1e-4,1e-3")
    parser.add_argument("--tol", type=float, default=1.0e-10)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--max-iterations", type=int, default=2000)
    parser.add_argument("--cg-max-iterations",
                        type=int,
                        default=None,
                        help="inner CG cap (default: the library default)")
    parser.add_argument("--lstsq-limit-gb",
                        type=float,
                        default=1.0,
                        help="skip lstsq when its design matrix exceeds this")
    parser.add_argument("--json", default=None, help="append JSON lines here")
    args = parser.parse_args()
    if not (args.inner or args.fits):
        args.inner = args.fits = True

    env = environment()
    print("environment:", json.dumps(env))
    print("command:", " ".join(sys.argv))
    sink = open(args.json, "a") if args.json else None

    def emit(record):
        if sink is not None:
            sink.write(json.dumps({**record, "env": env}) + "\n")
            sink.flush()

    backend_names = backends(args.backends)
    rhos = [float(r) for r in args.rhos.split(",")]
    limit_bytes = int(args.lstsq_limit_gb * 1e9)
    for case_name in args.cases.split(","):
        loaded = load_case(case_name.strip())
        if loaded is None:
            continue
        eri, one_body_eigenvalues, leaf_counts = loaded
        if args.inner:
            run_inner(case_name, eri, leaf_counts, backend_names, rhos,
                      args.tol, args.repeats, args.cg_max_iterations,
                      limit_bytes, emit)
        if args.fits:
            run_fits(case_name, eri, one_body_eigenvalues, leaf_counts,
                     backend_names, rhos, args.tol, args.max_iterations,
                     args.cg_max_iterations, limit_bytes, emit)
    if sink is not None:
        sink.close()


if __name__ == "__main__":
    main()
