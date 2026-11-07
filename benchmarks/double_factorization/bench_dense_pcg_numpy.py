# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Matched NumPy benchmark: plain CG, pairwise PCG, dense Cholesky PCG, lstsq.

Linear H8 and H10 (1.4 bohr, STO-6G, canonical RHF orbitals converged to
1e-12) at T = 2n leaves, rho in {0, 2.5e-5, 5e-5}, five initialization seeds.
Every arm receives the same cached integrals and the same initial generators
(seed 0: real-Schur logarithm of the eigen-X-DF rotations; seeds 1-4: a
multiplicative Gaussian perturbation of packed norm 1e-3). Fixed-rotation
controls solve only the cores at the shared start; complete fits run
compressed_double_factorization with the prototype's settings (200 L-BFGS-B
steps, ftol = gtol = 1e-8, CG tolerance 1e-10 in the loop and at the end,
cold start, cap = max(50, 2n * n^2)). Every arm is judged by the original
operator's residual, recomputed outside the solver. NumPy float64 only; the
process sets single-threaded BLAS before importing NumPy and never touches
CuPy.

    python3 benchmarks/double_factorization/bench_dense_pcg_numpy.py \\
        --mode pilot --out results_dir
    python3 benchmarks/double_factorization/bench_dense_pcg_numpy.py \\
        --mode grid --out results_dir --resume
"""
from __future__ import annotations

import os

for _variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                  "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_variable, "1")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")  # never pick up a GPU

import argparse  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import pathlib  # noqa: E402
import platform  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
import warnings  # noqa: E402

import numpy as np  # noqa: E402
import scipy  # noqa: E402
import scipy.linalg  # noqa: E402

from cudaq_algorithms import double_factorization as df  # noqa: E402
from cudaq_algorithms.double_factorization import _factorization as F  # noqa: E402

SOURCE_REVISION = None  # filled in main() from git when available

ARMS = {
    "cg": dict(inner_solver="cg", cg_preconditioner="none"),
    "pcg": dict(inner_solver="cg", cg_preconditioner="pairwise"),
    "dense": dict(inner_solver="cg", cg_preconditioner="dense_cholesky"),
    "lstsq": dict(inner_solver="lstsq"),
}
ORBITALS = (8, 10)
RHOS_GRID = (0.0, 2.5e-5, 5.0e-5)
RHOS_PILOT = (0.0, 5.0e-5)
SEEDS_GRID = (0, 1, 2, 3, 4)
SEEDS_PILOT = (1, )
PERTURBATION_NORM = 1.0e-3
OUTER_TOLERANCE = 1.0e-8
CG_TOLERANCE = 1.0e-10
MAX_ITERATIONS = 200

# --------------------------------------------------------------------------- #
# Inputs: integrals and shared starts
# --------------------------------------------------------------------------- #


def sha(array) -> str:
    return hashlib.sha256(
        np.ascontiguousarray(array).tobytes()).hexdigest()[:16]


def build_integrals(n: int, cache: pathlib.Path) -> dict:
    """Canonical RHF/STO-6G integrals for linear H_n at 1.4 bohr, converged to
    1e-12 (max 200 cycles, convergence required), chemist (pq|rs) with the
    full eightfold symmetry. Cached as .npz with provenance."""
    path = cache / f"H{n}_sto6g_d1.4_rhf1e-12.npz"
    if path.exists():
        data = np.load(path)
        return {k: data[k] for k in data.files}
    from pyscf import ao2mo, gto, scf
    mol = gto.M(atom=[("H", (0, 0, 1.4 * i)) for i in range(n)],
                basis="sto-6g",
                unit="Bohr",
                charge=0,
                spin=0,
                verbose=0)
    mf = scf.RHF(mol)
    mf.conv_tol = 1.0e-12
    mf.max_cycle = 200
    mf.run()
    if not mf.converged:
        raise RuntimeError(f"RHF did not converge for H{n}")
    eri = np.asarray(ao2mo.restore("s1", ao2mo.kernel(mol, mf.mo_coeff), n))
    one_body = mf.mo_coeff.T @ mf.get_hcore() @ mf.mo_coeff
    for perm in ((1, 0, 2, 3), (0, 1, 3, 2), (2, 3, 0, 1)):
        assert np.allclose(eri, eri.transpose(perm), atol=1e-12)
    import pyscf
    record = dict(eri=eri,
                  one_body=one_body,
                  core_energy=np.array(mol.energy_nuc()),
                  rhf_energy=np.array(mf.e_tot),
                  pyscf_version=np.array(pyscf.__version__))
    cache.mkdir(parents=True, exist_ok=True)
    np.savez(path, **record)
    return record


def real_schur_log(rotation: np.ndarray) -> np.ndarray:
    """Real skew-symmetric logarithm of a special-orthogonal matrix via the
    real Schur form. 2x2 blocks are rotations by an angle (log is the
    antisymmetric angle block); 1x1 blocks are +1 (log 0) or -1, which
    occur in pairs for det = +1 and are paired into a rotation by pi."""
    t, q = scipy.linalg.schur(rotation, output="real")
    n = rotation.shape[0]
    log_t = np.zeros((n, n))
    minus_ones = []
    i = 0
    while i < n:
        if i + 1 < n and abs(t[i + 1, i]) > 1e-12:
            c, s = t[i, i], t[i + 1, i]
            angle = np.arctan2(s, c)
            log_t[i, i + 1] = -angle
            log_t[i + 1, i] = angle
            i += 2
        else:
            if t[i, i] < 0.0:
                minus_ones.append(i)
            i += 1
    if len(minus_ones) % 2:
        raise ValueError("odd number of -1 eigenvalues: det(rotation) != +1")
    for a, b in zip(minus_ones[0::2], minus_ones[1::2]):
        log_t[a, b] = -np.pi
        log_t[b, a] = np.pi
    skew = q @ log_t @ q.T
    skew = 0.5 * (skew - skew.T)
    if not np.allclose(scipy.linalg.expm(skew), rotation, atol=1e-10):
        raise RuntimeError("real-Schur logarithm did not reproduce rotation")
    return skew


def special_orthogonal(rotation: np.ndarray) -> np.ndarray:
    """Flip the first column if det < 0 (leaves u_k u_k^T unchanged)."""
    rotation = np.array(rotation, dtype=float, copy=True)
    if np.linalg.det(rotation) < 0.0:
        rotation[:, 0] *= -1.0
    return rotation


def shared_generators(eri: np.ndarray, n: int, seed: int) -> np.ndarray:
    """Initial generators for all arms: eigen-X-DF base (seed 0) or the
    base perturbed multiplicatively by a seeded Gaussian skew of packed norm
    PERTURBATION_NORM (seeds >= 1)."""
    num_leaves = 2 * n
    xdf = df.explicit_double_factorization(
        eri,
        threshold=0.0,
        second_factor_threshold=0.0,
        max_num_leaves=num_leaves,
        first_factorization="eigendecomposition",
        backend="numpy")
    base = [real_schur_log(special_orthogonal(u)) for u in xdf.leaf_rotations]
    while len(base) < num_leaves:
        base.append(np.zeros((n, n)))
    if seed == 0:
        return np.stack(base)
    rng = np.random.Generator(np.random.PCG64(seed))
    out = []
    rows, cols = np.triu_indices(n, k=1)
    for generator in base:
        upper = rng.standard_normal(rows.shape[0])
        upper *= PERTURBATION_NORM / np.linalg.norm(upper)
        delta = np.zeros((n, n))
        delta[rows, cols] = upper
        delta -= delta.T
        rotation = scipy.linalg.expm(generator) @ scipy.linalg.expm(delta)
        out.append(real_schur_log(special_orthogonal(rotation)))
    return np.stack(out)


def load_case(n: int, seed: int, cache: pathlib.Path) -> dict:
    integrals = build_integrals(n, cache)
    start_path = cache / f"H{n}_start_seed{seed}.npy"
    if start_path.exists():
        generators = np.load(start_path)
    else:
        generators = shared_generators(integrals["eri"], n, seed)
        np.save(start_path, generators)
    return dict(n=n,
                seed=seed,
                eri=integrals["eri"],
                one_body=integrals["one_body"],
                generators=generators,
                checksums=dict(eri=sha(integrals["eri"]),
                               one_body=sha(integrals["one_body"]),
                               generators=sha(generators)),
                rhf_energy=float(integrals["rhf_energy"]),
                pyscf_version=str(integrals["pyscf_version"]))


# --------------------------------------------------------------------------- #
# Audits outside the solver
# --------------------------------------------------------------------------- #


def rotations_from(generators: np.ndarray) -> list:
    return [scipy.linalg.expm(g) for g in generators]


def audit(eri, one_body, rotations, cores, rho) -> dict:
    """Independent recomputation: original normal-equation residual,
    objective (with regularization), fitted tensor error, core norm,
    centered LCU one-norm, and a tangent-projected gradient norm."""
    op = F._InnerOperator(eri, rotations, np, rho)
    z = np.stack(cores)
    residual = np.linalg.norm(op.residual(z))
    recon = F._reconstruct_dev(rotations, z, np)
    fit_residual = eri - recon
    objective = 0.5 * float(np.sum(fit_residual**2)) + rho * float(np.sum(z**
                                                                          2))
    factorization = df.DoubleFactorization(num_orbitals=eri.shape[0],
                                           leaf_rotations=list(rotations),
                                           leaf_cores=list(cores),
                                           method="C-DF")
    kappa = df.centered_one_body(one_body, factorization)
    lam = df.double_factorization_one_norm(factorization,
                                           np.linalg.eigvalsh(kappa), "lcu")
    # Gradient of the objective with respect to U^t (Eq. 17), projected onto
    # the tangent space of SO(n) at U^t (skew part of U^T G), as a basis-free
    # stationarity measure; infinity norm over all entries.
    grad_u = -4.0 * np.einsum("aqrs,tqk,tkl,trl,tsl->tak",
                              fit_residual,
                              np.stack(rotations),
                              z,
                              np.stack(rotations),
                              np.stack(rotations),
                              optimize=True)
    tangent = [0.5 * (u.T @ g - g.T @ u) for u, g in zip(rotations, grad_u)]
    grad_inf = max(float(np.abs(t).max()) for t in tangent)
    return dict(true_residual=residual,
                rhs_norm=op.rhs_norm,
                relative_residual=residual /
                op.rhs_norm if op.rhs_norm else 0.0,
                objective=objective,
                reconstruction_error=float(np.linalg.norm(fit_residual)),
                core_norm=float(np.linalg.norm(z)),
                one_norm_lcu_centered=lam,
                tangent_gradient_inf=grad_inf)


# --------------------------------------------------------------------------- #
# Experiments
# --------------------------------------------------------------------------- #


def environment() -> dict:
    info = dict(python=platform.python_version(),
                numpy=np.__version__,
                scipy=scipy.__version__,
                cpu=platform.processor() or platform.machine(),
                threads={
                    k: os.environ.get(k)
                    for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                              "MKL_NUM_THREADS")
                },
                source_revision=SOURCE_REVISION)
    try:
        import threadpoolctl
        info["threadpools"] = threadpoolctl.threadpool_info()
    except Exception:
        info["threadpools"] = "threadpoolctl unavailable"
    try:
        import cudaq
        info["cudaq"] = cudaq.__version__
    except Exception:
        info["cudaq"] = None
    try:
        info["blas"] = str(
            np.__config__.CONFIG["Build Dependencies"]["blas"]["name"])
        info["lapack"] = str(
            np.__config__.CONFIG["Build Dependencies"]["lapack"]["name"])
    except Exception:
        info["blas"] = info["lapack"] = "unknown"
    return info


def fit_settings(n: int, rho: float, generators: np.ndarray) -> dict:
    return dict(num_leaves=2 * n,
                max_iterations=MAX_ITERATIONS,
                tolerance=OUTER_TOLERANCE,
                regularization=rho,
                cg_tolerance=CG_TOLERANCE,
                cg_optimization_tolerance=CG_TOLERANCE,
                cg_max_iterations=max(50, (2 * n) * n * n),
                cg_warm_start=False,
                initial_generators=[g for g in generators],
                backend="numpy",
                inner_fallback="none")


def run_control(case, rho, arm) -> dict:
    """Fixed-rotation core solve at the shared start, timed as a complete
    call (operator + preconditioner setup + iterations + checks)."""
    n = case["n"]
    rotations = rotations_from(case["generators"])
    overrides = ARMS[arm]
    cap = max(50, (2 * n) * n * n)

    def solve():
        if overrides["inner_solver"] == "lstsq":
            return F._solve_inner_cores_lstsq(case["eri"], rotations, np, rho,
                                              CG_TOLERANCE)
        return F._solve_inner_cores_cg(case["eri"],
                                       rotations,
                                       np,
                                       rho,
                                       CG_TOLERANCE,
                                       cap,
                                       None,
                                       preconditioner=overrides.get(
                                           "cg_preconditioner", "none"))

    solve()  # warm-up, discarded
    start = time.perf_counter()
    cores, diag = solve()
    elapsed = time.perf_counter() - start
    checks = audit(case["eri"], case["one_body"], rotations, cores, rho)
    return dict(kind="control",
                n=n,
                seed=case["seed"],
                rho=rho,
                arm=arm,
                converged=diag.converged,
                termination=diag.termination,
                iterations=diag.iterations,
                solver_residual=diag.residual_norm,
                threshold=diag.threshold,
                cg_max_iterations=cap,
                details=diag.details,
                message=diag.message,
                time_s=elapsed,
                **checks)


def run_fit(case, rho, arm) -> dict:
    n = case["n"]
    settings = fit_settings(n, rho, case["generators"])
    settings.update(ARMS[arm])
    # Untimed warm-up with a one-step budget; nothing is carried over.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        df.compressed_double_factorization(case["eri"], **{
            **settings, "max_iterations": 1
        })
        start = time.perf_counter()
        fit = df.compressed_double_factorization(case["eri"], **settings)
        elapsed = time.perf_counter() - start
    audit_start = time.perf_counter()
    checks = audit(case["eri"], case["one_body"], fit.leaf_rotations,
                   fit.leaf_cores, rho)
    audit_s = time.perf_counter() - audit_start
    inner, summary = fit.inner_solve, fit.inner_solve_summary
    strict = bool(inner.converged and summary.num_unconverged == 0)
    return dict(
        kind="fit",
        n=n,
        seed=case["seed"],
        rho=rho,
        arm=arm,
        settings={
            k: v
            for k, v in settings.items() if k != "initial_generators"
        },
        time_s=elapsed,
        audit_s=audit_s,
        optimizer_success=fit.optimizer_success,
        optimizer_status=fit.optimizer_status,
        optimizer_message=fit.optimizer_message,
        optimizer_nit=fit.optimizer_nit,
        optimizer_grad_norm_loop=fit.optimizer_grad_norm,
        final_converged=inner.converged,
        final_termination=inner.termination,
        final_iterations=inner.iterations,
        final_solver_residual=inner.residual_norm,
        final_threshold=inner.threshold,
        final_details=inner.details,
        final_message=inner.message,
        loop_solves=summary.num_solves,
        loop_unconverged=summary.num_unconverged,
        loop_total_iterations=summary.total_iterations,
        loop_worst_residual_to_threshold=summary.worst_residual_to_threshold,
        loop_terminations=dict(summary.terminations),
        library_objective=fit.objective,
        all_inner_verified=strict,
        combined_success=bool(fit.optimizer_success and inner.converged),
        **checks)


def record_key(r) -> tuple:
    return (r["kind"], r["n"], r["seed"], r["rho"], r["arm"])


def main():
    global SOURCE_REVISION
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mode", choices=("pilot", "grid"), default="pilot")
    parser.add_argument("--out", required=True, help="results directory")
    parser.add_argument("--cache",
                        default=None,
                        help="input cache (default: out/inputs)")
    parser.add_argument("--arms", default=",".join(ARMS))
    parser.add_argument("--orbitals", default=",".join(map(str, ORBITALS)))
    parser.add_argument("--seeds", default=None)
    parser.add_argument("--rhos", default=None)
    parser.add_argument("--skip-controls", action="store_true")
    parser.add_argument("--skip-fits", action="store_true")
    parser.add_argument("--resume",
                        action="store_true",
                        help="skip (kind, n, seed, rho, arm) already recorded")
    args = parser.parse_args()
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cache = pathlib.Path(args.cache) if args.cache else out / "inputs"
    try:
        import subprocess
        SOURCE_REVISION = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=pathlib.Path(__file__).resolve().parent,
            text=True).strip()
    except Exception:
        SOURCE_REVISION = "unknown"
    env = environment()
    (out / "environment.json").write_text(
        json.dumps(env, indent=1, default=str))
    print("environment:", json.dumps(env, default=str)[:400])

    seeds = ([int(s) for s in args.seeds.split(",")] if args.seeds else
             list(SEEDS_PILOT if args.mode == "pilot" else SEEDS_GRID))
    rhos = ([float(r) for r in args.rhos.split(",")] if args.rhos else
            list(RHOS_PILOT if args.mode == "pilot" else RHOS_GRID))
    arms = [a.strip() for a in args.arms.split(",")]
    orbitals = [int(x) for x in args.orbitals.split(",")]
    results_path = out / "results.jsonl"
    done = set()
    if args.resume and results_path.exists():
        for line in results_path.read_text().splitlines():
            r = json.loads(line)
            if r.get(
                    "source_revision") == SOURCE_REVISION and "error" not in r:
                done.add(record_key(r))
    sink = results_path.open("a")

    def emit(record, case):
        record.update(source_revision=SOURCE_REVISION,
                      checksums=case["checksums"],
                      env_hash=hashlib.sha256(
                          json.dumps(env, sort_keys=True,
                                     default=str).encode()).hexdigest()[:12],
                      timestamp=time.strftime("%Y-%m-%dT%H:%M:%S"))
        sink.write(json.dumps(record, default=float) + "\n")
        sink.flush()

    for n in orbitals:
        for seed in seeds:
            case = load_case(n, seed, cache)
            (out / f"inputs_H{n}_seed{seed}.json").write_text(
                json.dumps(dict(n=n,
                                seed=seed,
                                checksums=case["checksums"],
                                rhf_energy=case["rhf_energy"],
                                pyscf=case["pyscf_version"],
                                perturbation_norm=PERTURBATION_NORM),
                           indent=1))
            for rho in rhos:
                for arm in arms:
                    for kind, runner, skip in (("control", run_control,
                                                args.skip_controls),
                                               ("fit", run_fit,
                                                args.skip_fits)):
                        if skip or (kind, n, seed, rho, arm) in done:
                            continue
                        try:
                            record = runner(case, rho, arm)
                        except Exception as error:  # keep the failure row
                            record = dict(
                                kind=kind,
                                n=n,
                                seed=seed,
                                rho=rho,
                                arm=arm,
                                error=f"{type(error).__name__}: {error}")
                        emit(record, case)
                        if "error" in record:
                            print(
                                f"H{n} seed={seed} rho={rho:g} {arm:5} {kind:7} ERROR {record['error'][:100]}"
                            )
                        elif kind == "control":
                            print(
                                f"H{n} seed={seed} rho={rho:g} {arm:5} control  "
                                f"conv={record['converged']!s:5} it={record['iterations']:>5} "
                                f"rel={record['relative_residual']:.1e} {record['time_s']*1e3:8.1f} ms"
                            )
                        else:
                            print(
                                f"H{n} seed={seed} rho={rho:g} {arm:5} fit      "
                                f"outer={record['optimizer_success']!s:5} nit={record['optimizer_nit']:>3} "
                                f"final={record['final_converged']!s:5} loop={record['loop_solves']-record['loop_unconverged']}/{record['loop_solves']} "
                                f"rel={record['relative_residual']:.1e} obj={record['objective']:.3e} "
                                f"lam={record['one_norm_lcu_centered']:.2f} {record['time_s']:7.1f} s"
                            )
    sink.close()


if __name__ == "__main__":
    main()
