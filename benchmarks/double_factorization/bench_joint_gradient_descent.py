# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Joint gradient descent vs variable projection, on CPU (NumPy) and GPU
(CuPy), on the shared H8/H10 inputs and starts of
``bench_dense_pcg_numpy.py``.

Cases: linear H8 and H10 (1.4 bohr, STO-6G, RHF to 1e-12), T = 2n leaves,
rho in {0, 2.5e-5, 5e-5}, seeds 0-4 (seed 0: real-Schur logarithm of the
eigen-X-DF rotations; seeds 1-4: packed perturbations of norm 1e-3). The
integrals and initial generators X0 are the cached arrays of the dense
benchmark, reused read-only through ``--cache``. The initial cores Z0 are
prepared once per (n, seed, rho) at the fixed initial rotations by the dense
Cholesky PCG inner solve at tolerance 1e-12 (its residual and cost are
recorded and the cost is charged to every joint fit of that case); the
optional ``heuristic`` arm instead uses the library's solver-free default
cores. Identical host arrays are handed to the NumPy and CuPy arms.

Arms:

- ``joint_numpy`` / ``joint_cupy``: ``optimization_strategy=
  "joint_gradient_descent"``, 2000 accepted updates, block gradient
  tolerance 1e-8, conditional-core residual target 1e-10, Armijo c1 = 1e-4,
  factor 0.5, 20 backtracks, adaptive step from 1.0, step scales from the
  frozen pilot tuning (``TUNED``).
- ``vp_numpy`` / ``vp_cupy``: the existing variable projection with plain CG
  (``inner_solver="cg"``, preconditioner none, fallback off, warm start
  off, in-loop and final tolerance 1e-10, cap max(50, T n^2), 200
  L-BFGS-B steps, ftol = gtol = 1e-8), the dense benchmark's "cg" arm.
- ``dense_numpy`` (optional): variable projection with the dense Cholesky
  preconditioner.
- ``joint_numpy_long`` / ``joint_cupy_long`` (supplementary): the joint
  arms with a 40000-update budget, roughly the wall time of the CPU
  variable-projection arms at these sizes.
- ``joint_numpy_heuristic`` / ``joint_cupy_heuristic`` (supplementary):
  the joint arms started from the library's solver-free default cores.

Timing: BLAS/OpenMP threads are pinned to 1 before NumPy is imported; the
requested backend is forced (a CuPy arm that cannot run is recorded as an
error, never silently run on NumPy); one untimed warm-up per configuration
per process; the device is synchronized before and after every timed
region; sequential runs. Each record carries the Z0 preparation time, the
library call time (which includes host-to-device transfers, the loop, the
final audit, and the result transfer), the loop time reported by the
library, the external audit time, and the end-to-end time
(preparation + call), plus peak memory, versions, revision, checksums,
settings, warnings, and bounded progress histories (the joint per-update
history; for variable projection the L-BFGS-B objective per step through a
benchmark-only callback).

    python3 benchmarks/double_factorization/bench_joint_gradient_descent.py \\
        --mode pilot --out results_dir --cache inputs_dir --device 1
    python3 benchmarks/double_factorization/bench_joint_gradient_descent.py \\
        --mode tune --out results_dir --cache inputs_dir --device 1
    python3 benchmarks/double_factorization/bench_joint_gradient_descent.py \\
        --mode grid --out results_dir --cache inputs_dir --device 1 --resume
    python3 benchmarks/double_factorization/bench_joint_gradient_descent.py \\
        --report --out results_dir
"""
from __future__ import annotations

import os
import sys

for _variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                  "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_variable, "1")

import argparse  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import pathlib  # noqa: E402
import resource  # noqa: E402
import time  # noqa: E402
import tracemalloc  # noqa: E402
import warnings  # noqa: E402

_HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))


def _parse_device(argv):
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--device", default=None)
    known, _ = parser.parse_known_args(argv)
    return known.device


_DEVICE = _parse_device(sys.argv[1:])
if _DEVICE is not None:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(_DEVICE)
else:
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import numpy as np  # noqa: E402
import scipy  # noqa: E402

import bench_dense_pcg_numpy as dense  # noqa: E402  (shared protocol)
from cudaq_algorithms import double_factorization as df  # noqa: E402
from cudaq_algorithms.double_factorization import _factorization as F  # noqa: E402

try:
    import cupy  # noqa: E402
except Exception:  # pragma: no cover - CPU-only environment
    cupy = None

SOURCE_REVISION = "unknown"

ORBITALS = (8, 10)
RHOS_PILOT = (0.0, 5.0e-5)
RHOS_GRID = (0.0, 2.5e-5, 5.0e-5)
SEEDS_PILOT = (1, )
SEEDS_GRID = (0, 1, 2, 3, 4)

JOINT_MAX_ITERATIONS = 2000
JOINT_LONG_MAX_ITERATIONS = 40000
JOINT_TOLERANCE = 1.0e-8
JOINT_CORE_RESIDUAL_TOLERANCE = 1.0e-10
Z0_TOLERANCE = 1.0e-12

# Frozen after the pilot tuning (``--mode tune``): block step scales and
# initial step of the joint arms. ``TUNING_CANDIDATES`` is the declared set.
TUNED = dict(generator_step_scale=4.0, core_step_scale=1.0, initial_step=1.0)
TUNING_CANDIDATES = (
    dict(generator_step_scale=1.0, core_step_scale=1.0, initial_step=1.0),
    dict(generator_step_scale=1.0, core_step_scale=4.0, initial_step=1.0),
    dict(generator_step_scale=4.0, core_step_scale=1.0, initial_step=1.0),
    dict(generator_step_scale=1.0, core_step_scale=1.0, initial_step=0.1),
)
TUNING_ITERATIONS = 300

ARMS = {
    "joint_numpy":
    dict(kind="joint", backend="numpy"),
    "joint_cupy":
    dict(kind="joint", backend="cupy"),
    "vp_numpy":
    dict(kind="vp", backend="numpy", cg_preconditioner="none"),
    "vp_cupy":
    dict(kind="vp", backend="cupy", cg_preconditioner="none"),
    "dense_numpy":
    dict(kind="vp", backend="numpy", cg_preconditioner="dense_cholesky"),
    "joint_numpy_heuristic":
    dict(kind="joint", backend="numpy", heuristic=True),
    "joint_cupy_heuristic":
    dict(kind="joint", backend="cupy", heuristic=True),
    # Supplementary equal-wall-time arms: the joint loop with twenty times
    # the primary update budget, which at n <= 10 brings its CPU time into
    # the range of the variable-projection arms.
    "joint_numpy_long":
    dict(kind="joint",
         backend="numpy",
         max_iterations=JOINT_LONG_MAX_ITERATIONS),
    "joint_cupy_long":
    dict(kind="joint",
         backend="cupy",
         max_iterations=JOINT_LONG_MAX_ITERATIONS),
}
PRIMARY_ARMS = ("joint_numpy", "joint_cupy", "vp_numpy", "vp_cupy")

# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def sync(backend: str) -> None:
    if backend == "cupy" and cupy is not None:
        cupy.cuda.Stream.null.synchronize()


def gpu_memory_reset():
    if cupy is None:
        return None
    pool = cupy.get_default_memory_pool()
    pool.free_all_blocks()
    return pool


def gpu_peak_bytes(pool) -> int | None:
    return None if pool is None else int(pool.total_bytes())


def rss_bytes() -> int:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024


def environment() -> dict:
    info = dense.environment()
    info["source_revision"] = SOURCE_REVISION
    info["cuda_visible_devices"] = os.environ.get("CUDA_VISIBLE_DEVICES")
    if cupy is not None:
        try:
            info["cupy"] = cupy.__version__
            info["cuda_runtime"] = cupy.cuda.runtime.runtimeGetVersion()
            if df.cupy_gpu_available():
                props = cupy.cuda.runtime.getDeviceProperties(0)
                info["gpu"] = props["name"].decode() if isinstance(
                    props["name"], bytes) else str(props["name"])
                info["gpu_memory_bytes"] = int(props["totalGlobalMem"])
        except Exception as error:  # pragma: no cover
            info["cupy_error"] = str(error)
    else:
        info["cupy"] = None
    return info


def prepare_cores(case, rho, out: pathlib.Path) -> dict:
    """Shared high-accuracy Z0 at the fixed initial rotations: dense
    Cholesky PCG at Z0_TOLERANCE on NumPy, cached with its diagnostics."""
    n, seed = case["n"], case["seed"]
    path = out / "starts" / f"H{n}_seed{seed}_rho{rho:g}_Z0.npz"
    if path.exists():
        data = np.load(path, allow_pickle=False)
        meta = json.loads(str(data["meta"]))
        if meta["generators_sha"] == case["checksums"]["generators"]:
            return dict(cores=data["cores"], **meta)
    rotations = dense.rotations_from(case["generators"])
    cap = max(50, 10 * (2 * n) * n * n)
    start = time.perf_counter()
    cores, diag = F._solve_inner_cores_cg(case["eri"],
                                          rotations,
                                          np,
                                          rho,
                                          Z0_TOLERANCE,
                                          cap,
                                          None,
                                          preconditioner="dense_cholesky")
    elapsed = time.perf_counter() - start
    cores = np.stack(cores)
    meta = dict(solver="cg/dense_cholesky",
                tolerance=Z0_TOLERANCE,
                converged=bool(diag.converged),
                termination=diag.termination,
                iterations=int(diag.iterations),
                residual_norm=float(diag.residual_norm),
                rhs_norm=float(diag.rhs_norm),
                relative_residual=float(diag.residual_ratio),
                prepare_s=elapsed,
                generators_sha=case["checksums"]["generators"],
                cores_sha=dense.sha(cores))
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, cores=cores, meta=np.array(json.dumps(meta)))
    return dict(cores=cores, **meta)


def joint_settings(
        n: int, rho: float, tuned: dict, max_iterations: int,
        backend: str) -> tuple[dict, df.JointGradientDescentOptions]:
    options = df.JointGradientDescentOptions(
        generator_step_scale=tuned["generator_step_scale"],
        core_step_scale=tuned["core_step_scale"],
        initial_step=tuned["initial_step"],
        armijo_coefficient=1.0e-4,
        backtracking_factor=0.5,
        max_backtracks=20,
        step_policy="adaptive",
        core_residual_tolerance=JOINT_CORE_RESIDUAL_TOLERANCE,
        history_length=max_iterations + 1)
    settings = dict(num_leaves=2 * n,
                    regularization=rho,
                    max_iterations=max_iterations,
                    tolerance=JOINT_TOLERANCE,
                    optimization_strategy="joint_gradient_descent",
                    backend=backend)
    return settings, options


def vp_settings(n: int, rho: float, backend: str, preconditioner: str) -> dict:
    settings = dense.fit_settings(n, rho, np.zeros((2 * n, n, n)))
    settings.pop("initial_generators")
    settings.update(backend=backend,
                    inner_solver="cg",
                    cg_preconditioner=preconditioner)
    return settings


class _MinimizeRecorder:
    """Benchmark-only instrumentation: wraps ``scipy.optimize.minimize`` as
    used by the variable-projection path to record (time, objective) per
    L-BFGS-B step through its ``intermediate_result`` callback."""

    def __init__(self):
        self.records = []
        self.start = None
        self.original = F.scipy.optimize.minimize

    def __enter__(self):
        recorder = self

        def wrapped(fun, x0, **kwargs):
            recorder.start = time.perf_counter()
            recorder.records = []

            def callback(intermediate_result):
                recorder.records.append(
                    dict(iteration=len(recorder.records) + 1,
                         elapsed=time.perf_counter() - recorder.start,
                         objective=float(intermediate_result.fun)))

            kwargs["callback"] = callback
            return recorder.original(fun, x0, **kwargs)

        F.scipy.optimize.minimize = wrapped
        return self

    def __exit__(self, *exc):
        F.scipy.optimize.minimize = self.original
        return False


def run_joint(case, rho, arm, tuned, max_iterations, z0: dict | None) -> dict:
    n = case["n"]
    spec = ARMS[arm]
    backend = spec["backend"]
    settings, options = joint_settings(n, rho, tuned, max_iterations, backend)
    generators = [g for g in case["generators"]]
    cores = None if spec.get("heuristic") else [c for c in z0["cores"]]
    pool = gpu_memory_reset() if backend == "cupy" else None
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        # Untimed warm-up with a one-update budget; nothing is carried over.
        df.compressed_double_factorization(case["eri"],
                                           **{
                                               **settings, "max_iterations": 1
                                           },
                                           joint_options=options,
                                           initial_generators=generators,
                                           initial_cores=cores)
        # Host peak memory is probed on an untimed three-update run (the
        # per-update allocation pattern is the same); tracemalloc would
        # otherwise slow the timed call.
        peak_host = None
        if backend == "numpy":
            tracemalloc.start()
            df.compressed_double_factorization(
                case["eri"],
                **{
                    **settings, "max_iterations": 3
                },
                joint_options=options,
                initial_generators=generators,
                initial_cores=cores)
            peak_host = tracemalloc.get_traced_memory()[1]
            tracemalloc.stop()
        sync(backend)
        rss_before = rss_bytes()
        start = time.perf_counter()
        fit = df.compressed_double_factorization(case["eri"],
                                                 **settings,
                                                 joint_options=options,
                                                 initial_generators=generators,
                                                 initial_cores=cores)
        sync(backend)
        call_s = time.perf_counter() - start
    audit_start = time.perf_counter()
    checks = dense.audit(case["eri"], case["one_body"], fit.leaf_rotations,
                         fit.leaf_cores, rho)
    audit_s = time.perf_counter() - audit_start
    joint = fit.joint
    history = joint.history or []
    prepare_s = 0.0 if z0 is None else float(z0["prepare_s"])
    return dict(kind="fit",
                n=n,
                seed=case["seed"],
                rho=rho,
                arm=arm,
                strategy="joint_gradient_descent",
                backend=joint.backend,
                settings={
                    **settings, "options": options.as_dict()
                },
                z0=(None if z0 is None else {
                    k: v
                    for k, v in z0.items() if k != "cores"
                }),
                prepare_s=prepare_s,
                call_s=call_s,
                loop_s=joint.elapsed_seconds,
                audit_s=audit_s,
                end_to_end_s=prepare_s + call_s,
                time_s=call_s,
                per_update_s=(joint.elapsed_seconds /
                              joint.iterations if joint.iterations else None),
                per_evaluation_s=(joint.elapsed_seconds /
                                  joint.objective_evaluations
                                  if joint.objective_evaluations else None),
                peak_host_bytes=peak_host,
                rss_delta_bytes=rss_bytes() - rss_before,
                peak_device_bytes=gpu_peak_bytes(pool),
                optimizer_success=fit.optimizer_success,
                optimizer_status=fit.optimizer_status,
                optimizer_message=fit.optimizer_message,
                optimizer_nit=fit.optimizer_nit,
                optimizer_grad_norm=fit.optimizer_grad_norm,
                termination=joint.termination,
                iterations=joint.iterations,
                objective_evaluations=joint.objective_evaluations,
                gradient_evaluations=joint.gradient_evaluations,
                backtracks=joint.backtracks,
                host_syncs=joint.host_syncs,
                final_step=joint.final_step,
                generator_gradient_inf=joint.generator_gradient_inf,
                core_gradient_inf=joint.core_gradient_inf,
                core_residual_norm=joint.core_residual_norm,
                core_residual_threshold=joint.core_residual_threshold,
                core_residual_ratio=joint.core_residual_ratio,
                core_residual_converged=joint.core_residual_converged,
                core_residual_checks=joint.core_residual_checks,
                library_objective=joint.objective,
                combined_success=bool(fit.converged),
                warnings=[str(w.message)[:300] for w in caught],
                history=[
                    dict(iteration=h["iteration"],
                         objective=h["objective"],
                         gx=h["generator_gradient_inf"],
                         gz=h["core_gradient_inf"],
                         step=h["step"],
                         backtracks=h["backtracks"]) for h in history
                ],
                **checks)


def run_vp(case, rho, arm) -> dict:
    n = case["n"]
    spec = ARMS[arm]
    backend = spec["backend"]
    settings = vp_settings(n, rho, backend, spec["cg_preconditioner"])
    generators = [g for g in case["generators"]]
    pool = gpu_memory_reset() if backend == "cupy" else None
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        df.compressed_double_factorization(case["eri"],
                                           **{
                                               **settings, "max_iterations": 1
                                           },
                                           initial_generators=generators)
        peak_host = None
        if backend == "numpy":
            tracemalloc.start()
            df.compressed_double_factorization(
                case["eri"],
                **{
                    **settings, "max_iterations": 3
                },
                initial_generators=generators)
            peak_host = tracemalloc.get_traced_memory()[1]
            tracemalloc.stop()
        sync(backend)
        rss_before = rss_bytes()
        with _MinimizeRecorder() as recorder:
            start = time.perf_counter()
            fit = df.compressed_double_factorization(
                case["eri"], **settings, initial_generators=generators)
            sync(backend)
            call_s = time.perf_counter() - start
    audit_start = time.perf_counter()
    checks = dense.audit(case["eri"], case["one_body"], fit.leaf_rotations,
                         fit.leaf_cores, rho)
    audit_s = time.perf_counter() - audit_start
    inner, summary = fit.inner_solve, fit.inner_solve_summary
    return dict(
        kind="fit",
        n=n,
        seed=case["seed"],
        rho=rho,
        arm=arm,
        strategy="variable_projection",
        backend=backend,
        settings=settings,
        prepare_s=0.0,
        call_s=call_s,
        loop_s=None,
        audit_s=audit_s,
        end_to_end_s=call_s,
        time_s=call_s,
        per_update_s=call_s / fit.optimizer_nit if fit.optimizer_nit else None,
        peak_host_bytes=peak_host,
        rss_delta_bytes=rss_bytes() - rss_before,
        peak_device_bytes=gpu_peak_bytes(pool),
        optimizer_success=fit.optimizer_success,
        optimizer_status=fit.optimizer_status,
        optimizer_message=fit.optimizer_message,
        optimizer_nit=fit.optimizer_nit,
        optimizer_grad_norm=fit.optimizer_grad_norm,
        final_converged=inner.converged,
        final_termination=inner.termination,
        final_iterations=inner.iterations,
        final_solver_residual=inner.residual_norm,
        final_threshold=inner.threshold,
        loop_solves=summary.num_solves,
        loop_unconverged=summary.num_unconverged,
        loop_total_iterations=summary.total_iterations,
        loop_terminations=dict(summary.terminations),
        library_objective=fit.objective,
        combined_success=bool(fit.optimizer_success and inner.converged),
        all_inner_verified=bool(inner.converged
                                and summary.num_unconverged == 0),
        warnings=[str(w.message)[:300] for w in caught],
        history=recorder.records,
        **checks)


def record_key(r) -> tuple:
    return (r["kind"], r["n"], r["seed"], r["rho"], r["arm"],
            json.dumps(r.get("tuned"), sort_keys=True))


# --------------------------------------------------------------------------- #
# Report
# --------------------------------------------------------------------------- #


def _load(out: pathlib.Path) -> list:
    path = out / "results.jsonl"
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines()]


def _fmt(value, spec=".3e"):
    return "-" if value is None else format(value, spec)


def _time_to_target(record, target):
    """Elapsed time at which the recorded progress first reaches
    ``objective <= target`` (joint: per-update time is uniform within the
    loop, so elapsed is interpolated from the loop time; VP: callback
    stamps)."""
    history = record.get("history") or []
    if not history:
        return None
    if record["strategy"] == "joint_gradient_descent":
        per_update = record.get("per_update_s") or 0.0
        for h in history:
            if h["objective"] <= target:
                return h["iteration"] * per_update
        return None
    for h in history:
        if h["objective"] <= target:
            return h["elapsed"]
    return None


def write_report(out: pathlib.Path) -> pathlib.Path:
    records = [r for r in _load(out) if r.get("kind") == "fit"]
    errors = [r for r in _load(out) if "error" in r]
    tuning = [r for r in records if r.get("tuned") is not None]
    fits = [r for r in records if r.get("tuned") is None]
    env_path = out / "environment.json"
    env = json.loads(env_path.read_text()) if env_path.exists() else {}
    lines = [
        "# Joint gradient descent vs variable projection: CPU and GPU",
        "",
        f"Source revision: `{env.get('source_revision', 'unknown')}`; NumPy "
        f"{env.get('numpy')}, SciPy {env.get('scipy')}, CuPy "
        f"{env.get('cupy')}, GPU {env.get('gpu')}; BLAS threads "
        f"{env.get('threads')}.",
        "",
        "Every fit below shares the same objective, regularization, cached "
        "integrals, initial generators, and (joint arms) the same prepared "
        "Z0. Joint arms: 2000 accepted updates, block gradient tolerance "
        "1e-8, core residual target 1e-10. VP arms: 200 L-BFGS-B steps, "
        "ftol = gtol = 1e-8, plain CG at 1e-10 (dense arm: dense Cholesky "
        "PCG).",
        "",
    ]
    if fits:
        lines += [
            "## Per-fit results",
            "",
            "| n | seed | rho | arm | backend | termination | success | "
            "nit | objective | ||g_X||_inf | ||g_Z||_inf | core resid ratio "
            "| recon err | core norm | lambda (lcu) | tangent grad | loop s "
            "| call s | end-to-end s | peak dev MiB |",
            "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|",
        ]
        for r in sorted(fits,
                        key=lambda r: (r["n"], r["seed"], r["rho"], r["arm"])):
            joint = r["strategy"] == "joint_gradient_descent"
            dev = r.get("peak_device_bytes")
            lines.append(
                f"| {r['n']} | {r['seed']} | {r['rho']:g} | {r['arm']} | "
                f"{r['backend']} | "
                f"{r.get('termination') if joint else r.get('optimizer_message', '')[:40]} | "
                f"{r['combined_success']} | {r['optimizer_nit']} | "
                f"{_fmt(r['objective'])} | "
                f"{_fmt(r.get('generator_gradient_inf')) if joint else '-'} | "
                f"{_fmt(r.get('core_gradient_inf')) if joint else '-'} | "
                f"{_fmt(r.get('core_residual_ratio') if joint else r.get('relative_residual'))} | "
                f"{_fmt(r['reconstruction_error'])} | {r['core_norm']:.3f} | "
                f"{r['one_norm_lcu_centered']:.3f} | "
                f"{_fmt(r['tangent_gradient_inf'])} | "
                f"{_fmt(r.get('loop_s'), '.2f')} | {r['call_s']:.2f} | "
                f"{r['end_to_end_s']:.2f} | "
                f"{'-' if dev is None else f'{dev / 2**20:.0f}'} |")
        lines.append("")
        # Aggregates per (n, rho, arm).
        lines += [
            "## Aggregates per (n, rho, arm): medians over seeds",
            "",
            "| n | rho | arm | fits | successes | median objective | "
            "median lambda | median recon err | median end-to-end s | "
            "median per-update ms | median time to within 10% of best | "
            "median time to within 1% of best |",
            "|---|---|---|---|---|---|---|---|---|---|---|---|",
        ]
        groups = {}
        for r in fits:
            groups.setdefault((r["n"], r["rho"], r["arm"]), []).append(r)
        best = {}
        for r in fits:
            key = (r["n"], r["seed"], r["rho"])
            best[key] = min(best.get(key, np.inf), r["objective"])
        for (n, rho, arm), rows in sorted(groups.items()):
            t10, t1 = [], []
            for r in rows:
                b = best[(r["n"], r["seed"], r["rho"])]
                value = _time_to_target(r, 1.10 * b)
                t10.append(np.inf if value is None else value)
                value = _time_to_target(r, 1.01 * b)
                t1.append(np.inf if value is None else value)
            per_update = [
                1e3 * r["per_update_s"] for r in rows if r.get("per_update_s")
            ]
            lines.append(
                f"| {n} | {rho:g} | {arm} | {len(rows)} | "
                f"{sum(r['combined_success'] for r in rows)} | "
                f"{np.median([r['objective'] for r in rows]):.3e} | "
                f"{np.median([r['one_norm_lcu_centered'] for r in rows]):.3f} | "
                f"{np.median([r['reconstruction_error'] for r in rows]):.3e} | "
                f"{np.median([r['end_to_end_s'] for r in rows]):.2f} | "
                f"{np.median(per_update) if per_update else float('nan'):.2f} | "
                f"{np.median(t10):.2f} | {np.median(t1):.2f} |")
        lines.append("")
    if tuning:
        lines += [
            "## Pilot tuning (declared candidates, "
            f"{TUNING_ITERATIONS} updates each)", "",
            "| n | rho | s_x | s_z | alpha_0 | backend | objective | "
            "||g_X||_inf | ||g_Z||_inf | backtracks | loop s |",
            "|---|---|---|---|---|---|---|---|---|---|---|"
        ]
        for r in sorted(tuning,
                        key=lambda r:
                        (r["n"], r["rho"], json.dumps(r["tuned"]))):
            t = r["tuned"]
            lines.append(
                f"| {r['n']} | {r['rho']:g} | {t['generator_step_scale']} | "
                f"{t['core_step_scale']} | {t['initial_step']} | "
                f"{r['backend']} | {r['objective']:.4e} | "
                f"{r['generator_gradient_inf']:.2e} | "
                f"{r['core_gradient_inf']:.2e} | {r['backtracks']} | "
                f"{r['loop_s']:.2f} |")
        lines.append("")
    if errors:
        lines += ["## Errors", ""]
        for r in errors:
            lines.append(f"- H{r['n']} seed={r['seed']} rho={r['rho']:g} "
                         f"{r['arm']}: {r['error'][:300]}")
        lines.append("")
    path = out / "REPORT.md"
    path.write_text("\n".join(lines))
    return path


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #


def main():
    global SOURCE_REVISION
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mode",
                        choices=("pilot", "tune", "grid"),
                        default="pilot")
    parser.add_argument("--out", required=True, help="results directory")
    parser.add_argument("--cache",
                        default=None,
                        help="input cache (default: out/inputs)")
    parser.add_argument("--device",
                        default=None,
                        help="CUDA device index for the CuPy arms")
    parser.add_argument("--arms", default=",".join(PRIMARY_ARMS))
    parser.add_argument("--orbitals", default=",".join(map(str, ORBITALS)))
    parser.add_argument("--seeds", default=None)
    parser.add_argument("--rhos", default=None)
    parser.add_argument("--max-iterations",
                        type=int,
                        default=JOINT_MAX_ITERATIONS)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--report",
                        action="store_true",
                        help="only write REPORT.md from results.jsonl")
    args = parser.parse_args()
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if args.report:
        print("wrote", write_report(out))
        return
    cache = pathlib.Path(args.cache) if args.cache else out / "inputs"
    try:
        import subprocess
        SOURCE_REVISION = subprocess.check_output(["git", "rev-parse", "HEAD"],
                                                  cwd=_HERE,
                                                  text=True).strip()
    except Exception:
        SOURCE_REVISION = "unknown"
    dense.SOURCE_REVISION = SOURCE_REVISION
    env = environment()
    (out / "environment.json").write_text(
        json.dumps(env, indent=1, default=str))
    print("environment:", json.dumps(env, default=str)[:600])

    seeds = ([int(s) for s in args.seeds.split(",")] if args.seeds else
             list(SEEDS_GRID if args.mode == "grid" else SEEDS_PILOT))
    rhos = ([float(r) for r in args.rhos.split(",")] if args.rhos else
            list(RHOS_GRID if args.mode == "grid" else RHOS_PILOT))
    arms = [a.strip() for a in args.arms.split(",")]
    orbitals = [int(x) for x in args.orbitals.split(",")]
    for arm in arms:
        if arm not in ARMS:
            raise SystemExit(f"unknown arm {arm!r}; choose from {list(ARMS)}")
    needs_gpu = [a for a in arms if ARMS[a]["backend"] == "cupy"]
    gpu_ok = df.cupy_gpu_available()
    print(f"GPU usable: {gpu_ok} (CUDA_VISIBLE_DEVICES="
          f"{os.environ.get('CUDA_VISIBLE_DEVICES')!r})")

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

    def describe(record):
        if "error" in record:
            return f"ERROR {record['error'][:120]}"
        if record["strategy"] == "joint_gradient_descent":
            return (
                f"{record['termination']:<28} nit={record['iterations']:>4} "
                f"gx={record['generator_gradient_inf']:.1e} "
                f"gz={record['core_gradient_inf']:.1e} "
                f"core={record['core_residual_ratio']:.1e} "
                f"obj={record['objective']:.4e} "
                f"lam={record['one_norm_lcu_centered']:.2f} "
                f"loop={record['loop_s']:.1f}s e2e={record['end_to_end_s']:.1f}s"
            )
        return (f"outer={record['optimizer_success']!s:<5} "
                f"nit={record['optimizer_nit']:>4} "
                f"final={record['final_converged']!s:<5} "
                f"rel={record['relative_residual']:.1e} "
                f"obj={record['objective']:.4e} "
                f"lam={record['one_norm_lcu_centered']:.2f} "
                f"e2e={record['end_to_end_s']:.1f}s")

    for n in orbitals:
        for seed in seeds:
            case = dense.load_case(n, seed, cache)
            (out / f"inputs_H{n}_seed{seed}.json").write_text(
                json.dumps(dict(n=n,
                                seed=seed,
                                checksums=case["checksums"],
                                rhf_energy=case["rhf_energy"],
                                pyscf=case["pyscf_version"],
                                perturbation_norm=dense.PERTURBATION_NORM),
                           indent=1))
            for rho in rhos:
                z0 = None
                if any(ARMS[a]["kind"] == "joint"
                       and not ARMS[a].get("heuristic") for a in arms):
                    z0 = prepare_cores(case, rho, out)
                    print(f"H{n} seed={seed} rho={rho:g} Z0: "
                          f"{z0['solver']} {z0['termination']} "
                          f"rel={z0['relative_residual']:.1e} "
                          f"{z0['prepare_s']:.2f}s")
                if args.mode == "tune":
                    plan = [(arm, tuned) for arm in arms
                            if ARMS[arm]["kind"] == "joint"
                            for tuned in TUNING_CANDIDATES]
                else:
                    plan = [(arm, None) for arm in arms]
                for arm, tuned in plan:
                    key = ("fit", n, seed, rho, arm,
                           json.dumps(tuned, sort_keys=True))
                    if key in done:
                        continue
                    spec = ARMS[arm]
                    if spec["backend"] == "cupy" and not gpu_ok:
                        record = dict(kind="fit",
                                      n=n,
                                      seed=seed,
                                      rho=rho,
                                      arm=arm,
                                      tuned=tuned,
                                      error="GPU unavailable: the CuPy arm "
                                      "was not run")
                        emit(record, case)
                        print(f"H{n} seed={seed} rho={rho:g} {arm:<12} "
                              f"{describe(record)}")
                        continue
                    try:
                        if spec["kind"] == "joint":
                            budget = (TUNING_ITERATIONS if tuned else spec.get(
                                "max_iterations", args.max_iterations))
                            record = run_joint(case, rho, arm, tuned or TUNED,
                                               budget, z0)
                        else:
                            record = run_vp(case, rho, arm)
                        record["tuned"] = tuned
                    except Exception as error:  # keep the failure row
                        record = dict(kind="fit",
                                      n=n,
                                      seed=seed,
                                      rho=rho,
                                      arm=arm,
                                      tuned=tuned,
                                      error=f"{type(error).__name__}: {error}")
                    emit(record, case)
                    print(f"H{n} seed={seed} rho={rho:g} {arm:<12} "
                          f"{describe(record)}")
    sink.close()
    print("wrote", write_report(out))


if __name__ == "__main__":
    main()
