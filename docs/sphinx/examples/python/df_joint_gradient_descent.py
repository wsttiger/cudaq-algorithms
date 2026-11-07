# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# [Begin Documentation]
"""Joint gradient descent for compressed double factorization, on the NumPy
backend and, when a GPU is usable, on the CuPy backend.

The default C-DF strategy (``"variable_projection"``) optimizes the leaf
rotations with L-BFGS-B and solves the cores at every step. The opt-in
``"joint_gradient_descent"`` strategy updates the antisymmetric generators
and the symmetric cores together by full-batch gradient descent with Armijo
backtracking and performs no inner core solve. This example

  * runs the joint strategy on a small exact-rank synthetic tensor and reads
    its convergence rule off the result: both packed gradient blocks within
    ``tolerance`` and the independently recomputed conditional-core
    residual within its own target;
  * shows how to select the backend explicitly and checks that the CuPy
    evaluation agrees with NumPy on the same starting point;
  * runs a compressed (fewer leaves than rank) fit under a small budget and
    labels the honest outcome: the objective decreases monotonically, and
    an unmet gradient tolerance is reported as such, never as success.

Every claim printed is asserted, so the script fails if the loop misbehaves.
"""
from __future__ import annotations

import warnings

import numpy as np

import cudaq_algorithms as algorithms

df = algorithms.double_factorization


def synthetic_eri(n, rank, seed):
    """An 8-fold-symmetric chemist-notation ERI of known rank."""
    rng = np.random.default_rng(seed)
    leaves = rng.standard_normal((rank, n, n))
    leaves = 0.5 * (leaves + leaves.transpose(0, 2, 1))
    return np.einsum("xpq,xrs->pqrs", leaves, leaves)


def report(label, fit):
    joint = fit.joint
    print(f"  {label}: {'converged' if fit.converged else 'UNCONVERGED'} "
          f"({joint.termination}) on {joint.backend}")
    print(f"    accepted updates={joint.iterations}  objective evaluations="
          f"{joint.objective_evaluations}  backtracks={joint.backtracks}  "
          f"final step={joint.final_step}")
    print(f"    ||g_X||_inf={joint.generator_gradient_inf:.2e}  "
          f"||g_Z||_inf={joint.core_gradient_inf:.2e}  "
          f"tolerance={joint.gradient_tolerance:g}")
    print(f"    conditional-core residual={joint.core_residual_norm:.2e} "
          f"<= threshold {joint.core_residual_threshold:.2e}: "
          f"{joint.core_residual_converged}")
    print(
        f"    objective={joint.objective:.3e}  core norm={joint.core_norm:.3f}"
        f"  {joint.elapsed_seconds * 1e3:.0f} ms")
    # A joint fit carries no inner-solve diagnostics: nothing was solved.
    assert fit.inner_solve is None and fit.inner_solve_summary is None
    assert fit.optimization_strategy == "joint_gradient_descent"
    assert fit.optimizer_success == joint.success == fit.converged
    return joint


def main():
    gpu = df.cupy_gpu_available()
    print("Joint gradient descent for C-DF")
    print(f"  GPU usable for the CuPy backend: {gpu}")

    # 1. Exact-rank case: 2 leaves fit a rank-2 tensor exactly, and the
    #    complete convergence rule is reachable. The gradient tolerance is
    #    an absolute infinity norm per packed block; the core residual
    #    target is relative to the projected right-hand side.
    eri = synthetic_eri(4, 2, 0)
    options = df.JointGradientDescentOptions(core_residual_tolerance=1e-8,
                                             history_length=5)
    print("\nExact-rank synthetic (n=4, rank 2, 2 leaves):")
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # a converged fit must not warn
        exact = df.compressed_double_factorization(
            eri,
            num_leaves=2,
            max_iterations=6000,
            tolerance=1e-8,
            optimization_strategy="joint_gradient_descent",
            joint_options=options,
            backend="numpy")
    joint = report("joint GD, NumPy", exact)
    assert joint.termination == "converged"
    assert df.factorization_error(eri, exact) < 1e-6
    print(f"    reconstruction error={df.factorization_error(eri, exact):.2e}")

    # 2. Backend selection. The same starting point is handed to both
    #    backends; the CuPy run keeps every tensor on the device and reads
    #    back scalars only. The first accepted update agrees to float64
    #    rounding; long trajectories may differ in the last digits.
    if gpu:
        print("\nNumPy vs CuPy from one shared start (8 updates):")
        rng = np.random.default_rng(7)
        generators = [0.3 * rng.standard_normal((4, 4)) for _ in range(2)]
        generators = [g - g.T for g in generators]  # antisymmetric
        runs = {}
        for backend in ("numpy", "cupy"):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                runs[backend] = df.compressed_double_factorization(
                    eri,
                    num_leaves=2,
                    max_iterations=8,
                    tolerance=1e-12,
                    optimization_strategy="joint_gradient_descent",
                    joint_options=df.JointGradientDescentOptions(
                        history_length=10),
                    initial_generators=generators,
                    backend=backend)
        for backend, fit in runs.items():
            first = fit.joint.history[1]["objective"]
            print(f"  {backend:5}: objective after the first update "
                  f"{first:.12e}, host scalar reads {fit.joint.host_syncs}")
        host, device = runs["numpy"].joint, runs["cupy"].joint
        assert device.backend == "cupy"
        assert np.isclose(host.history[1]["objective"],
                          device.history[1]["objective"],
                          rtol=1e-10)
    else:
        print("\n  (CuPy backend not usable here: backend='cupy' would raise "
              "rather than fall back to NumPy.)")

    # 3. Compressed fit under a budget: rank 5 into 2 leaves. The objective
    #    is monotone along the accepted steps (Armijo), and the unmet
    #    tolerance is reported as a termination reason, not as success.
    compressed_eri = synthetic_eri(4, 5, 1)
    print("\nCompressed synthetic (n=4, rank 5, 2 leaves, 300 updates):")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        compressed = df.compressed_double_factorization(
            compressed_eri,
            num_leaves=2,
            max_iterations=300,
            tolerance=1e-8,
            optimization_strategy="joint_gradient_descent",
            joint_options=df.JointGradientDescentOptions(history_length=301),
            backend="numpy")
    joint = report("joint GD, NumPy", compressed)
    objectives = [record["objective"] for record in joint.history]
    assert all(b < a for a, b in zip(objectives, objectives[1:]))
    assert joint.termination == "max_iterations" and not compressed.converged
    assert len(caught) == 1 and issubclass(caught[0].category, RuntimeWarning)
    print(f"    objective {objectives[0]:.4e} -> {objectives[-1]:.4e} over "
          f"{len(objectives) - 1} monotone updates; one RuntimeWarning "
          "reported the unmet tolerance")

    # The default strategy is unchanged: the same call without
    # optimization_strategy runs variable projection with its inner solve.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        default = df.compressed_double_factorization(compressed_eri,
                                                     num_leaves=2,
                                                     backend="numpy")
    assert default.optimization_strategy == "variable_projection"
    assert default.joint is None and default.inner_solve is not None
    print(f"\n  default strategy: {default.optimization_strategy} "
          f"(objective {default.objective:.4e})")


if __name__ == "__main__":
    main()
