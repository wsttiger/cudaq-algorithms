# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the joint gradient-descent C-DF strategy: packed coordinates,
exact partial gradients, the spectral Frechet adjoint, the shared descent
loop on NumPy and CuPy, its termination contract, and API compatibility."""
import math
import os
import warnings

import numpy as np
import pytest
import scipy.linalg

import cudaq_algorithms as algorithms
from cudaq_algorithms.double_factorization import _factorization as F
from cudaq_algorithms.double_factorization import _joint as J

os.environ.setdefault("OMP_NUM_THREADS", "1")

df = algorithms.double_factorization

_BACKENDS = ["numpy"]
if (os.environ.get("CUDAQ_DEFAULT_SIMULATOR", "qpp-cpu").startswith("nvidia")
        or df.cupy_gpu_available()):
    _BACKENDS.append("cupy")

JOINT = "joint_gradient_descent"


@pytest.fixture(autouse=True)
def _skip_cupy_if_unusable(request):
    params = getattr(getattr(request.node, "callspec", None), "params", None)
    if not params or params.get("backend") != "cupy":
        return
    if not df.cupy_gpu_available():
        pytest.skip("CuPy/GPU is not usable")


def _xp(backend):
    if backend == "numpy":
        return np
    import cupy
    return cupy


def _synthetic_eri(n, num_vectors, seed):
    rng = np.random.default_rng(seed)
    leaves = rng.standard_normal((num_vectors, n, n))
    leaves = 0.5 * (leaves + leaves.transpose(0, 2, 1))
    return np.einsum("xpq,xrs->pqrs", leaves, leaves)


def _random_skew(num_leaves, n, seed, scale=1.0):
    rng = np.random.default_rng(seed)
    x = scale * rng.standard_normal((num_leaves, n, n))
    return 0.5 * (x - x.transpose(0, 2, 1))


def _random_symmetric(num_leaves, n, seed, scale=1.0):
    rng = np.random.default_rng(seed)
    z = scale * rng.standard_normal((num_leaves, n, n))
    return 0.5 * (z + z.transpose(0, 2, 1))


def _joint_objective(eri, generators, cores, rho):
    rotations = np.stack([scipy.linalg.expm(g) for g in generators])
    recon = np.einsum("tpk,tqk,tkl,trl,tsl->pqrs",
                      rotations,
                      rotations,
                      cores,
                      rotations,
                      rotations,
                      optimize=True)
    return 0.5 * float(np.sum(
        (eri - recon)**2)) + rho * float(np.sum(cores**2))


def _joint_fit(eri, num_leaves, **kwargs):
    kwargs.setdefault("backend", "numpy")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return df.compressed_double_factorization(eri,
                                                  num_leaves,
                                                  optimization_strategy=JOINT,
                                                  **kwargs)


# --------------------------------------------------------------------------- #
# (1) packed coordinates
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("n", [1, 2, 5])
def test_packed_coordinates_round_trip_and_preserve_frobenius(n):
    num_leaves = 3
    x = _random_skew(num_leaves, n, 1)
    z = _random_symmetric(num_leaves, n, 2)
    px = J.pack_skew(x, np)
    pz = J.pack_symmetric_stack(z, np)
    assert px.shape == (num_leaves * J.skew_parameter_count(n), )
    assert pz.shape == (num_leaves * J.symmetric_parameter_count(n), )
    assert np.allclose(J.unpack_skew(px, num_leaves, n, np), x, atol=1e-15)
    assert np.allclose(J.unpack_symmetric_stack(pz, num_leaves, n, np),
                       z,
                       atol=1e-15)
    assert np.isclose(np.linalg.norm(px), np.linalg.norm(x))
    assert np.isclose(np.linalg.norm(pz), np.linalg.norm(z))
    # The symmetric packing is the existing orthonormal convention.
    assert np.allclose(pz, F.pack_symmetric(z))


def test_packed_gradients_are_adjoint_to_unpacking():
    """<G, unpack(e)> == <pack_gradient(G), e> for every coordinate e."""
    num_leaves, n = 2, 4
    rng = np.random.default_rng(3)
    g = rng.standard_normal((num_leaves, n, n))  # arbitrary ambient gradient
    px = J.pack_skew_gradient(g, np)
    for index in range(px.size):
        e = np.zeros(px.size)
        e[index] = 1.0
        direction = J.unpack_skew(e, num_leaves, n, np)
        assert np.isclose(np.sum(g * direction), px[index], atol=1e-13)
    pz = J.pack_symmetric_gradient(g, np)
    for index in range(pz.size):
        e = np.zeros(pz.size)
        e[index] = 1.0
        direction = J.unpack_symmetric_stack(e, num_leaves, n, np)
        assert np.isclose(np.sum(g * direction), pz[index], atol=1e-13)


# --------------------------------------------------------------------------- #
# (2), (3) exact partial gradients
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("backend", _BACKENDS)
@pytest.mark.parametrize("rho", [0.0, 2.0e-3])
def test_joint_gradients_match_central_finite_differences(backend, rho):
    xp = _xp(backend)
    n, num_leaves = 4, 3
    eri = _synthetic_eri(n, 2, 11)
    x = J.pack_skew(_random_skew(num_leaves, n, 12), np)
    z = J.pack_symmetric_stack(_random_symmetric(num_leaves, n, 13), np)
    evaluator = J.JointEvaluator(xp.asarray(eri), num_leaves, rho, xp)
    state = evaluator.gradient(evaluator.evaluate(xp.asarray(x),
                                                  xp.asarray(z)))
    gx = np.asarray(F.to_numpy(state.gradient_x))
    gz = np.asarray(F.to_numpy(state.gradient_z))

    def objective(px, pz):
        return _joint_objective(
            eri, J.unpack_skew(px, num_leaves, n, np),
            J.unpack_symmetric_stack(pz, num_leaves, n, np), rho)

    assert np.isclose(state.objective, objective(x, z), rtol=1e-12)
    h = 1.0e-6
    for index in range(x.size):
        e = np.zeros(x.size)
        e[index] = h
        fd = (objective(x + e, z) - objective(x - e, z)) / (2 * h)
        assert abs(fd - gx[index]) <= 1e-6 * max(1.0, abs(fd))
    for index in range(z.size):
        e = np.zeros(z.size)
        e[index] = h
        fd = (objective(x, z + e) - objective(x, z - e)) / (2 * h)
        assert abs(fd - gz[index]) <= 1e-6 * max(1.0, abs(fd))
    # Combined directional derivative along a random joint direction.
    rng = np.random.default_rng(14)
    dx, dz = rng.standard_normal(x.size), rng.standard_normal(z.size)
    fd = (objective(x + h * dx, z + h * dz) -
          objective(x - h * dx, z - h * dz)) / (2 * h)
    assert np.isclose(fd, gx @ dx + gz @ dz, rtol=1e-6)


@pytest.mark.parametrize("backend", _BACKENDS)
@pytest.mark.parametrize("rho", [0.0, 1.0e-2])
def test_core_gradient_forms_agree(backend, rho):
    """-P_t(Delta) + 2 rho Z == ((A + 2 rho I) Z - R)_t."""
    xp = _xp(backend)
    n, num_leaves = 4, 3
    eri = xp.asarray(_synthetic_eri(n, 3, 21))
    x = xp.asarray(J.pack_skew(_random_skew(num_leaves, n, 22), np))
    z = xp.asarray(
        J.pack_symmetric_stack(_random_symmetric(num_leaves, n, 23), np))
    evaluator = J.JointEvaluator(eri, num_leaves, rho, xp)
    state = evaluator.evaluate(x, z)
    projection_form = evaluator.core_gradient_matrix(state)
    operator = F._InnerOperator(eri, state.rotations, xp, rho)
    normal_form = operator.apply(state.cores) - operator.rhs
    assert float(xp.abs(projection_form - normal_form).max()) < 1e-12
    # Both are symmetric, so the packed gradient loses nothing.
    assert float(
        xp.abs(projection_form -
               projection_form.transpose(0, 2, 1)).max()) < 1e-12


# --------------------------------------------------------------------------- #
# (4) spectral Frechet adjoint
# --------------------------------------------------------------------------- #


def _skew_with_eigenvalues(n, values, seed):
    """Antisymmetric matrix with +-i*values eigenvalue pairs (and 0 for an odd
    n) in a random orthogonal basis."""
    rng = np.random.default_rng(seed)
    q, _ = np.linalg.qr(rng.standard_normal((n, n)))
    block = np.zeros((n, n))
    for i, value in enumerate(values):
        block[2 * i, 2 * i + 1] = -value
        block[2 * i + 1, 2 * i] = value
    return q @ block @ q.T


@pytest.mark.parametrize("backend", _BACKENDS)
@pytest.mark.parametrize(
    "case",
    [
        "zero",
        "repeated",
        "near_repeated",
        "random",
        "large_norm",
    ],
)
def test_spectral_adjoint_matches_scipy_and_finite_differences(backend, case):
    xp = _xp(backend)
    n = 4
    if case == "zero":
        generators = np.zeros((2, n, n))
    elif case == "repeated":
        generators = np.stack([
            _skew_with_eigenvalues(n, [0.7, 0.7], 1),
            _skew_with_eigenvalues(n, [1.3, 1.3], 2)
        ])
    elif case == "near_repeated":
        generators = np.stack([
            _skew_with_eigenvalues(n, [0.7, 0.7 + 1e-9], 3),
            _skew_with_eigenvalues(n, [0.2, 0.2 - 3e-12], 4)
        ])
    elif case == "random":
        generators = _random_skew(3, n, 5)
    else:
        generators = _random_skew(3, n, 6, scale=8.0)
    rng = np.random.default_rng(7)
    grad_u = rng.standard_normal(generators.shape)
    lam, vec, rot = algorithms.double_factorization._backend.skew_spectral_factors(
        xp.asarray(generators), xp)
    # The rotations are orthogonal and the imaginary remainder negligible.
    full = vec * xp.exp(-1j * lam)[:, None, :] @ xp.conj(vec).transpose(
        0, 2, 1)
    assert float(xp.abs(xp.imag(full)).max()) < 1e-12
    eye = xp.eye(n)
    assert float(xp.abs(rot.transpose(0, 2, 1) @ rot - eye).max()) < 1e-12
    expected_rot = np.stack([scipy.linalg.expm(g) for g in generators])
    assert np.allclose(F.to_numpy(rot), expected_rot, atol=1e-12)
    spectral = F.to_numpy(
        J.rotation_adjoint_spectral(lam, vec, xp.asarray(grad_u), xp))
    reference = J.rotation_adjoint_scipy(generators, grad_u)
    assert np.allclose(spectral, reference, atol=1e-10, rtol=1e-10)
    # Adjoint identity against a finite-difference Frechet derivative:
    # <grad_u, d exp(X)[E]> == <G_X, E> for a random direction E.
    e = rng.standard_normal(generators.shape)
    h = 1e-6
    frechet = np.stack([
        (scipy.linalg.expm(g + h * d) - scipy.linalg.expm(g - h * d)) / (2 * h)
        for g, d in zip(generators, e)
    ])
    assert np.isclose(np.sum(grad_u * frechet),
                      np.sum(spectral * e),
                      rtol=1e-6,
                      atol=1e-9)


def test_divided_difference_kernel_has_no_singularity():
    lam = np.array([[0.3, 0.3, 0.3 + 1e-13, -1.1]])
    vec = np.stack([np.eye(4, dtype=complex)])
    grad = np.ones((1, 4, 4))
    out = J.rotation_adjoint_spectral(lam, vec, grad, np)
    assert np.all(np.isfinite(out))
    # Diagonal blocks of exactly repeated eigenvalues reduce to the derivative
    # of exp along a commuting direction: K = exp(-i lam).
    assert np.isclose(out[0, 0, 1], np.cos(0.3))


# --------------------------------------------------------------------------- #
# (5) NumPy / CuPy parity
# --------------------------------------------------------------------------- #


def _requires_cupy():
    if not df.cupy_gpu_available():
        pytest.skip("CuPy/GPU is not usable")
    import cupy
    return cupy


def test_cupy_evaluator_matches_numpy_on_identical_inputs():
    cupy = _requires_cupy()
    n, num_leaves, rho = 5, 4, 1.0e-3
    eri = _synthetic_eri(n, 3, 31)
    x = J.pack_skew(_random_skew(num_leaves, n, 32), np)
    z = J.pack_symmetric_stack(_random_symmetric(num_leaves, n, 33), np)
    host = J.JointEvaluator(eri, num_leaves, rho, np)
    device = J.JointEvaluator(cupy.asarray(eri), num_leaves, rho, cupy)
    hs = host.gradient(host.evaluate(x, z))
    ds = device.gradient(device.evaluate(cupy.asarray(x), cupy.asarray(z)))
    assert isinstance(ds.gradient_x, cupy.ndarray)
    assert isinstance(ds.residual, cupy.ndarray)
    assert np.isclose(hs.objective, ds.objective, rtol=1e-12)
    assert np.allclose(hs.gradient_x, cupy.asnumpy(ds.gradient_x), atol=1e-11)
    assert np.allclose(hs.gradient_z, cupy.asnumpy(ds.gradient_z), atol=1e-11)
    assert np.isclose(hs.gradient_x_inf, ds.gradient_x_inf, rtol=1e-11)
    assert np.isclose(hs.gradient_z_inf, ds.gradient_z_inf, rtol=1e-11)


def test_cupy_trajectory_matches_numpy_for_a_short_run():
    _requires_cupy()
    n, num_leaves = 4, 3
    eri = _synthetic_eri(n, 2, 41)
    generators = _random_skew(num_leaves, n, 42, scale=0.3)
    cores = _random_symmetric(num_leaves, n, 43)
    options = df.JointGradientDescentOptions(history_length=10)
    fits = {}
    for backend in ("numpy", "cupy"):
        fits[backend] = _joint_fit(eri,
                                   num_leaves,
                                   max_iterations=8,
                                   tolerance=1e-12,
                                   initial_generators=list(generators),
                                   initial_cores=list(cores),
                                   joint_options=options,
                                   backend=backend)
    host, device = fits["numpy"].joint, fits["cupy"].joint
    assert device.backend == "cupy" and host.backend == "numpy"
    assert host.iterations == device.iterations == 8
    # Same accepted steps and objectives along the short trajectory, within
    # float64 rounding (not bitwise).
    for a, b in zip(host.history, device.history):
        assert a["step"] == b["step"]
        assert np.isclose(a["objective"], b["objective"], rtol=1e-9)
    # The first accepted update agrees to float64 accuracy.
    assert np.isclose(host.history[1]["objective"],
                      device.history[1]["objective"],
                      rtol=1e-12)
    for a, b in zip(fits["numpy"].leaf_cores, fits["cupy"].leaf_cores):
        assert isinstance(b, np.ndarray)
        assert np.allclose(a, b, atol=1e-8)


# --------------------------------------------------------------------------- #
# (6) descent loop contract
# --------------------------------------------------------------------------- #


def test_armijo_steps_decrease_the_objective_and_report_truthfully():
    n, num_leaves = 4, 2
    eri = _synthetic_eri(n, 3, 51)
    options = df.JointGradientDescentOptions(history_length=100)
    fit = _joint_fit(eri,
                     num_leaves,
                     max_iterations=40,
                     tolerance=1e-12,
                     joint_options=options)
    joint = fit.joint
    assert joint.termination == "max_iterations"
    assert joint.success is False and fit.converged is False
    assert fit.optimizer_success is False
    assert fit.optimizer_status == df.JOINT_TERMINATIONS["max_iterations"]
    assert fit.optimizer_nit == joint.iterations == 40
    objectives = [h["objective"] for h in joint.history]
    assert all(b < a for a, b in zip(objectives, objectives[1:]))
    for before, after in zip(joint.history, joint.history[1:]):
        slope = -(options.generator_step_scale *
                  before["generator_gradient_inf"]**2 +
                  options.core_step_scale * before["core_gradient_inf"]**2)
        # The infinity-norm bound is weaker than the true slope, so this is
        # a necessary condition of the Armijo rule at every accepted step.
        assert after["objective"] <= before["objective"] + (
            options.armijo_coefficient * after["step"] * slope)
    assert joint.objective_evaluations == 1 + 40 + joint.backtracks + 1
    assert joint.gradient_evaluations == 1 + 40 + 1
    assert fit.inner_solve is None and fit.inner_solve_summary is None
    assert fit.inner_solve_history is None
    assert fit.optimization_strategy == JOINT
    assert math.isclose(fit.objective, joint.objective)
    assert np.isclose(
        fit.objective,
        _joint_objective(
            eri, [scipy.linalg.logm(u).real for u in fit.leaf_rotations],
            np.stack(fit.leaf_cores), 0.0),
        rtol=1e-10)
    assert joint.core_norm > 0.0 and joint.core_residual_checks >= 1


def test_trials_are_independent_evaluations(monkeypatch):
    """A rejected trial leaves no trace: the accepted state is evaluated from
    the packed vectors alone, and gradients are computed only at accepted
    states."""
    n, num_leaves = 3, 2
    eri = _synthetic_eri(n, 2, 61)
    seen = []
    original_evaluate = J.JointEvaluator.evaluate

    def spy(self, x, z):
        state = original_evaluate(self, x, z)
        seen.append((np.asarray(x).copy(), np.asarray(z).copy(), state))
        return state

    monkeypatch.setattr(J.JointEvaluator, "evaluate", spy)
    fit = _joint_fit(eri,
                     num_leaves,
                     max_iterations=5,
                     tolerance=1e-12,
                     joint_options=df.JointGradientDescentOptions(
                         initial_step=1e3, step_policy="fixed"))
    joint = fit.joint
    assert joint.backtracks > 0
    with_gradient = [s for _, _, s in seen if s.gradient_x is not None]
    # Gradients exist exactly at the accepted states (+ initial, + audit).
    assert len(with_gradient) == joint.iterations + 2
    # Re-evaluating any stored point from its packed vectors reproduces it.
    evaluator = J.JointEvaluator(eri, num_leaves, 0.0, np)
    for x, z, state in seen[:6]:
        again = evaluator.evaluate(x, z)
        assert again.objective == state.objective


def test_line_search_exhaustion_is_reported_not_hidden():
    n, num_leaves = 3, 2
    eri = _synthetic_eri(n, 2, 71)
    fit = _joint_fit(eri,
                     num_leaves,
                     max_iterations=10,
                     tolerance=1e-12,
                     joint_options=df.JointGradientDescentOptions(
                         initial_step=1e6,
                         step_policy="fixed",
                         max_backtracks=0))
    assert fit.joint.termination == "line_search_failed"
    assert fit.joint.iterations == 0
    assert fit.converged is False
    assert "line_search_failed" in fit.optimizer_message


def test_numerical_failure_at_the_start_is_reported():
    n, num_leaves = 3, 2
    eri = _synthetic_eri(n, 2, 81)
    huge = [1e200 * np.eye(n) for _ in range(num_leaves)]
    fit = _joint_fit(eri,
                     num_leaves,
                     max_iterations=10,
                     tolerance=1e-8,
                     initial_cores=huge)
    assert fit.joint.termination == "numerical_failure"
    assert fit.joint.success is False
    assert fit.optimizer_status == df.JOINT_TERMINATIONS["numerical_failure"]


def test_unsuccessful_joint_fit_warns_once():
    eri = _synthetic_eri(3, 2, 91)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        df.compressed_double_factorization(eri,
                                           2,
                                           max_iterations=3,
                                           tolerance=1e-12,
                                           optimization_strategy=JOINT,
                                           backend="numpy")
    messages = [w for w in caught if issubclass(w.category, RuntimeWarning)]
    assert len(messages) == 1
    assert "joint gradient descent did not converge" in str(
        messages[0].message)


def test_objective_stagnation_termination():
    eri = _synthetic_eri(3, 2, 92)
    fit = _joint_fit(
        eri,
        2,
        max_iterations=500,
        tolerance=1e-14,
        joint_options=df.JointGradientDescentOptions(objective_tolerance=1e-2))
    assert fit.joint.termination == "objective_stagnation"
    assert fit.converged is False


# --------------------------------------------------------------------------- #
# (7) problem classes
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("backend", _BACKENDS)
def test_zero_eri_converges_immediately_with_zero_cores(backend):
    n, num_leaves = 3, 2
    eri = np.zeros((n, n, n, n))
    generators = _random_skew(num_leaves, n, 101)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        fit = df.compressed_double_factorization(
            eri,
            num_leaves,
            optimization_strategy=JOINT,
            initial_generators=list(generators),
            backend=backend)
    joint = fit.joint
    assert joint.success and fit.converged
    assert joint.termination == "converged" and joint.iterations == 0
    assert joint.core_rhs_norm == 0.0 and joint.core_residual_norm == 0.0
    assert joint.core_residual_threshold == 0.0
    assert all(np.all(core == 0.0) for core in fit.leaf_cores)


@pytest.mark.parametrize("backend", _BACKENDS)
def test_exact_rank_synthetic_converges_with_the_full_rule(backend):
    # Plain gradient descent is slow on these ill-conditioned fits (several
    # thousand updates even at n = 4), so the converging check uses the
    # smallest nontrivial exact-rank case.
    n, num_leaves = 2, 2
    eri = _synthetic_eri(n, num_leaves, 0)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        fit = df.compressed_double_factorization(
            eri,
            num_leaves,
            max_iterations=3000,
            tolerance=1e-8,
            optimization_strategy=JOINT,
            backend=backend,
            joint_options=df.JointGradientDescentOptions(
                core_residual_tolerance=1e-8))
    joint = fit.joint
    assert joint.success and fit.converged
    assert joint.termination == "converged"
    assert joint.generator_gradient_inf <= 1e-8
    assert joint.core_gradient_inf <= 1e-8
    assert joint.core_residual_converged
    assert joint.core_residual_ratio <= 1e-8
    assert joint.backend == backend
    assert df.factorization_error(eri, fit) < 1e-5
    for core in fit.leaf_cores:
        assert np.allclose(core, core.T, atol=1e-14)


def test_compressed_synthetic_descends_and_monitors_core_norm_at_rho_zero():
    n, num_leaves = 4, 2
    eri = _synthetic_eri(n, 5, 121)  # rank 5 > 2 leaves: a compressed fit
    fit = _joint_fit(
        eri,
        num_leaves,
        max_iterations=200,
        tolerance=1e-12,
        joint_options=df.JointGradientDescentOptions(history_length=1))
    start = _joint_objective(
        eri,
        J.validate_stack(F._initial_generators(eri, num_leaves, n, "numpy"),
                         num_leaves, n, "g", True),
        F.to_numpy(
            J.default_initial_cores(
                eri,
                np.stack([
                    scipy.linalg.expm(g)
                    for g in F._initial_generators(eri, num_leaves, n, "numpy")
                ]), 0.0, np)), 0.0)
    assert fit.joint.objective < start
    assert fit.joint.generators_init == "x-df"
    assert fit.joint.cores_init == "projection_heuristic"
    assert math.isfinite(fit.joint.core_norm)
    assert fit.joint.core_norm == pytest.approx(
        np.linalg.norm(np.stack(fit.leaf_cores)))


@pytest.mark.parametrize("backend", _BACKENDS)
@pytest.mark.parametrize("rho", [0.0, 0.25])
def test_one_orbital_optimizes_the_cores_only(backend, rho):
    """n == 1: no generator coordinates; the cores are optimized from
    nonoptimal starting values. At rho > 0 they converge to the closed form
    Z^t = R / (T + 2 rho); at rho = 0 only their sum is determined (the
    differences lie in the singular normal operator's null space) and
    equals R. The gradient target is set above the float64 Armijo floor
    sqrt(eps F) of the nonzero regularized optimum."""
    num_leaves = 3
    eri = np.full((1, 1, 1, 1), 0.8)
    start = [np.array([[value]]) for value in (5.0, -2.0, 0.3)]
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        fit = df.compressed_double_factorization(
            eri,
            num_leaves,
            regularization=rho,
            max_iterations=5000,
            tolerance=1e-8,
            optimization_strategy=JOINT,
            initial_cores=start,
            backend=backend,
            joint_options=df.JointGradientDescentOptions(
                core_residual_tolerance=1e-8))
    joint = fit.joint
    assert joint.success and fit.converged
    assert joint.generator_gradient_inf == 0.0
    assert joint.iterations > 0
    values = [core[0, 0] for core in fit.leaf_cores]
    if rho > 0.0:
        expected = 0.8 / (num_leaves + 2 * rho)
        assert np.allclose(values, expected, atol=1e-7)
    else:
        assert np.isclose(sum(values), 0.8, atol=1e-8)
    for rotation in fit.leaf_rotations:
        assert np.isclose(rotation[0, 0], 1.0)
    assert joint.cores_init == "caller"


def test_default_initial_cores_are_the_documented_heuristic():
    n, num_leaves, rho = 3, 2, 0.1
    eri = _synthetic_eri(n, 2, 131)
    rotations = np.stack(
        [scipy.linalg.expm(g) for g in _random_skew(num_leaves, n, 132)])
    cores = J.default_initial_cores(eri, rotations, rho, np)
    for t in range(num_leaves):
        projected = F._project_eri_into_leaf(eri, rotations[t], np)
        assert np.allclose(cores[t],
                           projected / (num_leaves * (1 + 2 * rho)),
                           atol=1e-13)


# --------------------------------------------------------------------------- #
# (8) no inner solver, no per-step host conversion
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("backend", _BACKENDS)
def test_joint_strategy_never_calls_an_inner_core_solver(backend, monkeypatch):

    def forbidden(*args, **kwargs):
        raise AssertionError("inner core solver called in joint mode")

    for name in ("_solve_inner_cores", "_solve_inner_cores_cg",
                 "_solve_inner_cores_lstsq", "_lstsq"):
        monkeypatch.setattr(F, name, forbidden)
    eri = _synthetic_eri(3, 2, 141)
    fit = _joint_fit(eri,
                     2,
                     max_iterations=6,
                     tolerance=1e-12,
                     regularization=1e-3,
                     backend=backend)
    assert fit.joint.iterations == 6
    assert fit.inner_solve is None


def test_cupy_loop_makes_no_per_step_host_array_conversion(monkeypatch):
    cupy = _requires_cupy()
    counts = {"to_numpy": 0, "asnumpy": 0}
    original_to_numpy = J.to_numpy
    original_asnumpy = cupy.asnumpy

    def counting_to_numpy(array):
        counts["to_numpy"] += 1
        return original_to_numpy(array)

    def counting_asnumpy(array, *args, **kwargs):
        counts["asnumpy"] += 1
        return original_asnumpy(array, *args, **kwargs)

    monkeypatch.setattr(J, "to_numpy", counting_to_numpy)
    monkeypatch.setattr(F, "to_numpy", counting_to_numpy)
    monkeypatch.setattr(cupy, "asnumpy", counting_asnumpy)
    eri = _synthetic_eri(4, 3, 151)
    generators = _random_skew(3, 4, 152, scale=0.3)
    results = {}
    for budget in (5, 25):
        counts["to_numpy"] = counts["asnumpy"] = 0
        fit = _joint_fit(eri,
                         3,
                         max_iterations=budget,
                         tolerance=1e-14,
                         initial_generators=list(generators),
                         backend="cupy")
        assert fit.joint.iterations == budget
        results[budget] = (dict(counts), fit.joint.host_syncs)
    # Array conversions do not grow with the iteration count; scalar host
    # reads do (objective per trial, two gradient norms and a slope per
    # accepted update) and are accounted for.
    assert results[5][0] == results[25][0]
    assert results[25][1] > results[5][1]
    assert results[25][0]["asnumpy"] <= 2 * 3 + 4


# --------------------------------------------------------------------------- #
# (9) API compatibility and validation
# --------------------------------------------------------------------------- #


def test_default_strategy_is_variable_projection_and_positional_api_holds():
    eri = _synthetic_eri(3, 2, 161)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        fit = df.compressed_double_factorization(eri, 2, 50, 1.0e-8, 1.0e-3,
                                                 "cg", 1.0e-10, None, True,
                                                 None, None, "numpy")
    assert fit.optimization_strategy == "variable_projection"
    assert fit.joint is None
    assert fit.inner_solve is not None and fit.inner_solve.solver == "cg"
    explicit = df.explicit_double_factorization(eri)
    assert explicit.optimization_strategy is None and explicit.joint is None
    assert explicit.converged


def test_joint_mode_rejects_inner_solve_settings_and_vice_versa():
    eri = _synthetic_eri(3, 2, 171)
    for kwargs in (dict(inner_solver="cg"), dict(cg_preconditioner="pairwise"),
                   dict(inner_fallback="lstsq"), dict(cg_warm_start=False),
                   dict(inner_solve_history=True), dict(cg_tolerance=1e-6),
                   dict(cg_max_iterations=5),
                   dict(cg_optimization_tolerance=1e-6)):
        with pytest.raises(ValueError, match="inner-solve settings"):
            df.compressed_double_factorization(eri,
                                               2,
                                               optimization_strategy=JOINT,
                                               backend="numpy",
                                               **kwargs)
    with pytest.raises(ValueError, match="joint_options applies"):
        df.compressed_double_factorization(
            eri,
            2,
            joint_options=df.JointGradientDescentOptions(),
            backend="numpy")
    with pytest.raises(ValueError, match="initial_cores applies"):
        df.compressed_double_factorization(eri,
                                           2,
                                           initial_cores=[np.eye(3)] * 2,
                                           backend="numpy")
    with pytest.raises(ValueError, match="optimization_strategy must be"):
        df.compressed_double_factorization(eri,
                                           2,
                                           optimization_strategy="adam",
                                           backend="numpy")
    with pytest.raises(TypeError):
        df.compressed_double_factorization(eri,
                                           2,
                                           optimization_strategy=JOINT,
                                           joint_options={"initial_step": 1},
                                           backend="numpy")
    with pytest.raises(ValueError, match="regularization"):
        df.compressed_double_factorization(eri,
                                           2,
                                           regularization=-1e-3,
                                           backend="numpy")


def test_joint_options_validation():
    df.JointGradientDescentOptions().validate()
    bad = [
        dict(generator_step_scale=0.0),
        dict(core_step_scale=-1.0),
        dict(initial_step=float("inf")),
        dict(armijo_coefficient=1.0),
        dict(backtracking_factor=0.0),
        dict(max_backtracks=-1),
        dict(step_policy="random"),
        dict(core_residual_tolerance=-1e-10),
        dict(objective_tolerance=float("nan")),
        dict(history_length=-5),
        dict(rotation_adjoint="pade"),
    ]
    for kwargs in bad:
        with pytest.raises(ValueError, match="JointGradientDescentOptions"):
            df.JointGradientDescentOptions(**kwargs).validate()
    options = df.JointGradientDescentOptions(core_step_scale=0.5)
    assert options.as_dict()["core_step_scale"] == 0.5
    eri = _synthetic_eri(3, 2, 181)
    fit = _joint_fit(eri,
                     2,
                     max_iterations=2,
                     tolerance=1e-12,
                     joint_options=options)
    assert fit.joint.options == options.as_dict()
    assert fit.joint.core_step_scale == 0.5
    with pytest.raises(ValueError, match="NumPy backend only"):
        J.JointEvaluator(eri, 2, 0.0, object(), rotation_adjoint="scipy")


def test_initial_generators_and_cores_are_validated():
    n, num_leaves = 3, 2
    eri = _synthetic_eri(n, 2, 191)
    skew = _random_skew(num_leaves, n, 192)
    symmetric = _random_symmetric(num_leaves, n, 193)
    # Round-off asymmetry is cleaned, material asymmetry rejected.
    noisy = skew + 1e-13 * np.ones_like(skew)
    fit = _joint_fit(eri,
                     num_leaves,
                     max_iterations=1,
                     tolerance=1e-12,
                     initial_generators=list(noisy),
                     initial_cores=list(symmetric + 1e-13))
    assert fit.joint.generators_init == "caller"
    with pytest.raises(ValueError, match="antisymmetric"):
        _joint_fit(eri,
                   num_leaves,
                   initial_generators=list(skew + 1e-3 * np.eye(n)))
    with pytest.raises(ValueError, match="symmetric"):
        _joint_fit(eri,
                   num_leaves,
                   initial_cores=list(symmetric + 1e-3 * skew))
    with pytest.raises(ValueError, match="shape"):
        _joint_fit(eri, num_leaves, initial_cores=[np.eye(n)])
    with pytest.raises(ValueError, match="finite"):
        _joint_fit(eri,
                   num_leaves,
                   initial_cores=[np.eye(n) * np.nan] * num_leaves)
    # Indefinite symmetric cores are allowed.
    indefinite = [np.diag([1.0, -1.0, 0.0])] * num_leaves
    fit = _joint_fit(eri,
                     num_leaves,
                     max_iterations=1,
                     tolerance=1e-12,
                     initial_cores=indefinite)
    assert fit.joint.iterations == 1


def test_cupy_request_without_a_gpu_is_an_error(monkeypatch):
    monkeypatch.setattr(algorithms.double_factorization._backend,
                        "cupy_gpu_available", lambda: False)
    eri = _synthetic_eri(3, 2, 201)
    with pytest.raises(RuntimeError, match="cupy"):
        df.compressed_double_factorization(eri,
                                           2,
                                           optimization_strategy=JOINT,
                                           backend="cupy")


def test_rotation_adjoint_option_gives_the_same_fit():
    eri = _synthetic_eri(4, 3, 211)
    generators = _random_skew(2, 4, 212, scale=0.5)
    fits = [
        _joint_fit(eri,
                   2,
                   max_iterations=15,
                   tolerance=1e-12,
                   initial_generators=list(generators),
                   joint_options=df.JointGradientDescentOptions(
                       rotation_adjoint=adjoint, history_length=20))
        for adjoint in ("spectral", "scipy")
    ]
    for a, b in zip(fits[0].joint.history, fits[1].joint.history):
        assert np.isclose(a["objective"], b["objective"], rtol=1e-10)
