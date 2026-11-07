# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for explicit (X-DF) and compressed (C-DF) double factorization."""
import os
import warnings

import numpy as np
import pytest

import cudaq_algorithms as algorithms

os.environ.setdefault("OMP_NUM_THREADS", "1")

df = algorithms.double_factorization

# NumPy always. CuPy when the GPU simulator leg is nvidia*, or when the
# kernel probe passes: these tests are classical linear algebra, so a
# working GPU must still be covered on a qpp-cpu / unset dev box. A dead
# GPU on nvidia* is collected then skipped at call time (visible skip);
# a dead GPU on qpp-cpu fails the probe and is not collected.
_BACKENDS = ["numpy"]
if (os.environ.get("CUDAQ_DEFAULT_SIMULATOR", "qpp-cpu").startswith("nvidia")
        or df.cupy_gpu_available()):
    _BACKENDS.append("cupy")


@pytest.fixture(autouse=True)
def _skip_cupy_if_unusable(request):
    # Call-time skip so a device that disappears after collection is a skip
    # with a reason, not a failure.
    params = getattr(getattr(request.node, "callspec", None), "params", None)
    if not params or params.get("backend") != "cupy":
        return
    if not df.cupy_gpu_available():
        pytest.skip("CuPy/GPU is not usable")


def _synthetic_eri(n, num_vectors, seed):
    """An 8-fold-symmetric chemist-notation ERI of known low rank."""
    rng = np.random.default_rng(seed)
    leaves = rng.standard_normal((num_vectors, n, n))
    leaves = 0.5 * (leaves + leaves.transpose(0, 2, 1))
    return np.einsum("xpq,xrs->pqrs", leaves, leaves)


def _h2o_eri():
    pytest.importorskip("pyscf")
    from pyscf import ao2mo, gto, scf
    mol = gto.M(atom="O 0 0 0; H 0 0 0.957; H 0 0.926 -0.24",
                basis="sto-3g",
                verbose=0)
    mf = scf.RHF(mol).run()
    n = mf.mo_coeff.shape[1]
    return np.asarray(ao2mo.restore("s1", ao2mo.kernel(mol, mf.mo_coeff), n))


def _max_orthogonality_error(factorization):
    return max((float(np.linalg.norm(u.T @ u - np.eye(u.shape[0])))
                for u in factorization.leaf_rotations),
               default=0.0)


@pytest.mark.parametrize("backend", _BACKENDS)
def test_explicit_double_factorization_is_exact_at_full_rank(backend):
    eri = _h2o_eri()
    factorization = df.explicit_double_factorization(eri,
                                                     threshold=0.0,
                                                     backend=backend)
    assert factorization.method == "X-DF"
    assert df.factorization_error(eri, factorization) < 1.0e-9
    assert _max_orthogonality_error(factorization) < 1.0e-10
    # Cores are symmetric.
    for core in factorization.leaf_cores:
        assert np.allclose(core, core.T, atol=1.0e-12)


@pytest.mark.parametrize("backend", _BACKENDS)
def test_explicit_double_factorization_truncation_monotone(backend):
    eri = _h2o_eri()
    full = df.explicit_double_factorization(eri,
                                            threshold=0.0,
                                            backend=backend)
    coarse = df.explicit_double_factorization(eri,
                                              threshold=1.0e-2,
                                              backend=backend)
    fine = df.explicit_double_factorization(eri,
                                            threshold=1.0e-4,
                                            backend=backend)
    # More aggressive truncation -> fewer leaves and larger (or equal) error.
    assert coarse.num_leaves <= fine.num_leaves <= full.num_leaves
    assert (df.factorization_error(eri, coarse)
            >= df.factorization_error(eri, fine) - 1.0e-12)
    # max_num_leaves caps the leaf count.
    capped = df.explicit_double_factorization(eri,
                                              threshold=0.0,
                                              max_num_leaves=3,
                                              backend=backend)
    assert capped.num_leaves == 3


@pytest.mark.parametrize("backend", _BACKENDS)
def test_pivoted_cholesky_is_default_and_rank_revealing(backend):
    eri = _h2o_eri()
    n = eri.shape[0]
    factorization = df.explicit_double_factorization(eri,
                                                     threshold=1.0e-12,
                                                     backend=backend)
    # Pivoted Cholesky is the default first factorization.
    assert factorization.first_factorization == "cholesky"
    # The ERI supermatrix is PSD, so Cholesky stops at the true rank
    # (<= the symmetric-pair dimension n(n+1)/2), not n^2.
    assert factorization.num_leaves <= n * (n + 1) // 2
    assert df.factorization_error(eri, factorization) < 1.0e-8


@pytest.mark.parametrize("backend", _BACKENDS)
def test_cholesky_and_eigendecomposition_agree(backend):
    eri = _h2o_eri()
    cholesky = df.explicit_double_factorization(eri,
                                                threshold=0.0,
                                                first_factorization="cholesky",
                                                backend=backend)
    eigen = df.explicit_double_factorization(
        eri,
        threshold=0.0,
        first_factorization="eigendecomposition",
        backend=backend)
    assert eigen.first_factorization == "eigendecomposition"
    # Both reconstruct the same ERI exactly, regardless of first-factor method.
    assert df.factorization_error(eri, cholesky) < 1.0e-9
    assert df.factorization_error(eri, eigen) < 1.0e-9
    np.testing.assert_allclose(cholesky.reconstruct_eri(),
                               eigen.reconstruct_eri(),
                               atol=1.0e-9)


@pytest.mark.parametrize("backend", _BACKENDS)
def test_compressed_double_factorization_beats_explicit(backend):
    eri = _synthetic_eri(n=4, num_vectors=3, seed=7)
    for num_leaves in (1, 2):
        explicit = df.explicit_double_factorization(eri,
                                                    threshold=0.0,
                                                    max_num_leaves=num_leaves,
                                                    backend=backend)
        compressed = df.compressed_double_factorization(eri,
                                                        num_leaves=num_leaves,
                                                        max_iterations=500,
                                                        backend=backend)
        assert compressed.method == "C-DF"
        assert compressed.num_leaves == num_leaves
        # Compression is never worse than the explicit factorization.
        assert (df.factorization_error(eri, compressed)
                <= df.factorization_error(eri, explicit) + 1.0e-6)


@pytest.mark.parametrize("backend", _BACKENDS)
def test_compressed_double_factorization_exact_at_true_rank(backend):
    eri = _synthetic_eri(n=4, num_vectors=3, seed=11)
    compressed = df.compressed_double_factorization(eri,
                                                    num_leaves=3,
                                                    max_iterations=1500,
                                                    backend=backend)
    assert df.factorization_error(eri, compressed) < 1.0e-4
    assert _max_orthogonality_error(compressed) < 1.0e-9


@pytest.mark.parametrize("backend", _BACKENDS)
def test_rcdf_regularization_shrinks_cores_and_one_norm(backend):
    eri = _synthetic_eri(n=4, num_vectors=3, seed=7)
    one_body = np.array([0.4, -0.3, 0.2, -0.1])

    plain = df.compressed_double_factorization(eri,
                                               num_leaves=2,
                                               max_iterations=800,
                                               regularization=0.0,
                                               backend=backend)
    regularized = df.compressed_double_factorization(eri,
                                                     num_leaves=2,
                                                     max_iterations=800,
                                                     regularization=1.0e-2,
                                                     backend=backend)

    def core_norm(factorization):
        return sum(float(np.linalg.norm(z)) for z in factorization.leaf_cores)

    # RC-DF shrinks the cores ...
    assert core_norm(regularized) < core_norm(plain)
    # ... lowering the Hamiltonian one-norm in both conventions ...
    for convention in ("lcu", "burg"):
        assert (df.double_factorization_one_norm(regularized,
                                                 one_body,
                                                 convention=convention)
                < df.double_factorization_one_norm(plain,
                                                   one_body,
                                                   convention=convention))
    # ... at the cost of reconstruction accuracy.
    assert (df.factorization_error(eri, regularized)
            >= df.factorization_error(eri, plain) - 1.0e-9)


@pytest.mark.parametrize("backend", _BACKENDS)
@pytest.mark.parametrize("eri_scale", [1.0, 1.0e-12])
def test_compressed_matrix_free_cg_matches_lstsq(backend, eri_scale):
    # The matrix-free conjugate-gradient inner solve (RC-DF Eqs. 25-30) is an
    # alternative to the explicit design-matrix least squares; both must give
    # the same reconstruction error, with and without regularization.
    # eri_scale=1e-12 keeps ||R||_F below cg_tolerance. L-BFGS then stops at
    # the X-DF start, so that arm compares the core solve at fixed rotations.
    eri = eri_scale * _synthetic_eri(n=4, num_vectors=3, seed=11)
    eri_norm = float(np.linalg.norm(eri))
    for regularization in (0.0, 1.0e-2):
        lstsq = df.compressed_double_factorization(
            eri,
            num_leaves=2,
            max_iterations=800,
            regularization=regularization,
            inner_solver="lstsq",
            backend=backend)
        cg = df.compressed_double_factorization(eri,
                                                num_leaves=2,
                                                max_iterations=800,
                                                regularization=regularization,
                                                inner_solver="cg",
                                                backend=backend)
        assert cg.method == "C-DF"
        error_gap = abs(
            df.factorization_error(eri, cg) -
            df.factorization_error(eri, lstsq))
        # Absolute slack on O(1) integrals. Relative to ||eri|| when the
        # tensor itself is smaller than that slack.
        if eri_scale == 1.0:
            assert error_gap < 1.0e-6
        else:
            assert error_gap / eri_norm < 1.0e-6


@pytest.mark.parametrize("backend", _BACKENDS)
def test_cg_exact_at_true_rank_and_rejects_unknown_solver(backend):
    # True rank and a rejected solver name do not depend on the integral scale.
    eri = _synthetic_eri(n=4, num_vectors=3, seed=11)
    # CG is exact at the true rank, like the least-squares solver.
    exact = df.compressed_double_factorization(eri,
                                               num_leaves=3,
                                               max_iterations=1500,
                                               inner_solver="cg",
                                               backend=backend)
    assert df.factorization_error(eri, exact) < 1.0e-4
    with pytest.raises(ValueError):
        df.compressed_double_factorization(eri,
                                           num_leaves=2,
                                           inner_solver="bogus",
                                           backend=backend)


@pytest.mark.parametrize("backend", _BACKENDS)
def test_cg_accelerators_preserve_accuracy(backend):
    eri = _synthetic_eri(n=4, num_vectors=3, seed=11)
    # The warm-start + inexact-in-loop accelerators (defaults) must reach the
    # same final reconstruction as the cold-start, tight-in-loop CG: the single
    # final solve is always tightened to cg_tolerance.
    for regularization in (0.0, 1.0e-2):
        accel = df.compressed_double_factorization(
            eri,
            num_leaves=2,
            max_iterations=800,
            regularization=regularization,
            inner_solver="cg",
            backend=backend)
        plain = df.compressed_double_factorization(
            eri,
            num_leaves=2,
            max_iterations=800,
            regularization=regularization,
            inner_solver="cg",
            cg_warm_start=False,
            cg_optimization_tolerance=1.0e-10,
            backend=backend)
        assert abs(
            df.factorization_error(eri, accel) -
            df.factorization_error(eri, plain)) < 1.0e-4


def test_auto_backend_is_size_aware(monkeypatch):
    from cudaq_algorithms.double_factorization._backend import resolve_backend
    import cudaq_algorithms.double_factorization._backend as B
    # Explicit choices are honored regardless of size.
    assert resolve_backend("numpy")[1] == "numpy"

    # A tiny problem under the GPU threshold always resolves to NumPy, and
    # must not run the kernel probe (that would pin a CUDA context).
    def _must_not_probe():
        raise AssertionError("kernel probe ran for a below-threshold auto")

    monkeypatch.setattr(B, "cupy_gpu_available", _must_not_probe)
    assert resolve_backend("auto", problem_size=4, gpu_min_size=1000)[1] \
        == "numpy"
    monkeypatch.undo()
    # A large problem resolves to CuPy when a GPU is present, else NumPy.
    expected = "cupy" if df.cupy_gpu_available() else "numpy"
    assert resolve_backend("auto", problem_size=10000, gpu_min_size=1)[1] \
        == expected


def test_one_norm_conventions():
    eri = _synthetic_eri(n=4, num_vectors=3, seed=9)
    factorization = df.explicit_double_factorization(eri, threshold=0.0)
    one_body = np.array([0.5, -1.0, 0.25, 0.75])
    base = float(np.sum(np.abs(one_body)))

    lcu = df.double_factorization_one_norm(factorization,
                                           one_body,
                                           convention="lcu")
    burg = df.double_factorization_one_norm(factorization,
                                            one_body,
                                            convention="burg")
    assert lcu >= base
    assert burg >= base
    with pytest.raises(ValueError):
        df.double_factorization_one_norm(factorization,
                                         one_body,
                                         convention="bogus")


def test_reconstruct_eri_matches_helper():
    eri = _synthetic_eri(n=4, num_vectors=2, seed=3)
    factorization = df.explicit_double_factorization(eri, threshold=0.0)
    np.testing.assert_allclose(df.reconstruct_eri(factorization),
                               factorization.reconstruct_eri())


def test_modified_one_body_integrals():
    eri = _synthetic_eri(n=4, num_vectors=2, seed=5)
    one_body = np.diag([0.1, -0.2, 0.3, -0.4])
    kappa = df.modified_one_body_integrals(one_body, eri)
    expected = one_body - 0.5 * np.einsum("prqr->pq", eri)
    np.testing.assert_allclose(kappa, expected)
    assert np.allclose(kappa, kappa.T)


def test_double_factorization_one_norm_nonnegative_and_additive():
    eri = _synthetic_eri(n=4, num_vectors=3, seed=9)
    factorization = df.explicit_double_factorization(eri, threshold=0.0)
    one_body_eigenvalues = np.array([0.5, -1.0, 0.25, 0.75])
    lam = df.double_factorization_one_norm(factorization, one_body_eigenvalues)
    assert lam >= float(np.sum(np.abs(one_body_eigenvalues)))


def test_invalid_inputs_raise():
    with pytest.raises(ValueError):
        df.explicit_double_factorization(np.zeros((3, 3)))  # not rank-4
    with pytest.raises(ValueError):
        df.compressed_double_factorization(_synthetic_eri(4, 2, 1),
                                           num_leaves=0)


def _openfermion_factorize():
    """OpenFermion's explicit-DF ``factorize``, or skip if unavailable.

    The public ``openfermion.resource_estimates.df`` API is gated behind an
    optional ``jax`` dependency, but the factorization itself only needs NumPy,
    so fall back to loading the module file directly when the gate blocks the
    normal import.
    """
    pytest.importorskip("openfermion")
    try:
        from openfermion.resource_estimates import df as of_df
        if hasattr(of_df, "factorize"):
            return of_df.factorize
    except Exception:
        pass
    import importlib.util
    import openfermion
    path = os.path.join(os.path.dirname(openfermion.__file__),
                        "resource_estimates", "df", "factorize_df.py")
    if not os.path.exists(path):
        pytest.skip("OpenFermion explicit-DF reference is not available")
    spec = importlib.util.spec_from_file_location("_of_factorize_df", path)
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except Exception:
        pytest.skip("OpenFermion explicit-DF reference could not be loaded")
    return module.factorize


def test_explicit_double_factorization_matches_openfermion_reference():
    """Cross-check X-DF against OpenFermion's independent explicit-DF reference.

    Compared on the reconstructed ERI tensor (convention-independent), since the
    two implementations store leaves differently and apply different second-factor
    truncation conventions.
    """
    factorize = _openfermion_factorize()
    eri = _h2o_eri()
    n = eri.shape[0]
    reference_eri, _factors, rank, _num_eigenvectors = factorize(
        eri, thresh=1.0e-13)

    # The reference reconstructs the ERI, and its first-factor rank cannot exceed
    # the symmetric-pair dimension n(n+1)/2.
    assert np.linalg.norm(reference_eri - eri) < 1.0e-9
    assert rank <= n * (n + 1) // 2

    mine = df.explicit_double_factorization(eri, threshold=0.0)
    np.testing.assert_allclose(mine.reconstruct_eri(),
                               reference_eri,
                               atol=1.0e-9)


def test_asymmetric_eri_is_rejected():
    eri = _synthetic_eri(4, 3, seed=11)
    eri[0, 1, 2, 3] += 0.05  # break (pq|rs) == (qp|rs)
    with pytest.raises(ValueError, match="chemist symmetries"):
        df.explicit_double_factorization(eri)


def test_indefinite_eri_warns_on_cholesky_path():
    # A negative-weight leaf makes the supermatrix indefinite while keeping
    # the within-pair index symmetries intact.
    eri = _synthetic_eri(4, 3, seed=12) - 2.0 * _synthetic_eri(4, 1, seed=13)
    with pytest.warns(RuntimeWarning, match="not positive semidefinite"):
        factorization = df.explicit_double_factorization(eri)
    # The dropped negative part shows up as reconstruction error far above
    # the threshold; the eigendecomposition path handles the same input.
    assert df.factorization_error(eri, factorization) > 1.0e-3
    exact = df.explicit_double_factorization(
        eri, first_factorization="eigendecomposition")
    assert df.factorization_error(eri, exact) < 1.0e-10


def test_one_norms_match_hand_computation():
    # One leaf with a core built from hand-chosen eigenpairs:
    # V = rotation by pi/6, lambda = (2.0, -0.5), Z = V diag(lambda) V^T.
    c, s = np.cos(np.pi / 6.0), np.sin(np.pi / 6.0)
    rotation = np.array([[c, -s], [s, c]])
    core = rotation @ np.diag([2.0, -0.5]) @ rotation.T
    factorization = df.DoubleFactorization(num_orbitals=2,
                                           leaf_rotations=[np.eye(2)],
                                           leaf_cores=[core],
                                           method="X-DF")
    one_body_eigenvalues = [0.3, -0.7]  # sum |F| = 1.0

    # lcu: sum|F| + sum_{k<l}|Z_kl| + 1/4 sum_k |Z_kk|, entries by hand:
    # Z00 = 2c^2 - 0.5s^2, Z11 = 2s^2 - 0.5c^2, Z01 = 2.5cs.
    z00 = 2.0 * c**2 - 0.5 * s**2
    z11 = 2.0 * s**2 - 0.5 * c**2
    z01 = 2.5 * c * s
    expected_lcu = 1.0 + abs(z01) + 0.25 * (abs(z00) + abs(z11))
    assert df.double_factorization_one_norm(factorization,
                                            one_body_eigenvalues,
                                            convention="lcu") == pytest.approx(
                                                expected_lcu, abs=1e-12)

    # burg (gauge-fixed eigen form): 1/4 sum_i |lambda_i| (sum_k |v_ki|)^2;
    # both eigenvector columns have |c| + |s| absolute column sum.
    expected_burg = 1.0 + 0.25 * (2.0 + 0.5) * (c + s)**2
    assert df.double_factorization_one_norm(
        factorization, one_body_eigenvalues,
        convention="burg") == pytest.approx(expected_burg, abs=1e-12)


def test_modified_one_body_matches_independent_loop():
    rng = np.random.default_rng(21)
    n = 3
    one_body = rng.standard_normal((n, n))
    eri = _synthetic_eri(n, 2, seed=22)

    expected = np.empty((n, n))
    for p in range(n):
        for q in range(n):
            correction = 0.0
            for r in range(n):
                correction += eri[p, r, q, r]
            expected[p, q] = one_body[p, q] - 0.5 * correction

    np.testing.assert_allclose(df.modified_one_body_integrals(one_body, eri),
                               expected,
                               atol=1e-14)


def test_second_factor_threshold_truncates_cores():
    eri = _synthetic_eri(4, 3, seed=23)
    exact = df.explicit_double_factorization(eri)
    # Pick a threshold from the data: the median importance-weighted gamma
    # across the exact cores, so roughly half the entries get zeroed.
    weighted = []
    for core in exact.leaf_cores:
        # Cholesky-path cores are outer(gamma, gamma), so |gamma_k| is
        # sqrt of the core diagonal; the code weights |gamma_k| by
        # importance = sum_k |gamma_k|.
        gammas = np.sqrt(np.maximum(np.diag(core), 0.0))
        weighted.extend(np.sum(gammas) * gammas)
    threshold = float(np.median([w for w in weighted if w > 1.0e-12]))
    truncated = df.explicit_double_factorization(
        eri, second_factor_threshold=threshold)
    heavy = df.explicit_double_factorization(eri,
                                             second_factor_threshold=1.0e6)

    err_exact = df.factorization_error(eri, exact)
    err_truncated = df.factorization_error(eri, truncated)
    assert err_exact < 1.0e-10
    assert err_truncated > err_exact

    # Truncation zeroes small gamma entries: the (rank-one) cores keep
    # their rank but gain zero diagonal entries where gammas were dropped.
    def zero_diagonal_count(cores):
        return sum(
            int(np.sum(np.isclose(np.diag(core), 0.0, atol=1.0e-12)))
            for core in cores)

    assert zero_diagonal_count(truncated.leaf_cores) > zero_diagonal_count(
        exact.leaf_cores)
    # An absurd threshold zeroes every core entirely.
    assert all(np.allclose(core, 0.0) for core in heavy.leaf_cores)


def test_negative_definite_supermatrix_warns_with_no_leaves():
    eri = -_synthetic_eri(4, 3, seed=24)  # negative semidefinite supermatrix
    with pytest.warns(RuntimeWarning, match="not positive semidefinite"):
        factorization = df.explicit_double_factorization(eri)
    assert factorization.num_leaves == 0
    assert df.factorization_error(eri, factorization) > 1.0


def test_eigendecomposition_threshold_zero_stops_at_numerical_rank():
    eri = _synthetic_eri(5, 3, seed=25)
    factorization = df.explicit_double_factorization(
        eri, threshold=0.0, first_factorization="eigendecomposition")
    assert factorization.num_leaves == 3
    assert df.factorization_error(eri, factorization) < 1.0e-10


def test_pair_swap_asymmetry_rejected():
    eri = _synthetic_eri(4, 3, seed=26)
    eri[0, 0, 1, 1] += 0.05  # breaks (pq|rs) == (rs|pq) only
    with pytest.raises(ValueError, match="chemist symmetries"):
        df.explicit_double_factorization(eri)


@pytest.mark.parametrize("backend", _BACKENDS)
def test_second_factor_threshold_includes_first_factor_eigenvalue(backend):
    # The small second-factor mode has scaled importance
    # 100 * (sqrt(0.99) + 0.1) * 0.1 > 0.2, so it must not be truncated.
    leaf = np.diag([0.1, np.sqrt(0.99)])
    eri = 100.0 * np.einsum("pq,rs->pqrs", leaf, leaf)
    factorization = df.explicit_double_factorization(
        eri,
        threshold=0.0,
        second_factor_threshold=0.2,
        first_factorization="eigendecomposition",
        backend=backend)

    # The threshold separates the two criteria: the scale-free importance
    # of the small mode sits below 0.2 while the scaled one sits above, so
    # this test fails loudly if the |lambda_t| factor is ever dropped again.
    small_mode_importance = (0.1 + np.sqrt(0.99)) * 0.1
    assert small_mode_importance < 0.2 < 100.0 * small_mode_importance

    # Both modes must remain because both scaled importance values exceed 0.2.
    assert np.all(np.abs(np.diag(factorization.leaf_cores[0])) > 0.0)
    # Retaining both modes must reconstruct this rank-one ERI to round-off.
    assert df.factorization_error(eri, factorization) < 1.0e-12


@pytest.mark.parametrize("backend", _BACKENDS)
def test_cdf_reports_when_optimizer_hits_iteration_limit(backend):
    # max_iterations=1 cannot finish a truncated C-DF; the result must still
    # be returned, but it must be marked unsuccessful and warn. Silent
    # success here is the product bug: callers cannot tell a cap-hit from
    # a stationary point.
    eri = _synthetic_eri(4, 3, seed=7)
    with pytest.warns(RuntimeWarning, match="did not converge"):
        factorization = df.compressed_double_factorization(eri,
                                                           num_leaves=2,
                                                           max_iterations=1,
                                                           backend=backend)
    assert factorization.method == "C-DF"
    assert factorization.optimizer_success is False
    assert factorization.optimizer_nit == 1
    assert factorization.optimizer_grad_norm is not None
    assert factorization.optimizer_grad_norm > 0.0


@pytest.mark.parametrize("backend", _BACKENDS)
@pytest.mark.parametrize("n, num_vectors, num_leaves", [(4, 3, 3), (1, 1, 1)])
def test_cdf_exact_rank_is_marked_converged(backend, n, num_vectors,
                                            num_leaves):
    # At the true leaf rank the X-DF warm start is already a minimizer, so
    # L-BFGS must report success and must not warn. n == 1 has no free
    # parameters, so the identity and closed-form cores are already exact.
    eri = _synthetic_eri(n, num_vectors, seed=11)
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        factorization = df.compressed_double_factorization(
            eri, num_leaves=num_leaves, max_iterations=1500, backend=backend)
    # Status records that this stationary point was accepted.
    assert factorization.optimizer_success is True
    # Stationary, including the empty parameter vector when n == 1.
    assert factorization.optimizer_grad_norm is not None
    assert factorization.optimizer_grad_norm < 1.0e-8
    # Closed-form cores at the true rank must reproduce the ERI.
    assert df.factorization_error(eri, factorization) < 1.0e-4


@pytest.mark.parametrize("backend", _BACKENDS)
@pytest.mark.parametrize("inner_solver", ["lstsq", "cg"])
def test_one_orbital_small_eri_keeps_the_closed_form_core(
        backend, inner_solver):
    # n == 1 skips L-BFGS. Both inner solvers must return the closed-form
    # core when the ERI is scaled below cg_tolerance.
    eri = 1.0e-12 * _synthetic_eri(1, 1, seed=11)
    eri_norm = float(np.linalg.norm(eri))
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        factorization = df.compressed_double_factorization(
            eri, num_leaves=1, inner_solver=inner_solver, backend=backend)
    # The one-orbital skip reports convergence and does not warn.
    assert factorization.optimizer_success is True
    # Relative: this tensor is ~1e-12, so a zero core still passes an
    # absolute bound of 1e-4.
    assert eri_norm > 0.0
    assert df.factorization_error(eri, factorization) / eri_norm < 1.0e-6
    assert float(np.linalg.norm(factorization.leaf_cores[0])) > 0.0


def test_xdf_leaves_optimizer_status_unset():
    # optimizer_* is a C-DF field; X-DF does not run L-BFGS.
    eri = _synthetic_eri(4, 2, seed=3)
    factorization = df.explicit_double_factorization(eri, threshold=0.0)
    assert factorization.optimizer_success is None
    assert factorization.optimizer_nit is None
    assert factorization.optimizer_grad_norm is None


@pytest.mark.parametrize("backend", _BACKENDS)
@pytest.mark.parametrize("inner_solver", ["lstsq", "cg"])
def test_cdf_objective_is_a_function_of_x(backend, inner_solver, monkeypatch):
    # Replay an L-BFGS line-search backtrack through the real objective:
    # evaluate at x, at a rejected trial x+dx, then at x again. SciPy
    # requires the same x to yield the same (f, g). The default CG warm
    # start used to seed the second f(x) from the trial's cores.
    # H2O/STO-3G at 4 leaves is rank-deficient in Z, so a wrong seed
    # changes (f, g) by a visible amount; the synthetic full-rank case
    # would not expose it.
    import cudaq_algorithms.double_factorization._factorization as F
    eri = _h2o_eri()
    orig = F.scipy.optimize.minimize
    replay = {}

    def wrapped(fun, x0, **kwargs):
        x0 = np.asarray(x0, dtype=float)
        dx = 1.0e-3 * np.random.default_rng(7).standard_normal(x0.shape)
        f0, g0 = fun(x0)
        fun(x0 + dx)
        f1, g1 = fun(x0)
        replay["df"] = abs(f0 - f1)
        replay["dg"] = float(np.linalg.norm(np.asarray(g0) - np.asarray(g1)))
        options = dict(kwargs.get("options") or {})
        options["maxiter"] = 1
        kwargs = dict(kwargs)
        kwargs["options"] = options
        return orig(fun, x0, **kwargs)

    monkeypatch.setattr(F.scipy.optimize, "minimize", wrapped)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        df.compressed_double_factorization(eri,
                                           num_leaves=4,
                                           max_iterations=1,
                                           inner_solver=inner_solver,
                                           backend=backend)
    assert replay, "minimize wrapper did not run"
    # lstsq is stateless, so this is a tight identity. CG must match it
    # once the warm start is keyed by x rather than by call order.
    assert replay["df"] < 1.0e-12
    assert replay["dg"] < 1.0e-9


def _enumerating_but_dead_cupy(monkeypatch):
    # QA-6 / RT-A: getDeviceCount is fine, the first compiled kernel is not.
    import cudaq_algorithms.double_factorization._backend as B
    import cudaq_algorithms.double_factorization._factorization as F

    class _DeadCuPy:
        ndarray = type("_DeadCuPyNdArray", (), {})

        class cuda:

            class runtime:

                @staticmethod
                def getDeviceCount():
                    return 2

        @staticmethod
        def arange(*args, **kwargs):
            raise RuntimeError("cudaErrorDevicesUnavailable: CUDA-capable "
                               "device(s) is/are busy or unavailable")

    monkeypatch.setattr(B, "_cupy", _DeadCuPy)
    # n=4 is below both real crossovers; drop them so auto would pick CuPy
    # if the probe still trusted getDeviceCount alone.
    monkeypatch.setattr(F, "AUTO_GPU_MIN_ORBITALS_EXPLICIT", 0)
    monkeypatch.setattr(F, "AUTO_GPU_MIN_ORBITALS_COMPRESSED", 0)


def test_cupy_gpu_available_requires_a_working_kernel(monkeypatch):
    # Enumeration is not enough: a kernel must run.
    import cudaq_algorithms.double_factorization._backend as B
    _enumerating_but_dead_cupy(monkeypatch)
    assert B.cupy_gpu_available() is False


@pytest.mark.parametrize("kind", ["xdf", "cdf"])
def test_auto_uses_numpy_when_cupy_kernel_probe_fails(kind, monkeypatch):
    # backend="auto" must stay on NumPy when the GPU is visible but unusable.
    _enumerating_but_dead_cupy(monkeypatch)
    eri = _synthetic_eri(n=4, num_vectors=2, seed=0)
    if kind == "xdf":
        factorization = df.explicit_double_factorization(eri,
                                                         threshold=0.0,
                                                         backend="auto")
    else:
        factorization = df.compressed_double_factorization(eri,
                                                           num_leaves=1,
                                                           backend="auto")
    assert factorization.method == ("X-DF" if kind == "xdf" else "C-DF")
    assert factorization.num_leaves >= 1


def test_explicit_cupy_backend_errors_when_kernel_probe_fails(monkeypatch):
    # backend="cupy" is a hard request: a dead GPU must still raise.
    _enumerating_but_dead_cupy(monkeypatch)
    eri = _synthetic_eri(n=4, num_vectors=2, seed=0)
    with pytest.raises(
            RuntimeError,
            match=r"not usable \(not installed, no device, or the kernel "
            r"probe failed\)"):
        df.explicit_double_factorization(eri, threshold=0.0, backend="cupy")


def _core_frobenius(cores):
    return float(np.sqrt(sum(np.sum(np.asarray(core)**2) for core in cores)))


@pytest.mark.parametrize("backend", _BACKENDS)
def test_cg_inner_cores_match_lstsq_below_cg_tolerance(backend):
    # CG tolerance is relative to ||R||_F, so the reconstruction must match
    # lstsq when the projected RHS is nonzero and smaller than cg_tolerance.
    import cudaq_algorithms.double_factorization._factorization as F
    from cudaq_algorithms.double_factorization._backend import (
        resolve_backend, to_device, to_numpy)

    base = _synthetic_eri(n=4, num_vectors=3, seed=11)
    explicit = df.explicit_double_factorization(base,
                                                threshold=0.0,
                                                max_num_leaves=2,
                                                backend=backend)
    # Fixed rotations, then scale only the ERI, so both solvers see the same U.
    eri = 1.0e-12 * base
    tolerance = 1.0e-10
    xp, _ = resolve_backend(backend)
    eri_dev = to_device(eri, xp)
    rotations = [to_device(u, xp) for u in explicit.leaf_rotations]
    rhs = xp.stack(
        [F._project_eri_into_leaf(eri_dev, u, xp) for u in rotations])
    rhs_norm = float(np.sqrt(to_numpy(xp.sum(rhs * rhs))))
    # The projected RHS is nonzero and smaller than cg_tolerance.
    assert 0.0 < rhs_norm < tolerance

    cg_cores, cg_diag = F._solve_inner_cores_cg(eri_dev,
                                                rotations,
                                                xp,
                                                tolerance=tolerance)
    cg = [to_numpy(core) for core in cg_cores]
    lstsq_cores, lstsq_diag = F._solve_inner_cores_lstsq(
        eri_dev, rotations, xp)
    lstsq = [to_numpy(core) for core in lstsq_cores]
    assert cg_diag.converged and cg_diag.termination == "converged"
    assert lstsq_diag.converged and lstsq_diag.termination == "direct"

    def reconstruct(cores):
        # (J, -J) is a null vector of the two-leaf design, so the core stacks
        # are a gauge choice. The reconstructed ERI is not.
        return sum(
            np.einsum("pk,qk,kl,rl,sl->pqrs", u, u, z, u, u)
            for u, z in zip(explicit.leaf_rotations, cores))

    # CG returns a nonzero core for this RHS.
    assert _core_frobenius(cg) > 0.0
    # Same fit. The relative CG tolerance shows up in this residual.
    assert (np.linalg.norm(reconstruct(cg) - reconstruct(lstsq)) /
            np.linalg.norm(eri)) < 1.0e-8


@pytest.mark.parametrize("backend", _BACKENDS)
def test_cg_inner_cores_are_zero_for_a_zero_rhs(backend):
    # R = 0 has minimum-norm solution Z = 0, including when a warm start is set.
    import cudaq_algorithms.double_factorization._factorization as F
    from cudaq_algorithms.double_factorization._backend import (
        resolve_backend, to_device, to_numpy)

    n = 4
    num_leaves = 2
    xp, _ = resolve_backend(backend)
    eri_dev = to_device(np.zeros((n, n, n, n)), xp)
    rotations = [to_device(np.eye(n), xp) for _ in range(num_leaves)]
    guess = xp.ones((num_leaves, n, n))
    cores, diag = F._solve_inner_cores_cg(eri_dev,
                                          rotations,
                                          xp,
                                          max_iterations=1,
                                          initial_guess=guess)
    # One iteration from this guess cannot reach 0, so an exact 0 is the early return.
    for core in cores:
        assert np.allclose(to_numpy(core), 0.0)
    assert diag.converged and diag.termination == "zero_rhs"
    assert diag.iterations == 0 and diag.residual_norm == 0.0
    assert diag.threshold == 0.0 and diag.rhs_norm == 0.0


# --------------------------------------------------------------------------- #
# Inner-solve diagnostics, verified stopping, and the pairwise preconditioner
# --------------------------------------------------------------------------- #


def _inner_setup(backend, eri, num_leaves, rho=0.0):
    """Fixed X-DF rotations on ``backend``: ``(xp, eri_dev, rotations, op)``."""
    import cudaq_algorithms.double_factorization._factorization as F
    from cudaq_algorithms.double_factorization._backend import (
        resolve_backend, to_device)
    xdf = df.explicit_double_factorization(eri,
                                           threshold=0.0,
                                           max_num_leaves=num_leaves,
                                           backend="numpy")
    rotations = [np.asarray(u, dtype=float) for u in xdf.leaf_rotations]
    while len(rotations) < num_leaves:
        rotations.append(np.eye(eri.shape[0]))
    xp, _ = resolve_backend(backend)
    eri_dev = to_device(eri, xp)
    rotations_dev = [to_device(u, xp) for u in rotations]
    return xp, eri_dev, rotations_dev, F._InnerOperator(
        eri_dev, rotations_dev, xp, rho)


def _dense_operator(op):
    """Assemble ``A + 2 rho I`` column by column from the matrix-free operator."""
    from cudaq_algorithms.double_factorization._backend import to_numpy
    xp = op.xp
    size = op.num_leaves * op.n * op.n
    dense = np.zeros((size, size))
    for column in range(size):
        unit = np.zeros(size)
        unit[column] = 1.0
        z = xp.asarray(unit.reshape(op.num_leaves, op.n, op.n))
        dense[:, column] = to_numpy(op.apply(z)).reshape(-1)
    return dense


@pytest.mark.parametrize("backend", _BACKENDS)
@pytest.mark.parametrize("rho", [1.0e-2, 0.0])
def test_pairwise_preconditioner_matches_dense_block_jacobi(backend, rho):
    # Odd leaf count: two paired leaves plus one unpaired leaf kept as its
    # diagonal block alpha I. The structured inverse must equal the inverse
    # of the dense block-Jacobi matrix assembled from the operator itself
    # (rho > 0, no floor active), and must be symmetric positive definite
    # even at rho = 0 where the pair block is singular along (J, -J).
    import cudaq_algorithms.double_factorization._factorization as F
    from cudaq_algorithms.double_factorization._backend import to_numpy
    eri = _synthetic_eri(n=3, num_vectors=4, seed=2)
    xp, _, _, op = _inner_setup(backend, eri, num_leaves=3, rho=rho)
    pre = F._PairwisePreconditioner(op)
    assert len(pre.pairs) == 1 and len(pre.unpaired) == 1
    (t, u), = pre.pairs
    assert t < u

    n = op.n
    size = 3 * n * n
    dense = _dense_operator(op)
    block = np.zeros_like(dense)
    for a in (t, u):
        for b in (t, u):
            block[a * n * n:(a + 1) * n * n,
                  b * n * n:(b + 1) * n * n] = dense[a * n * n:(a + 1) * n * n,
                                                     b * n * n:(b + 1) * n * n]
    w = pre.unpaired[0]
    block[w * n * n:(w + 1) * n * n,
          w * n * n:(w + 1) * n * n] = (op.alpha * np.eye(n * n))

    rng = np.random.default_rng(0)
    vectors = rng.standard_normal((4, size))
    applied = np.stack([
        to_numpy(pre.apply(xp.asarray(v.reshape(3, n, n)))).reshape(-1)
        for v in vectors
    ])
    if rho > 0.0:
        # No eigenvalue reaches the floor: exact block-Jacobi inverse.
        assert np.min(np.linalg.eigvalsh(block)) > pre.floor
        expected = np.linalg.solve(block, vectors.T).T
        assert np.allclose(applied, expected, atol=1.0e-10)
    # Symmetric: <x, P^-1 y> == <P^-1 x, y>; positive definite on these.
    gram = vectors @ applied.T
    assert np.allclose(gram, gram.T, atol=1.0e-10)
    assert np.all(np.einsum("ij,ij->i", vectors, applied) > 0.0)
    if rho == 0.0:
        # The floored direction is the pair null vector (J, -J).
        null = np.zeros((3, n, n))
        null[t] = 1.0
        null[u] = -1.0
        assert np.linalg.norm(block @ null.reshape(-1)) < 1.0e-10
        out = to_numpy(pre.apply(xp.asarray(null))).reshape(-1)
        assert np.allclose(out, null.reshape(-1) / pre.floor, rtol=1.0e-8)


def test_pair_leaves_is_greedy_and_deterministic():
    import cudaq_algorithms.double_factorization._factorization as F
    scores = np.array([
        [0.0, 5.0, 1.0, 1.0],
        [5.0, 0.0, 9.0, 1.0],
        [1.0, 9.0, 0.0, 1.0],
        [1.0, 1.0, 1.0, 0.0],
    ])
    # Highest score (1, 2) first, then the remaining pair (0, 3).
    assert F._pair_leaves(scores) == [(1, 2), (0, 3)]
    # Ties break by index order; an odd leaf is left unpaired.
    assert F._pair_leaves(np.ones((3, 3))) == [(0, 1)]
    assert F._pair_leaves(np.ones((1, 1))) == []


@pytest.mark.parametrize("backend", _BACKENDS)
@pytest.mark.parametrize("rho", [0.0, 1.0e-3])
def test_pcg_converges_to_the_same_fit_as_plain_cg_and_lstsq(backend, rho):
    # Three leaves (one unpaired): plain CG, pairwise PCG, and lstsq must
    # all satisfy the verified criterion and give the same reconstruction.
    # At rho = 0 the cores are a gauge choice (the design has a null
    # space), so only the reconstruction is compared.
    import cudaq_algorithms.double_factorization._factorization as F
    from cudaq_algorithms.double_factorization._backend import to_numpy
    eri = _synthetic_eri(n=4, num_vectors=5, seed=9)
    xp, eri_dev, rotations, op = _inner_setup(backend, eri, 3, rho)
    results = {}
    for name, kwargs in [("cg", dict(preconditioner="none")),
                         ("pcg", dict(preconditioner="pairwise")),
                         ("pcg-replace",
                          dict(preconditioner="pairwise",
                               residual_replacement_interval=2))]:
        cores, diag = F._solve_inner_cores_cg(eri_dev,
                                              rotations,
                                              xp,
                                              rho,
                                              tolerance=1.0e-10,
                                              operator=op,
                                              **kwargs)
        assert diag.converged and diag.termination == "converged"
        assert diag.residual_norm <= diag.threshold
        assert diag.preconditioner == kwargs["preconditioner"]
        # The diagnostics' verified residual is the operator residual.
        assert np.isclose(diag.residual_norm,
                          op.residual_norm(xp.stack(cores)),
                          rtol=1.0e-6,
                          atol=1.0e-14)
        results[name] = (to_numpy(F._reconstruct_dev(rotations, cores,
                                                     xp)), diag.iterations)
    lstsq_cores, lstsq_diag = F._solve_inner_cores_lstsq(eri_dev,
                                                         rotations,
                                                         xp,
                                                         rho,
                                                         operator=op)
    assert lstsq_diag.converged and lstsq_diag.termination == "direct"
    reference = to_numpy(F._reconstruct_dev(rotations, lstsq_cores, xp))
    for name, (reconstruction, _) in results.items():
        assert np.linalg.norm(reconstruction - reference) < 1.0e-7, name
    assert results["pcg"][1] <= results["cg"][1]


@pytest.mark.parametrize("backend", _BACKENDS)
def test_pcg_handles_identical_leaves_at_zero_regularization(backend):
    # Two identical leaves make the pair block (and the system) singular in
    # every direction (W, -W); the floor keeps the preconditioner finite and
    # PCG still reaches the verified criterion. Plain CG and PCG agree on
    # the reconstruction; their cores need not agree.
    import cudaq_algorithms.double_factorization._factorization as F
    from cudaq_algorithms.double_factorization._backend import (
        resolve_backend, to_device, to_numpy)
    eri = _synthetic_eri(n=3, num_vectors=2, seed=4)
    xp, _ = resolve_backend(backend)
    eri_dev = to_device(eri, xp)
    u = np.linalg.qr(np.random.default_rng(1).standard_normal((3, 3)))[0]
    rotations = [to_device(u, xp), to_device(u.copy(), xp)]
    reconstructions = []
    for preconditioner in ("none", "pairwise"):
        cores, diag = F._solve_inner_cores_cg(eri_dev,
                                              rotations,
                                              xp,
                                              0.0,
                                              preconditioner=preconditioner)
        assert diag.converged, diag
        reconstructions.append(
            to_numpy(F._reconstruct_dev(rotations, cores, xp)))
    assert np.linalg.norm(reconstructions[0] - reconstructions[1]) < 1.0e-8


@pytest.mark.parametrize("backend", _BACKENDS)
def test_cg_reports_an_exhausted_iteration_cap(backend):
    import cudaq_algorithms.double_factorization._factorization as F
    eri = _synthetic_eri(n=4, num_vectors=5, seed=9)
    xp, eri_dev, rotations, op = _inner_setup(backend, eri, 3)
    cores, diag = F._solve_inner_cores_cg(eri_dev,
                                          rotations,
                                          xp,
                                          max_iterations=1,
                                          operator=op)
    assert diag.converged is False
    assert diag.termination == "max_iterations"
    assert diag.iterations == 1
    assert diag.residual_norm > diag.threshold > 0.0
    assert diag.fallback is None
    # The approximate cores are still returned and symmetric.
    assert len(cores) == 3
    for core in cores:
        core = np.asarray(core if xp is np else xp.asnumpy(core))
        assert np.allclose(core, core.T)


@pytest.mark.parametrize("backend", _BACKENDS)
def test_cg_verifies_the_true_residual_before_accepting(backend):
    # Feed the recurrence a slightly wrong matvec while verification uses
    # the exact one. The recursive residual then reaches the threshold on
    # schedule, but the solver must not report convergence unless the
    # verified residual agrees. Accepting the recursive residual alone was
    # the product bug.
    import cudaq_algorithms.double_factorization._factorization as F
    eri = _synthetic_eri(n=4, num_vectors=5, seed=9)
    xp, eri_dev, rotations, exact = _inner_setup(backend, eri, 3)
    bias = 1.0e-3 * exact.rhs

    class _LyingOperator(F._InnerOperator):

        def apply(self, z):
            return super().apply(z) + bias

        def residual(self, z):
            return self.rhs - super().apply(z)

    lying = _LyingOperator(eri_dev, rotations, xp, 0.0)
    cores, diag = F._solve_inner_cores_cg(eri_dev,
                                          rotations,
                                          xp,
                                          tolerance=1.0e-10,
                                          operator=lying)
    verified = exact.residual_norm(xp.stack(cores))
    assert np.isclose(diag.residual_norm, verified, rtol=1.0e-6)
    assert diag.converged == (verified <= diag.threshold)
    # With this bias the lied-to recurrence cannot actually converge.
    assert diag.converged is False
    assert diag.termination in ("numerical_failure", "max_iterations")


@pytest.mark.parametrize("backend", _BACKENDS)
def test_cg_reports_curvature_breakdown_and_numerical_failure(backend):
    import cudaq_algorithms.double_factorization._factorization as F
    eri = _synthetic_eri(n=4, num_vectors=5, seed=9)
    xp, eri_dev, rotations, op = _inner_setup(backend, eri, 3)

    class _Indefinite(F._InnerOperator):

        def apply(self, z):
            return -super().apply(z)

    cores, diag = F._solve_inner_cores_cg(eri_dev,
                                          rotations,
                                          xp,
                                          operator=_Indefinite(
                                              eri_dev, rotations, xp, 0.0))
    assert diag.termination == "curvature_breakdown"
    assert diag.converged is False and diag.iterations == 0
    assert "p^T" in diag.message

    guess = xp.full((3, 4, 4), float("nan"))
    cores, diag = F._solve_inner_cores_cg(eri_dev,
                                          rotations,
                                          xp,
                                          initial_guess=guess,
                                          operator=op)
    assert diag.termination == "numerical_failure"
    assert diag.converged is False
    assert not np.isfinite(diag.residual_norm)


@pytest.mark.parametrize("backend", _BACKENDS)
def test_zero_rhs_with_pairwise_preconditioner(backend):
    import cudaq_algorithms.double_factorization._factorization as F
    from cudaq_algorithms.double_factorization._backend import (
        resolve_backend, to_device, to_numpy)
    xp, _ = resolve_backend(backend)
    eri_dev = to_device(np.zeros((3, 3, 3, 3)), xp)
    rotations = [to_device(np.eye(3), xp) for _ in range(3)]
    cores, diag = F._solve_inner_cores_cg(eri_dev,
                                          rotations,
                                          xp,
                                          preconditioner="pairwise",
                                          initial_guess=xp.ones((3, 3, 3)))
    assert diag.termination == "zero_rhs" and diag.converged
    assert all(np.allclose(to_numpy(c), 0.0) for c in cores)


def test_invalid_inner_options_raise():
    import cudaq_algorithms.double_factorization._factorization as F
    eri = _synthetic_eri(n=3, num_vectors=2, seed=1)
    with pytest.raises(ValueError, match="cg_preconditioner"):
        df.compressed_double_factorization(eri,
                                           num_leaves=2,
                                           inner_solver="cg",
                                           cg_preconditioner="jacobi")
    with pytest.raises(ValueError, match="inner_fallback"):
        df.compressed_double_factorization(eri,
                                           num_leaves=2,
                                           inner_solver="cg",
                                           inner_fallback="svd")
    xp, eri_dev, rotations, _ = _inner_setup("numpy", eri, 2)
    with pytest.raises(ValueError, match="preconditioner"):
        F._solve_inner_cores_cg(eri_dev, rotations, xp, preconditioner="x")
    with pytest.raises(ValueError, match="inner_fallback"):
        F._solve_inner_cores(eri_dev, rotations, xp, solver="cg", fallback="x")


def test_lstsq_design_bytes():
    # n^4 rows x T n(n+1)/2 columns, float64.
    assert df.lstsq_design_bytes(4, 2) == 8 * 256 * 20
    assert df.LSTSQ_FALLBACK_MAX_BYTES == 1 << 30


@pytest.mark.parametrize("backend", _BACKENDS)
def test_cdf_reports_and_warns_on_an_unconverged_final_inner_solve(backend):
    # A one-iteration inner cap cannot solve the cores. The fit must come
    # back marked unconverged with the final-solve diagnostics, warn once
    # about the final solve and once about the in-loop misses, and never
    # warn per in-loop solve.
    eri = _synthetic_eri(n=4, num_vectors=5, seed=9)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        fit = df.compressed_double_factorization(eri,
                                                 num_leaves=3,
                                                 max_iterations=20,
                                                 inner_solver="cg",
                                                 cg_max_iterations=1,
                                                 backend=backend)
    messages = [
        str(w.message) for w in caught
        if issubclass(w.category, RuntimeWarning)
    ]
    final = [m for m in messages if "final inner core solve" in m]
    loop = [m for m in messages if "during the optimization" in m]
    assert len(final) == 1 and len(loop) == 1
    assert fit.converged is False
    assert fit.inner_solve.converged is False
    assert fit.inner_solve.termination == "max_iterations"
    assert fit.inner_solve.iterations == 1
    assert fit.inner_solve.residual_norm > fit.inner_solve.threshold
    summary = fit.inner_solve_summary
    assert summary.num_solves >= 1
    assert summary.num_unconverged == summary.num_solves
    assert summary.worst_residual_to_threshold > 1.0
    assert summary.terminations == {"max_iterations": summary.num_solves}
    assert summary.max_iterations == 1
    assert summary.total_iterations == summary.num_solves
    # Objective is recomputed with the returned (approximate) cores.
    expected = 0.5 * df.factorization_error(eri, fit)**2
    assert np.isclose(fit.objective, expected, rtol=1.0e-8)


@pytest.mark.parametrize("backend", _BACKENDS)
def test_lstsq_fallback_rescues_a_capped_cg_solve(backend):
    import cudaq_algorithms.double_factorization._factorization as F
    eri = _synthetic_eri(n=4, num_vectors=5, seed=9)
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        fit = df.compressed_double_factorization(eri,
                                                 num_leaves=3,
                                                 max_iterations=400,
                                                 regularization=1.0e-3,
                                                 inner_solver="cg",
                                                 cg_max_iterations=1,
                                                 inner_fallback="lstsq",
                                                 backend=backend)
    assert fit.converged
    assert fit.inner_solve.solver == "lstsq"
    assert fit.inner_solve.fallback == "lstsq"
    assert fit.inner_solve.iterations == 1  # the CG attempt
    assert "cg terminated with max_iterations" in fit.inner_solve.message
    assert fit.inner_solve_summary.num_unconverged == 0
    # Same answer as the direct solver throughout.
    direct = df.compressed_double_factorization(eri,
                                                num_leaves=3,
                                                max_iterations=400,
                                                regularization=1.0e-3,
                                                inner_solver="lstsq",
                                                backend=backend)
    assert np.isclose(fit.objective, direct.objective, rtol=1.0e-6)

    # Over the memory bound the fallback is skipped and says so.
    xp, eri_dev, rotations, _ = _inner_setup(backend, eri, 3)
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(F, "LSTSQ_FALLBACK_MAX_BYTES", 0)
        cores, diag = F._solve_inner_cores(eri_dev,
                                           rotations,
                                           xp,
                                           solver="cg",
                                           cg_max_iterations=1,
                                           fallback="lstsq")
    assert diag.converged is False and diag.fallback is None
    assert "lstsq fallback skipped" in diag.message


@pytest.mark.parametrize("backend", _BACKENDS)
@pytest.mark.parametrize("inner_solver", ["lstsq", "cg"])
def test_cdf_diagnostics_on_a_converged_fit(backend, inner_solver):
    eri = _synthetic_eri(n=4, num_vectors=3, seed=11)
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        fit = df.compressed_double_factorization(eri,
                                                 num_leaves=3,
                                                 max_iterations=1500,
                                                 inner_solver=inner_solver,
                                                 inner_solve_history=True,
                                                 backend=backend)
    assert fit.converged and fit.optimizer_success
    assert isinstance(fit.optimizer_status, int)
    assert isinstance(fit.optimizer_message, str) and fit.optimizer_message
    inner = fit.inner_solve
    assert inner.solver == inner_solver
    assert inner.converged and inner.residual_norm <= inner.threshold
    # Both solvers are judged by the requested tolerance.
    assert inner.tolerance == 1.0e-10
    assert np.isclose(inner.threshold, 1.0e-10 * inner.rhs_norm)
    assert inner.residual_ratio <= 1.0e-10
    assert inner.termination == ("direct"
                                 if inner_solver == "lstsq" else "converged")
    summary = fit.inner_solve_summary
    assert summary.num_unconverged == 0
    assert summary.num_solves >= 1
    # History: every in-loop solve, then the final one.
    assert len(fit.inner_solve_history) == summary.num_solves + 1
    assert fit.inner_solve_history[-1] is inner
    assert np.isclose(fit.objective,
                      0.5 * df.factorization_error(eri, fit)**2,
                      rtol=1.0e-6,
                      atol=1.0e-20)
    assert fit.objective < 1.0e-8


def test_xdf_diagnostics_are_unset_and_converged():
    eri = _synthetic_eri(4, 2, seed=3)
    fit = df.explicit_double_factorization(eri, threshold=0.0)
    assert fit.converged is True
    for name in ("optimizer_status", "optimizer_message", "objective",
                 "inner_solve", "inner_solve_summary", "inner_solve_history"):
        assert getattr(fit, name) is None


def test_inner_solve_summary_record():
    diag = df.InnerSolveDiagnostics(solver="cg",
                                    preconditioner="none",
                                    converged=False,
                                    termination="max_iterations",
                                    iterations=7,
                                    residual_norm=3.0,
                                    threshold=1.0,
                                    rhs_norm=10.0,
                                    tolerance=0.1)
    assert diag.residual_ratio == 0.3
    summary = df.InnerSolveSummary()
    summary.record(diag)
    summary.record(
        df.InnerSolveDiagnostics(solver="cg",
                                 preconditioner="none",
                                 converged=True,
                                 termination="converged",
                                 iterations=2,
                                 residual_norm=0.5,
                                 threshold=1.0,
                                 rhs_norm=10.0,
                                 tolerance=0.1))
    assert summary.num_solves == 2 and summary.num_unconverged == 1
    assert summary.total_iterations == 9 and summary.max_iterations == 7
    assert summary.worst_residual_to_threshold == 3.0
    assert summary.terminations == {"max_iterations": 1, "converged": 1}


def test_lstsq_retries_with_scipy_when_the_backend_solver_fails(monkeypatch):
    # LAPACK gelsd can fail to converge on the rank-deficient rho = 0
    # design; the solve must then come from SciPy's gelss, not an exception.
    import cudaq_algorithms.double_factorization._factorization as F
    eri = _synthetic_eri(n=4, num_vectors=5, seed=9)
    xp, eri_dev, rotations, op = _inner_setup("numpy", eri, 3)
    reference, _ = F._solve_inner_cores_lstsq(eri_dev, rotations, xp)

    def failing(*args, **kwargs):
        raise np.linalg.LinAlgError("SVD did not converge")

    monkeypatch.setattr(np.linalg, "lstsq", failing)
    cores, diag = F._solve_inner_cores_lstsq(eri_dev, rotations, xp)
    assert diag.converged and diag.termination == "direct"
    for core, expected in zip(cores, reference):
        assert np.allclose(core, expected, atol=1.0e-8)


# --------------------------------------------------------------------------- #
# Molecular reference cases
# --------------------------------------------------------------------------- #


def _h8_sto6g_chain(tmp_path):
    """Linear H8, STO-6G, 1.4 bohr spacing, RHF orbitals, via an FCIDUMP
    written by PySCF and read back by the package's own reader. Returns
    ``(mean_field, one_body, eri, core_energy)``."""
    pytest.importorskip("pyscf")
    from pyscf import gto, scf, tools
    mol = gto.M(atom=[("H", (0, 0, 1.4 * i)) for i in range(8)],
                basis="sto-6g",
                unit="Bohr",
                verbose=0)
    mf = scf.RHF(mol).run()
    assert mf.converged
    path = tmp_path / "H8_sto6g_d1.4.fcidump"
    tools.fcidump.from_scf(mf, str(path))
    one_body, eri, core = algorithms.chemistry.from_fcidump(path.read_text())
    return mf, one_body, eri, core


def test_h8_sto6g_fcidump_round_trips_through_from_pyscf(tmp_path):
    mf, one_body, eri, core = _h8_sto6g_chain(tmp_path)
    one_body_ref, eri_ref, core_ref = algorithms.chemistry.from_pyscf(mf)
    # FCIDUMP text carries ~16 significant digits.
    assert np.allclose(one_body, one_body_ref, atol=1.0e-12)
    assert np.allclose(eri, eri_ref, atol=1.0e-12)
    assert abs(core - core_ref) < 1.0e-10
    assert eri.shape == (8, 8, 8, 8)


@pytest.mark.parametrize("backend", _BACKENDS)
def test_h8_sto6g_inner_solve_reports_the_old_budget_and_meets_the_new(
        backend, tmp_path):
    # The reported failure, pinned: at zero regularization the H8 chain's
    # inner system needs more CG iterations than the system dimension
    # T n^2. With that budget the solve must say so; with the default
    # budget (CG_ITERATION_CAP_FACTOR x) and with rho = 1e-3 it must meet
    # the verified criterion.
    import cudaq_algorithms.double_factorization._factorization as F
    _, _, eri, _ = _h8_sto6g_chain(tmp_path)
    num_leaves = 8
    xp, eri_dev, rotations, _ = _inner_setup(backend, eri, num_leaves, 0.0)
    old_budget = num_leaves * 8 * 8
    _, capped = F._solve_inner_cores_cg(eri_dev,
                                        rotations,
                                        xp,
                                        0.0,
                                        tolerance=1.0e-10,
                                        max_iterations=old_budget)
    assert capped.converged is False
    assert capped.termination == "max_iterations"
    assert capped.iterations == old_budget
    assert capped.residual_norm > 10.0 * capped.threshold
    _, default = F._solve_inner_cores_cg(eri_dev,
                                         rotations,
                                         xp,
                                         0.0,
                                         tolerance=1.0e-10)
    assert default.converged, default
    assert old_budget < default.iterations <= F._default_cg_iteration_cap(
        num_leaves, 8)
    _, regularized = F._solve_inner_cores_cg(eri_dev,
                                             rotations,
                                             xp,
                                             1.0e-3,
                                             tolerance=1.0e-10)
    assert regularized.converged
    assert regularized.iterations < default.iterations


def _femoco_reiher():
    """Reiher FeMoco CAS(54e, 54o) integrals from OpenFermion's shipped
    ``eri_reiher.h5`` (read with h5py; no resource-estimate extras needed)."""
    h5py = pytest.importorskip("h5py")
    openfermion = pytest.importorskip("openfermion")
    import pathlib
    path = (pathlib.Path(openfermion.__file__).parent / "resource_estimates" /
            "integrals" / "eri_reiher.h5")
    if not path.exists():
        pytest.skip("eri_reiher.h5 not shipped with this OpenFermion")
    with h5py.File(path, "r") as f:
        return (np.asarray(f["eri"][()], dtype=float),
                np.asarray(f["h0"][()], dtype=float), float(f["ecore"][()]))


def test_femoco_reiher_integrals_are_in_convention_and_solve_reports():
    # FeMoco is the large reference case. CI only checks that the shipped
    # integrals are what double_factorization expects and that one bounded
    # inner solve on an X-DF start returns self-consistent diagnostics; no
    # convergence is asserted at this size.
    import cudaq_algorithms.double_factorization._factorization as F
    eri, one_body, core = _femoco_reiher()
    assert eri.shape == (54, 54, 54, 54) and one_body.shape == (54, 54)
    assert np.allclose(eri, eri.transpose(1, 0, 2, 3))
    assert np.allclose(eri, eri.transpose(0, 1, 3, 2))
    assert np.allclose(eri, eri.transpose(2, 3, 0, 1))
    assert core < -13000.0  # frozen-core + nuclear constant of the CAS
    xdf = df.explicit_double_factorization(eri,
                                           threshold=0.0,
                                           max_num_leaves=8,
                                           backend="numpy")
    assert xdf.num_leaves == 8
    rotations = [np.asarray(u) for u in xdf.leaf_rotations]
    for rho in (0.0, 1.0e-3):
        cores, diag = F._solve_inner_cores_cg(eri,
                                              rotations,
                                              np,
                                              rho,
                                              tolerance=1.0e-10,
                                              max_iterations=2000)
        assert diag.rhs_norm > 0.0 and diag.threshold > 0.0
        assert diag.converged == (diag.residual_norm <= diag.threshold)
        assert diag.termination in ("converged", "max_iterations")
        assert len(cores) == 8 and cores[0].shape == (54, 54)
        if rho > 0.0:
            assert diag.converged, diag


# --------------------------------------------------------------------------- #
# Review follow-ups: requested tolerance through the fallback, positional
# compatibility, centered one-body for the one-norm
# --------------------------------------------------------------------------- #


def test_direct_solve_is_judged_by_the_requested_tolerance(monkeypatch):
    # Deterministic: make the backend lstsq return a solution whose verified
    # residual is ~5e-9 relative. A 1e-10 request must report converged=False
    # (with the "expected accuracy" note); a 1e-8 request converged=True. The
    # fallback path inherits the same decision.
    import cudaq_algorithms.double_factorization._factorization as F
    eri = _synthetic_eri(n=4, num_vectors=5, seed=9)
    xp, eri_dev, rotations, op = _inner_setup("numpy", eri, 3, 1.0e-3)
    exact = np.linalg.lstsq
    rng = np.random.default_rng(0)

    def perturbed(design, target, rcond=None):
        solution = exact(design, target, rcond=rcond)[0]
        noise = rng.standard_normal(solution.shape)
        noise *= 5.0e-9 * np.linalg.norm(target) / np.linalg.norm(
            design @ noise)
        return (solution + noise, )

    monkeypatch.setattr(np.linalg, "lstsq", perturbed)
    cores, strict = F._solve_inner_cores_lstsq(eri_dev,
                                               rotations,
                                               xp,
                                               1.0e-3,
                                               tolerance=1.0e-10,
                                               operator=op)
    assert strict.tolerance == 1.0e-10
    assert strict.converged == (strict.residual_ratio <= 1.0e-10)
    assert strict.converged is False
    assert "expected accuracy" in strict.message
    rng = np.random.default_rng(0)
    _, loose = F._solve_inner_cores_lstsq(eri_dev,
                                          rotations,
                                          xp,
                                          1.0e-3,
                                          tolerance=1.0e-8,
                                          operator=op)
    assert loose.converged and loose.tolerance == 1.0e-8
    # Through the fallback the requested CG tolerance still decides.
    rng = np.random.default_rng(0)
    _, fallback = F._solve_inner_cores(eri_dev,
                                       rotations,
                                       xp,
                                       1.0e-3,
                                       solver="cg",
                                       cg_tolerance=1.0e-10,
                                       cg_max_iterations=1,
                                       fallback="lstsq")
    assert fallback.fallback == "lstsq"
    assert fallback.converged == (fallback.residual_ratio <= 1.0e-10)
    assert fallback.converged is False


def test_fallback_contract_on_a_near_singular_system():
    # The reviewer's numerical case: nearly identical leaves at rho = 0.
    # Whatever the direct solve attains, the verdict must follow the
    # requested tolerance, never a loosened one.
    import scipy.linalg
    import cudaq_algorithms.double_factorization._factorization as F
    rng = np.random.default_rng(32)
    leaves = rng.standard_normal((5, 4, 4))
    leaves = 0.5 * (leaves + leaves.transpose(0, 2, 1))
    eri = np.einsum("xpq,xrs->pqrs", leaves, leaves)
    generators = rng.standard_normal((3, 4, 4))
    generators = generators - generators.transpose(0, 2, 1)
    rotations = [scipy.linalg.expm(3.0e-6 * g) for g in generators]
    for cap in (1, None):
        _, diag = F._solve_inner_cores(eri,
                                       rotations,
                                       np,
                                       solver="cg",
                                       cg_tolerance=1.0e-10,
                                       cg_max_iterations=cap,
                                       fallback="lstsq")
        assert diag.tolerance == 1.0e-10
        assert diag.converged == (diag.residual_ratio <= 1.0e-10)


def test_compressed_double_factorization_keeps_positional_arguments():
    # The pre-existing positional order ends with initial_generators and
    # backend; the new options are keyword-only behind them.
    eri = _synthetic_eri(n=4, num_vectors=3, seed=11)
    generators = [np.zeros((4, 4)) for _ in range(3)]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        fit = df.compressed_double_factorization(eri, 3, 1, 1.0e-10, 0.0, "cg",
                                                 1.0e-10, None, True, None,
                                                 generators, "numpy")
    assert fit.method == "C-DF" and fit.num_leaves == 3
    assert fit.inner_solve.solver == "cg"
    with pytest.raises(TypeError):
        df.compressed_double_factorization(eri, 3, 1, 1.0e-10, 0.0, "cg",
                                           1.0e-10, None, True, None,
                                           generators, "numpy", "pairwise")


def test_centered_one_body_one_orbital_analytic():
    # n = 1, h = 0, (00|00) = 1: one leaf with U = 1, Z = 1. The centered
    # Hamiltonian is 1/2 (N - 1)^2 = 1/2 N^2 - N + 1/2 = -1/2 N + 1/2 for a
    # spin-summed number operator N = n_up + n_dn with N^2 = N + 2 n_up n_dn
    # ... in the encoding's bookkeeping kappa = h + (sum_l Z_kl - Z_kk / 2)
    # = 1/2, and the LCU norm is |F| + 1/4 |Z| = 0.75, not 0.25.
    eri = np.ones((1, 1, 1, 1))
    fit = df.explicit_double_factorization(eri, threshold=0.0)
    kappa = df.centered_one_body(np.zeros((1, 1)), fit)
    assert np.allclose(kappa, [[0.5]])
    assert df.double_factorization_one_norm(fit, np.linalg.eigvalsh(kappa),
                                            "lcu") == pytest.approx(0.75)
    assert df.double_factorization_one_norm(fit, np.zeros(1),
                                            "lcu") == pytest.approx(0.25)
    with pytest.raises(ValueError):
        df.centered_one_body(np.zeros((2, 2)), fit)


def test_centered_one_body_matches_the_encoding_convention():
    # Cross-check against the DoubleFactorizedEncoding example: its alpha
    # minus the identity weight equals the LCU one-norm evaluated with the
    # centered one-body spectrum of the same factorization.
    import pathlib
    import sys
    examples = (pathlib.Path(__file__).resolve().parents[2] / "docs" /
                "sphinx" / "examples" / "python")
    if not (examples / "df_encoding.py").exists():
        pytest.skip("df_encoding example not available")
    sys.path.insert(0, str(examples))
    try:
        from df_encoding import DoubleFactorizedEncoding
    except Exception as error:  # pragma: no cover - CUDA-Q kernel env
        pytest.skip(f"df_encoding import failed: {error}")
    rng = np.random.default_rng(3)
    n = 3
    one_body = rng.normal(size=(n, n))
    one_body = 0.5 * (one_body + one_body.T)
    eri = _synthetic_eri(n, 2 * n, seed=3)
    encoding = DoubleFactorizedEncoding(one_body, eri)
    factorization = encoding.factorization
    kappa = df.centered_one_body(one_body, factorization)
    lam = df.double_factorization_one_norm(factorization,
                                           np.linalg.eigvalsh(kappa), "lcu")
    identity_weight = sum(
        abs(c) for c, qubits, _ in encoding.terms if not qubits)
    assert encoding.alpha - identity_weight == pytest.approx(lam, rel=1e-12)
