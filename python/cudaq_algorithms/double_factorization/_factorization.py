# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Explicit (X-DF) and compressed (C-DF) double factorization.

Follows Cohn, Motta, and Parrish, "Quantum Filter Diagonalization with
Compressed Double-Factorized Hamiltonians," PRX Quantum 2, 040352 (2021)
(arXiv:2104.08957).

The two-electron integrals are taken in chemist notation ``(pq|rs)`` over real
spatial orbitals (8-fold symmetry), i.e. an ``(n, n, n, n)`` array ``eri`` with
``eri[p, q, r, s] == (pq|rs)``. Double factorization writes

    (pq|rs) ~= sum_t sum_kl U^t_pk U^t_qk Z^t_kl U^t_rl U^t_sl

with orthogonal "leaf" rotations ``U^t`` (the Givens-rotation fabric) and
symmetric "core" matrices ``Z^t``. X-DF obtains these from nested
eigendecompositions (each ``Z^t`` is rank one). C-DF instead minimizes the
least-squares reconstruction error, allowing general symmetric ``Z^t`` and far
fewer leaves at equal accuracy.
"""
from __future__ import annotations

import warnings
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import scipy.linalg
import scipy.optimize

from numpy.typing import ArrayLike

from ._backend import (AUTO_GPU_MIN_ORBITALS_COMPRESSED,
                       AUTO_GPU_MIN_ORBITALS_EXPLICIT, ArrayModule,
                       DeviceArray, expm_skew_symmetric_batched,
                       resolve_backend, to_device, to_numpy)


@dataclass
class InnerSolveDiagnostics:
    """Outcome of one inner core solve ``(A + 2 rho I) Z = R`` at fixed leaf
    rotations.

    ``residual_norm`` is the *verified* residual ``||R - (A + 2 rho I) Z||_F``
    of the returned (symmetrized) cores, recomputed from the operator rather
    than read off the conjugate-gradient recurrence; ``threshold`` is
    ``tolerance * ||R||_F``, and ``converged`` is exactly
    ``residual_norm <= threshold``. ``termination`` is one of

    - ``"converged"`` -- the verified residual met the threshold;
    - ``"zero_rhs"`` -- ``R == 0``, so ``Z = 0`` is returned;
    - ``"max_iterations"`` -- the iteration cap was exhausted first;
    - ``"curvature_breakdown"`` -- a CG search direction had non-positive
      curvature ``p^T (A + 2 rho I) p``;
    - ``"numerical_failure"`` -- a non-finite quantity, or the recursive
      residual passed the threshold while the verified residual stopped
      improving (stagnation).

    ``solver`` is ``"cg"`` or ``"lstsq"``; ``preconditioner`` is ``"none"``
    or ``"pairwise"``; ``fallback`` names the direct solver that produced
    the cores when the iterative solve did not converge (``None``
    otherwise), and ``message`` carries free-form detail.
    """

    solver: str
    preconditioner: str
    converged: bool
    termination: str
    iterations: int
    residual_norm: float
    threshold: float
    rhs_norm: float
    tolerance: float
    fallback: Optional[str] = None
    message: str = ""

    @property
    def residual_ratio(self) -> float:
        """``residual_norm / ||R||_F`` (``0.0`` for a zero right-hand side);
        compare against ``tolerance``."""
        return self.residual_norm / self.rhs_norm if self.rhs_norm > 0 else 0.0


@dataclass
class InnerSolveSummary:
    """Bounded summary of the inner solves performed *during* the C-DF
    optimization (the final tightened solve is reported separately as
    ``DoubleFactorization.inner_solve``).

    ``num_unconverged`` counts solves whose verified residual missed the
    in-loop threshold; ``worst_residual_to_threshold`` is the largest
    ``residual_norm / threshold`` seen (``> 1`` means a miss);
    ``terminations`` counts termination reasons.
    """

    num_solves: int = 0
    num_unconverged: int = 0
    total_iterations: int = 0
    max_iterations: int = 0
    worst_residual_to_threshold: float = 0.0
    terminations: Dict[str, int] = field(default_factory=dict)

    def record(self, diagnostics: "InnerSolveDiagnostics") -> None:
        self.num_solves += 1
        self.total_iterations += diagnostics.iterations
        self.max_iterations = max(self.max_iterations, diagnostics.iterations)
        if not diagnostics.converged:
            self.num_unconverged += 1
        if diagnostics.threshold > 0.0:
            self.worst_residual_to_threshold = max(
                self.worst_residual_to_threshold,
                diagnostics.residual_norm / diagnostics.threshold)
        self.terminations[diagnostics.termination] = self.terminations.get(
            diagnostics.termination, 0) + 1


@dataclass
class DoubleFactorization:
    """Result of a double factorization of the two-electron integrals.

    ``leaf_rotations[t]`` is the orthogonal matrix ``U^t`` (shape
    ``(num_orbitals, num_orbitals)``) and ``leaf_cores[t]`` is the symmetric
    core ``Z^t`` for leaf ``t``. ``method`` is ``"X-DF"`` or ``"C-DF"``.

    C-DF fills the optimizer and inner-solve diagnostics (all ``None`` on
    X-DF):

    - ``optimizer_success``, ``optimizer_nit``, ``optimizer_status``,
      ``optimizer_message`` come from the L-BFGS-B result;
      ``optimizer_grad_norm`` is the gradient norm at the last optimizer
      evaluation, i.e. measured with the *in-loop* (loose) inner solve.
    - ``objective`` is the fitting objective ``1/2 ||eri - recon||_F^2 +
      rho ||Z||_F^2`` recomputed with the returned cores (the final, tight
      inner solve).
    - ``inner_solve`` is the :class:`InnerSolveDiagnostics` of that final
      solve; ``inner_solve_summary`` is the bounded
      :class:`InnerSolveSummary` over the in-loop solves, and
      ``inner_solve_history`` lists every solve (in-loop then final) when
      requested.

    When ``n == 1``, no optimization runs, so ``optimizer_success`` is
    ``True``, ``optimizer_nit`` is ``0``, and ``optimizer_grad_norm`` is
    ``0.0``.
    """

    num_orbitals: int
    leaf_rotations: List[np.ndarray]
    leaf_cores: List[np.ndarray]
    method: str
    first_factorization: Optional[
        str] = None  # "cholesky" or "eigendecomposition"
    leaf_weights: Optional[
        np.ndarray] = None  # first-factor pivots/eigenvalues
    optimizer_success: Optional[bool] = None
    optimizer_nit: Optional[int] = None
    optimizer_grad_norm: Optional[float] = None
    optimizer_status: Optional[int] = None
    optimizer_message: Optional[str] = None
    objective: Optional[float] = None
    inner_solve: Optional[InnerSolveDiagnostics] = None
    inner_solve_summary: Optional[InnerSolveSummary] = None
    inner_solve_history: Optional[List[InnerSolveDiagnostics]] = None

    @property
    def num_leaves(self) -> int:
        return len(self.leaf_rotations)

    @property
    def converged(self) -> bool:
        """``True`` when every iterative stage met its criterion: the outer
        optimizer reported success and the final inner solve's verified
        residual is within its threshold. X-DF has no iterative stage and
        is always ``True``."""
        if self.optimizer_success is False:
            return False
        if self.inner_solve is not None and not self.inner_solve.converged:
            return False
        return True

    def reconstruct_eri(self) -> np.ndarray:
        """Reconstruct the ``(n, n, n, n)`` chemist-notation ERI tensor."""
        n = self.num_orbitals
        eri = np.zeros((n, n, n, n), dtype=float)
        for rotation, core in zip(self.leaf_rotations, self.leaf_cores):
            eri += np.einsum("pk,qk,kl,rl,sl->pqrs",
                             rotation,
                             rotation,
                             core,
                             rotation,
                             rotation,
                             optimize=True)
        return eri


def _sorted_symmetric_eigendecomposition(
        matrix: DeviceArray,
        xp: ArrayModule) -> tuple[DeviceArray, DeviceArray]:
    """Eigenpairs of a symmetric matrix, ordered by descending |eigenvalue|."""
    eigenvalues, eigenvectors = xp.linalg.eigh(matrix)
    order = xp.argsort(xp.abs(eigenvalues))[::-1]
    return eigenvalues[order], eigenvectors[:, order]


def _pivoted_cholesky(
        matrix: DeviceArray, threshold: float, max_rank: Optional[int],
        xp: ArrayModule) -> tuple[List[DeviceArray], List[float]]:
    """Pivoted Cholesky factorization of a symmetric positive-semidefinite
    matrix: returns ``(vectors, pivots)`` with ``matrix ~= sum_t v_t v_t^T``.

    The largest residual-diagonal entry is pivoted on at each step, so the
    factorization is rank-revealing and stops once that pivot drops to or below
    ``threshold`` (or after ``max_rank`` vectors). This is the standard
    Cholesky/density-fitting decomposition of the (PSD) electron-repulsion
    integrals; non-positive residual pivots (numerical null space) terminate it.
    """
    m = matrix.shape[0]
    residual = xp.real(xp.diag(matrix)).astype(float).copy()
    initial_max = float(xp.max(residual)) if m else 0.0
    # Rank-revealing floor: stop once the residual pivot reaches the numerical
    # null space. Without this, threshold == 0 would keep extracting vectors with
    # vanishing (eventually denormal) pivots, and dividing by sqrt(pivot) yields
    # garbage rotations.
    floor = max(float(threshold), initial_max * 1.0e-14)
    vectors: List = []
    pivots: List[float] = []
    # Track the initial diagonal too: a (near-)negative-semidefinite input
    # yields few or zero pivots, and that is exactly the case that most
    # needs the warning below.
    most_negative = min(0.0, float(xp.min(residual))) if m else 0.0
    limit = m if max_rank is None else min(int(max_rank), m)
    for _ in range(limit):
        pivot_index = int(xp.argmax(residual))
        pivot = float(residual[pivot_index])
        if pivot <= floor:
            break
        column = matrix[:, pivot_index].astype(float).copy()
        for previous in vectors:
            column = column - previous * float(previous[pivot_index])
        vector = column / xp.sqrt(xp.asarray(pivot))
        vectors.append(vector)
        pivots.append(pivot)
        residual = residual - vector * vector
        most_negative = min(most_negative, float(xp.min(residual)))
        residual = xp.where(residual > 0.0, residual, 0.0)
    negative_scale = max(initial_max, -most_negative)
    if negative_scale > 0.0 and most_negative < -1.0e-8 * negative_scale:
        warnings.warn(
            "double_factorization: the ERI supermatrix is not positive "
            f"semidefinite (residual diagonal reached {most_negative:.3e}); "
            "the pivoted-Cholesky first factorization drops the negative "
            "part, so the reconstruction error can far exceed `threshold`. "
            "Use first_factorization='eigendecomposition' for indefinite "
            "inputs.", RuntimeWarning)
    return vectors, pivots


def _second_factorization(leaf: DeviceArray, scale: ArrayLike,
                          second_factor_threshold: float,
                          xp: ArrayModule) -> tuple[np.ndarray, np.ndarray]:
    """Eigendecompose a symmetric leaf ``L = U diag(gamma) U^T`` and return
    ``(U, Z)`` with the symmetric core ``Z = scale * outer(gamma, gamma)``."""
    leaf = 0.5 * (leaf + leaf.T)
    gamma, rotation = xp.linalg.eigh(leaf)
    if second_factor_threshold > 0.0:
        importance = xp.sum(xp.abs(gamma))
        gamma = xp.where(
            xp.abs(scale) * importance * xp.abs(gamma)
            > second_factor_threshold, gamma, 0.0)
    core = scale * xp.outer(gamma, gamma)
    return to_numpy(rotation), to_numpy(core)


def _validate_eri(eri: ArrayLike) -> int:
    eri = np.asarray(eri)
    if eri.ndim != 4 or len(set(eri.shape)) != 1:
        raise ValueError("eri must be a square rank-4 tensor (n, n, n, n) in "
                         "chemist notation (pq|rs).")
    # The leaf symmetrization silently assumes the real-orbital index
    # symmetries within each pair; a violation would otherwise surface only
    # as an unexplained reconstruction error. Pair swap is included: the
    # C-DF gradient folds four residual terms into a single -4 prefactor,
    # which assumes the full 8-fold symmetry on the unsymmetrized tensor
    # (X-DF symmetrizes the supermatrix and would hide a violation; C-DF
    # would silently optimize the wrong objective).
    if not (np.allclose(eri, eri.transpose(1, 0, 2, 3))
            and np.allclose(eri, eri.transpose(0, 1, 3, 2))
            and np.allclose(eri, eri.transpose(2, 3, 0, 1))):
        raise ValueError("eri must have the real-orbital chemist symmetries "
                         "(pq|rs) == (qp|rs) == (pq|sr) == (rs|pq).")
    return eri.shape[0]


def explicit_double_factorization(
        eri: ArrayLike,
        threshold: float = 1.0e-8,
        max_num_leaves: Optional[int] = None,
        second_factor_threshold: float = 0.0,
        first_factorization: str = "cholesky",
        backend: str = "auto") -> DoubleFactorization:
    """Explicit double factorization (X-DF).

    First factorization of the ERI supermatrix ``(pq|rs) = sum_t L^t_pq L^t_rs``
    into symmetric leaves ``L^t``:

    * ``first_factorization="cholesky"`` (default) -- pivoted Cholesky of the
      positive-semidefinite ERI matrix. Rank-revealing: it keeps leaves while the
      residual-diagonal pivot exceeds ``threshold`` (the numerical null space
      terminates it), so it stops at the true factorization rank.
    * ``first_factorization="eigendecomposition"`` -- symmetric eigendecomposition
      ``(pq|rs) = sum_t lambda_t V^t_pq V^t_rs`` keeping ``|lambda_t| > threshold``.
      Required for indefinite inputs; the ERI is PSD so Cholesky is preferred.

    ``max_num_leaves`` caps the leaf count. Second factorization: each symmetric
    leaf is eigendecomposed, ``L^t = U^t diag(gamma^t) (U^t)^T``, giving the
    rank-one core ``Z^t = outer(gamma^t, gamma^t)`` (scaled by ``lambda_t`` in the
    eigendecomposition case). ``second_factor_threshold`` optionally zeros small
    ``gamma^t_k`` (importance-weighted, matching OpenFermion's convention). On
    the eigendecomposition path the importance includes ``|lambda_t|``, so both
    paths compare the same absolute quantity -- the mode's contribution to the
    core one-norm; on the Cholesky path the pivot scale is already carried by
    the leaf vector's norm.

    Returns a :class:`DoubleFactorization` with NumPy arrays.
    """
    n = _validate_eri(eri)
    xp, _ = resolve_backend(backend,
                            problem_size=n,
                            gpu_min_size=AUTO_GPU_MIN_ORBITALS_EXPLICIT)
    eri_dev = to_device(np.asarray(eri, dtype=float), xp)
    supermatrix = eri_dev.reshape(n * n, n * n)
    supermatrix = 0.5 * (supermatrix + supermatrix.T)

    rotations: List[np.ndarray] = []
    cores: List[np.ndarray] = []
    weights: List[float] = []

    if first_factorization == "cholesky":
        vectors, pivots = _pivoted_cholesky(supermatrix, threshold,
                                            max_num_leaves, xp)
        for vector, pivot in zip(vectors, pivots):
            rotation, core = _second_factorization(vector.reshape(n, n), 1.0,
                                                   second_factor_threshold, xp)
            rotations.append(rotation)
            cores.append(core)
            weights.append(pivot)
    elif first_factorization == "eigendecomposition":
        eigenvalues, eigenvectors = _sorted_symmetric_eigendecomposition(
            supermatrix, xp)
        abs_eigenvalues = to_numpy(xp.abs(eigenvalues))
        # Same relative rank floor as the Cholesky path: threshold == 0
        # stops at the numerical rank instead of emitting n^2 null leaves.
        top = float(abs_eigenvalues[0]) if abs_eigenvalues.size else 0.0
        floor = max(float(threshold), top * 1.0e-14)
        for index in range(n * n):
            if abs_eigenvalues[index] <= floor:
                break
            if max_num_leaves is not None and len(rotations) >= max_num_leaves:
                break
            lam = eigenvalues[index]
            rotation, core = _second_factorization(
                eigenvectors[:, index].reshape(n, n), lam,
                second_factor_threshold, xp)
            rotations.append(rotation)
            cores.append(core)
            weights.append(float(to_numpy(lam)))
    else:
        raise ValueError("first_factorization must be 'cholesky' or "
                         "'eigendecomposition'.")

    return DoubleFactorization(num_orbitals=n,
                               leaf_rotations=rotations,
                               leaf_cores=cores,
                               method="X-DF",
                               first_factorization=first_factorization,
                               leaf_weights=np.asarray(weights))


def _leaf_outer_columns(rotation: DeviceArray, xp: ArrayModule) -> DeviceArray:
    """Return ``A`` with ``A[:, k] = vec(u_k u_k^T)`` for rotation columns u_k."""
    n = rotation.shape[0]
    outer = xp.einsum("pk,qk->pqk", rotation, rotation)
    return outer.reshape(n * n, n)


def _project_eri_into_leaf(eri_dev: DeviceArray, rotation: DeviceArray,
                           xp: ArrayModule) -> DeviceArray:
    """R^t_kl = sum_pqrs U^t_pk U^t_qk (pq|rs) U^t_rl U^t_sl (the RHS of the inner
    normal equations / projection of the ERIs into a leaf's rotated basis)."""
    a = xp.einsum("pk,qk->pqk", rotation, rotation)
    return xp.einsum("pqk,pqrs,rsl->kl", a, eri_dev, a, optimize=True)


class _InnerOperator:
    """The inner normal-equation operator ``(A + 2 rho I)`` at fixed leaf
    rotations, batched over leaves, plus its right-hand side ``R``.

    Solving ``(A + 2 rho I) Z = R`` gives the (ridge-regularized) least-squares
    optimal symmetric cores (RC-DF Eqs. 25-30). The operator factorizes as

        A(Z)^t = sum_{t'} M_{tt'} Z^{t'} M_{tt'}^T ,
        M_{tt'} = (U^t^T U^{t'}) elementwise-squared      (Eq. 27)

    so each application is a batched ``n x n`` matrix product (a strided
    GEMM under cuBLAS) and the ERIs are contracted only once, for ``R``.
    ``M_tt = I``, so each diagonal block of ``A + 2 rho I`` is ``alpha I``
    with ``alpha = 1 + 2 rho``.
    """

    def __init__(self, eri_dev: DeviceArray, rotations_dev: DeviceArray,
                 xp: ArrayModule, regularization: float):
        self.xp = xp
        self.regularization = float(regularization)
        self.alpha = 1.0 + 2.0 * self.regularization
        rotations = rotations_dev if _is_stacked(rotations_dev) else xp.stack(
            list(rotations_dev))
        self.num_leaves, self.n = rotations.shape[0], rotations.shape[1]
        metric = xp.einsum("tpk,upl->tukl", rotations, rotations)
        self.metric = metric * metric  # (t, t', k, l)
        self.rhs = xp.stack([
            _project_eri_into_leaf(eri_dev, rotations[t], xp)
            for t in range(self.num_leaves)
        ])
        self.rhs_norm = math.sqrt(float(to_numpy(xp.sum(self.rhs * self.rhs))))

    def apply(self, z: DeviceArray) -> DeviceArray:
        out = self.xp.einsum("tukl,ulm,tunm->tkn",
                             self.metric,
                             z,
                             self.metric,
                             optimize=True)
        if self.regularization > 0.0:
            out = out + (2.0 * self.regularization) * z
        return out

    def residual(self, z: DeviceArray) -> DeviceArray:
        """``R - (A + 2 rho I) Z``."""
        return self.rhs - self.apply(z)

    def residual_norm(self, z: DeviceArray) -> float:
        residual = self.residual(z)
        return math.sqrt(float(to_numpy(self.xp.sum(residual * residual))))


def _pair_leaves(scores: np.ndarray) -> List[tuple[int, int]]:
    """Greedy disjoint pairing of leaves by descending ``scores[t, u]``
    (``t < u``), ties broken by index order. Deterministic."""
    num_leaves = scores.shape[0]
    candidates = sorted(((-float(scores[t, u]), t, u)
                         for t in range(num_leaves)
                         for u in range(t + 1, num_leaves)))
    used = [False] * num_leaves
    pairs = []
    for _, t, u in candidates:
        if not used[t] and not used[u]:
            used[t] = used[u] = True
            pairs.append((t, u))
    return pairs


class _PairwisePreconditioner:
    """Pairwise block-Jacobi preconditioner for :class:`_InnerOperator`.

    Leaves are grouped into disjoint pairs ``(t, u)`` by the overlap score
    ``||M_tu||_F^2`` and the operator restricted to a pair is kept exactly,

        P_tu [X; Y] = [alpha X + M Y M^T ;  M^T X M + alpha Y],  M = M_tu,

    while every coupling across pairs is dropped. With ``M = Q S V^T`` and
    ``X' = Q^T X Q``, ``Y' = V^T Y V``, the pair block decouples into one
    2x2 system per entry ``(i, j)`` with matrix ``[[alpha, s_i s_j],
    [s_i s_j, alpha]]`` and eigenvalues ``alpha +- s_i s_j``, so the inverse
    is applied in ``O(n^3)`` per pair (four ``n x n`` products) without
    assembling anything of size ``(T n^2)^2``. An unpaired leaf (odd ``T``)
    keeps the diagonal block ``alpha I``.

    ``M_tu`` is doubly stochastic, so ``s_i <= 1`` and
    ``alpha - s_i s_j >= 2 rho``: at ``rho = 0`` the pair block is singular
    exactly along the design's own null direction ``(J, -J)`` (``J`` the
    all-ones core), and nearly so for near-identical leaves. Eigenvalues
    below ``floor * alpha`` are raised to that value, which keeps the applied
    inverse symmetric positive definite without changing the system being
    solved.
    """

    def __init__(self, operator: _InnerOperator, floor: float = 1.0e-6):
        xp = operator.xp
        self.xp = xp
        self.alpha = operator.alpha
        self.floor = float(floor) * self.alpha
        num_leaves = operator.num_leaves
        scores = to_numpy(
            xp.sum(operator.metric * operator.metric, axis=(2, 3)))
        self.pairs = _pair_leaves(scores)
        paired = {leaf for pair in self.pairs for leaf in pair}
        self.unpaired = [t for t in range(num_leaves) if t not in paired]
        if self.pairs:
            self.first = xp.asarray([t for t, _ in self.pairs])
            self.second = xp.asarray([u for _, u in self.pairs])
            blocks = operator.metric[self.first, self.second]  # (P, n, n)
            q, singular_values, vt = xp.linalg.svd(blocks)
            self.q = q
            self.v = vt.transpose(0, 2, 1)
            ss = singular_values[:, :, None] * singular_values[:, None, :]
            self.inverse_plus = 1.0 / (self.alpha + ss)
            self.inverse_minus = 1.0 / xp.maximum(self.alpha - ss, self.floor)
        if self.unpaired:
            self.unpaired_index = xp.asarray(self.unpaired)

    def apply(self, r: DeviceArray) -> DeviceArray:
        """``P^{-1} r`` for a stacked ``(T, n, n)`` array."""
        xp = self.xp
        out = xp.empty_like(r)
        if self.pairs:
            qt = self.q.transpose(0, 2, 1)
            vt = self.v.transpose(0, 2, 1)
            x = qt @ r[self.first] @ self.q
            y = vt @ r[self.second] @ self.v
            total = (x + y) * self.inverse_plus
            difference = (x - y) * self.inverse_minus
            x = 0.5 * (total + difference)
            y = 0.5 * (total - difference)
            out[self.first] = self.q @ x @ qt
            out[self.second] = self.v @ y @ vt
        if self.unpaired:
            out[self.unpaired_index] = r[self.unpaired_index] / self.alpha
        return out


_PRECONDITIONERS = ("none", "pairwise")

# Default CG iteration cap as a multiple of the system dimension
# ``num_leaves * n^2``. Exact arithmetic needs at most the dimension; in
# float64 the ill-conditioned (small ``rho``) molecular systems lose
# conjugacy and need several times that (up to ~7x the dimension measured
# on H8/STO-3G at ``rho = 0``), so the default budget is ten times the
# dimension. Converging solves stop early; only failing ones use it all.
CG_ITERATION_CAP_FACTOR = 10


def _default_cg_iteration_cap(num_leaves: int, n: int) -> int:
    return max(50, CG_ITERATION_CAP_FACTOR * num_leaves * n * n)


def _solve_inner_cores_cg(
    eri_dev: DeviceArray,
    rotations_dev: DeviceArray,
    xp: ArrayModule,
    regularization: float = 0.0,
    tolerance: float = 1.0e-10,
    max_iterations: Optional[int] = None,
    initial_guess: Optional[DeviceArray] = None,
    preconditioner: str = "none",
    preconditioner_floor: float = 1.0e-6,
    residual_replacement_interval: Optional[int] = None,
    operator: Optional[_InnerOperator] = None,
) -> tuple[List[DeviceArray], InnerSolveDiagnostics]:
    """Matrix-free (preconditioned) conjugate-gradient solve for the
    symmetric cores ``{Z^t}``: ``(A + 2 rho I) Z = R`` with the operator of
    :class:`_InnerOperator`, never forming the ``n^4 x num_params`` design
    matrix. The cores are stacked into one ``(num_leaves, n, n)`` array so a
    matvec is a single batched contraction, and the CG scalars stay on the
    device; only the convergence and curvature guards sync to the host.

    ``tolerance`` is relative to the right-hand side: the solve is accepted
    when the *verified* residual ``||R - (A + 2 rho I) Z||_F`` of the
    symmetrized cores is at most ``tolerance * ||R||_F`` (immediately when
    ``R`` is zero). The recursively updated CG residual only triggers that
    check; if it has drifted from the true residual, the true residual is
    substituted, the recurrence restarted, and iteration continues. With
    ``residual_replacement_interval = k`` the recursive residual is also
    replaced by the true residual every ``k`` iterations (the direction
    recurrence continues; one extra matvec per replacement).

    ``max_iterations`` defaults to ``CG_ITERATION_CAP_FACTOR`` times the
    system dimension ``num_leaves * n^2``; an explicit value is used as given.
    ``initial_guess`` (a stacked ``(num_leaves, n, n)`` array) warm-starts CG.
    ``preconditioner="pairwise"`` applies :class:`_PairwisePreconditioner`
    (eigenvalue floor ``preconditioner_floor``); ``"none"`` is plain CG.
    ``operator`` reuses an already-built :class:`_InnerOperator`.

    Returns ``(cores, diagnostics)``. The cores are returned with their
    diagnostics whatever the termination reason: an exhausted iteration cap,
    curvature breakdown, or stagnation is reported, not raised, and
    ``diagnostics.converged`` says whether the criterion was met.
    """
    if preconditioner not in _PRECONDITIONERS:
        raise ValueError(
            f"preconditioner must be one of {_PRECONDITIONERS}, got "
            f"{preconditioner!r}.")
    op = operator if operator is not None else _InnerOperator(
        eri_dev, rotations_dev, xp, regularization)
    n, num_leaves = op.n, op.num_leaves
    rhs_norm = op.rhs_norm
    threshold = float(tolerance) * rhs_norm

    def diagnostics(z, iterations, termination, message=""):
        residual_norm = op.residual_norm(z)
        converged = residual_norm <= threshold
        return InnerSolveDiagnostics(
            solver="cg",
            preconditioner=preconditioner,
            converged=converged,
            termination="converged" if converged else termination,
            iterations=iterations,
            residual_norm=residual_norm,
            threshold=threshold,
            rhs_norm=rhs_norm,
            tolerance=float(tolerance),
            message=message)

    def inner_product(x, y):
        return xp.sum(x * y)  # device scalar

    def symmetrize(z):
        return 0.5 * (z + z.transpose(0, 2, 1))

    # R = 0: the (ridge or minimum-norm) solution is Z = 0 whatever the guess.
    if rhs_norm == 0.0:
        z = xp.zeros((num_leaves, n, n))
        return [z[t] for t in range(num_leaves)
                ], InnerSolveDiagnostics(solver="cg",
                                         preconditioner=preconditioner,
                                         converged=True,
                                         termination="zero_rhs",
                                         iterations=0,
                                         residual_norm=0.0,
                                         threshold=0.0,
                                         rhs_norm=0.0,
                                         tolerance=float(tolerance))

    apply_preconditioner = (_PairwisePreconditioner(op,
                                                    preconditioner_floor).apply
                            if preconditioner == "pairwise" else None)

    if initial_guess is None:
        z = xp.zeros((num_leaves, n, n))
        residual = op.rhs.copy()  # r = R - A(0)
    else:
        z = symmetrize(xp.asarray(initial_guess))
        residual = op.residual(z)

    def restart(residual):
        preconditioned = (apply_preconditioner(residual)
                          if apply_preconditioner else residual)
        return (preconditioned.copy(), inner_product(residual, preconditioned),
                inner_product(residual, residual))

    direction, residual_dot, residual_norm_sq = restart(residual)
    threshold_sq = threshold * threshold
    limit = (max_iterations if max_iterations is not None else
             _default_cg_iteration_cap(num_leaves, n))
    iterations = 0
    last_verified_sq = math.inf
    termination = None
    message = ""

    while True:
        residual_norm_sq_host = float(to_numpy(residual_norm_sq))
        if not math.isfinite(residual_norm_sq_host):
            termination, message = "numerical_failure", "non-finite residual"
            break
        if residual_norm_sq_host <= threshold_sq:
            # The recursive residual passed: verify against the operator.
            z = symmetrize(z)
            residual = op.residual(z)
            verified_sq = float(to_numpy(inner_product(residual, residual)))
            if verified_sq <= threshold_sq:
                termination = "converged"
                break
            if verified_sq >= last_verified_sq:
                termination = "numerical_failure"
                message = ("stagnation: the recursive residual met the "
                           "threshold but the verified residual did not "
                           "improve between restarts")
                break
            # Drift: continue from the true residual with a fresh recurrence.
            last_verified_sq = verified_sq
            direction, residual_dot, residual_norm_sq = restart(residual)
            continue
        if iterations >= limit:
            termination = "max_iterations"
            break
        operator_direction = op.apply(direction)
        curvature = inner_product(direction, operator_direction)
        curvature_host = float(to_numpy(curvature))
        if not math.isfinite(curvature_host):
            termination, message = "numerical_failure", "non-finite curvature"
            break
        if curvature_host <= 0.0:
            termination = "curvature_breakdown"
            message = f"p^T (A + 2 rho I) p = {curvature_host:.3e}"
            break
        step = residual_dot / curvature  # device scalar
        z = z + step * direction
        iterations += 1
        if (residual_replacement_interval
                and iterations % residual_replacement_interval == 0):
            # Periodic replacement: the true residual takes the place of the
            # recursive one; the search direction recurrence continues.
            residual = op.residual(z)
        else:
            residual = residual - step * operator_direction
        preconditioned = (apply_preconditioner(residual)
                          if apply_preconditioner else residual)
        new_dot = inner_product(residual, preconditioned)
        beta = new_dot / residual_dot  # device scalar
        direction = preconditioned + beta * direction
        residual_dot = new_dot
        residual_norm_sq = inner_product(residual, residual)

    z = symmetrize(z)
    return [z[t]
            for t in range(num_leaves)], diagnostics(z, iterations,
                                                     termination, message)


def _lstsq(design: DeviceArray, target: DeviceArray,
           xp: ArrayModule) -> DeviceArray:
    """Minimum-norm least-squares solution of ``design @ x ~= target``.

    The backend's ``lstsq`` (LAPACK ``gelsd`` under NumPy) occasionally
    fails to converge on the rank-deficient ``rho = 0`` design matrices
    (observed on H8/STO-3G at 8 leaves). The solve is then repeated on the
    host with SciPy's ``gelss`` driver, whose one-sided Jacobi-free SVD is
    slower but does not share that failure; if that also fails the error
    propagates.
    """
    try:
        return xp.linalg.lstsq(design, target, rcond=None)[0]
    except (np.linalg.LinAlgError, RuntimeError, ValueError):
        solution = scipy.linalg.lstsq(to_numpy(design),
                                      to_numpy(target),
                                      lapack_driver="gelss")[0]
        return to_device(solution, xp)


def _solve_inner_cores_lstsq(
    eri_dev: DeviceArray,
    rotations_dev: DeviceArray,
    xp: ArrayModule,
    regularization: float = 0.0,
    tolerance: float = 1.0e-10,
    operator: Optional[_InnerOperator] = None,
) -> tuple[List[DeviceArray], InnerSolveDiagnostics]:
    """Least-squares optimal symmetric cores ``{Z^t}`` for fixed rotations.

    Solves, for fixed ``U^t``,

        min_Z  1/2 || eri - sum_t U^t Z^t (U^t)^T (congruence) ||_F^2
                  + rho * sum_{t,k,l} (Z^t_kl)^2

    exactly (linear in the symmetric ``Z^t``) via an explicit design matrix. The
    ``rho`` (``regularization``) term is the RC-DF L2 penalty
    (arXiv:2212.07957, Eq. 17); it shrinks the cores and conditions the otherwise
    rank-deficient inner system. See ``_solve_inner_cores_cg`` for the matrix-free
    alternative that scales to large orbital counts.

    Returns ``(cores, diagnostics)``; the diagnostics carry the verified
    normal-equation residual of the solution against the requested
    ``tolerance * ||R||_F`` (``termination="direct"``). When that target is
    missed but the residual is within ``LSTSQ_RESIDUAL_TOLERANCE``, the
    message says the direct solve is at its expected accuracy; the
    requested tolerance is never loosened.
    """
    n = eri_dev.shape[0]
    target = eri_dev.reshape(-1)

    columns = []
    metadata = []  # (leaf, k, l) with k <= l
    for leaf_index, rotation in enumerate(rotations_dev):
        a = _leaf_outer_columns(rotation, xp)  # (n*n, n)
        for k in range(n):
            for l in range(k, n):
                column = xp.outer(a[:, k], a[:, l])
                if l != k:
                    column = column + xp.outer(a[:, l], a[:, k])
                columns.append(column.reshape(-1))
                metadata.append((leaf_index, k, l))

    design = xp.stack(columns, axis=1)

    if regularization > 0.0:
        # Augmented rows enforce the L2 penalty rho * ||Z||_F^2 (full k, l). For
        # the k <= l parameterization, off-diagonal entries appear twice in
        # ||Z||_F^2, so they carry weight 2 (diagonal weight 1). The factor 2
        # below comes from the 1/2 on the least-squares term in Eq. 17.
        weights = xp.asarray(
            [1.0 if k == l else 2.0 for (_, k, l) in metadata])
        penalty = xp.sqrt(2.0 * regularization * weights)
        design = xp.concatenate([design, xp.diag(penalty)], axis=0)
        target = xp.concatenate([target, xp.zeros(penalty.shape[0])], axis=0)

    solution = _lstsq(design, target, xp)

    cores = [xp.zeros((n, n)) for _ in rotations_dev]
    for index, (leaf_index, k, l) in enumerate(metadata):
        value = solution[index]
        cores[leaf_index][k, l] = value
        cores[leaf_index][l, k] = value

    op = operator if operator is not None else _InnerOperator(
        eri_dev, rotations_dev, xp, regularization)
    tolerance = float(tolerance)
    threshold = tolerance * op.rhs_norm
    residual_norm = op.residual_norm(xp.stack(cores))
    converged = op.rhs_norm == 0.0 or residual_norm <= threshold
    message = ""
    if (not converged
            and residual_norm <= LSTSQ_RESIDUAL_TOLERANCE * op.rhs_norm):
        message = (f"direct solve at its expected accuracy (relative "
                   f"residual {residual_norm / op.rhs_norm:.2e} <= "
                   f"{LSTSQ_RESIDUAL_TOLERANCE:g}) but above the requested "
                   f"tolerance {tolerance:g}")
    return cores, InnerSolveDiagnostics(solver="lstsq",
                                        preconditioner="none",
                                        converged=converged,
                                        termination="direct",
                                        iterations=0,
                                        residual_norm=residual_norm,
                                        threshold=threshold,
                                        rhs_norm=op.rhs_norm,
                                        tolerance=tolerance,
                                        message=message)


# Relative normal-equation residual a direct ``lstsq`` solve can be expected
# to reach: it minimizes ``||J z - eri||`` with a truncated SVD of the design
# ``J``, and on the singular ``rho = 0`` molecular systems its residual
# ``||R - A z||_F / ||R||_F`` lands around 1e-9 (measured on H6/H2O in
# STO-3G) even though the least-squares fit itself is exact to round-off.
# This is reporting context only: the convergence decision always uses the
# requested tolerance.
LSTSQ_RESIDUAL_TOLERANCE = 1.0e-8


def lstsq_design_bytes(num_orbitals: int, num_leaves: int) -> int:
    """Size in bytes of the explicit float64 design matrix the ``"lstsq"``
    inner solver assembles: ``n^4`` rows by ``num_leaves * n (n + 1) / 2``
    symmetric-core parameters."""
    n = int(num_orbitals)
    return 8 * (n**4) * (int(num_leaves) * n * (n + 1) // 2)


# Upper bound on the explicit design matrix the ``"lstsq"`` fallback may
# assemble when a CG solve does not converge (``inner_fallback="lstsq"``).
LSTSQ_FALLBACK_MAX_BYTES = 1 << 30

_INNER_SOLVERS = ("lstsq", "cg")
_INNER_FALLBACKS = ("none", "lstsq")


def _solve_inner_cores(
    eri_dev: DeviceArray,
    rotations_dev: DeviceArray,
    xp: ArrayModule,
    regularization: float = 0.0,
    solver: str = "lstsq",
    cg_tolerance: float = 1.0e-10,
    cg_max_iterations: Optional[int] = None,
    initial_guess: Optional[DeviceArray] = None,
    preconditioner: str = "none",
    fallback: str = "none",
) -> tuple[List[DeviceArray], InnerSolveDiagnostics]:
    """Dispatch the inner core solve to the explicit ``"lstsq"`` solver or the
    matrix-free ``"cg"`` solver. ``initial_guess`` warm-starts CG and is ignored
    by the direct lstsq solver. With ``fallback="lstsq"`` a CG solve that does
    not converge is redone with the direct solver when its design matrix fits
    in ``LSTSQ_FALLBACK_MAX_BYTES``; the diagnostics then report
    ``fallback="lstsq"`` (or, when it does not fit, say so in ``message``)."""
    if solver == "lstsq":
        return _solve_inner_cores_lstsq(eri_dev, rotations_dev, xp,
                                        regularization, cg_tolerance)
    if solver != "cg":
        raise ValueError("inner_solver must be 'lstsq' or 'cg'.")
    if fallback not in _INNER_FALLBACKS:
        raise ValueError(
            f"inner_fallback must be one of {_INNER_FALLBACKS}, got "
            f"{fallback!r}.")
    operator = _InnerOperator(eri_dev, rotations_dev, xp, regularization)
    cores, diag = _solve_inner_cores_cg(eri_dev,
                                        rotations_dev,
                                        xp,
                                        regularization,
                                        cg_tolerance,
                                        cg_max_iterations,
                                        initial_guess,
                                        preconditioner=preconditioner,
                                        operator=operator)
    if diag.converged or fallback == "none":
        return cores, diag
    design_bytes = lstsq_design_bytes(operator.n, operator.num_leaves)
    if design_bytes > LSTSQ_FALLBACK_MAX_BYTES:
        diag.message = (f"{diag.message}; " if diag.message else "") + (
            f"lstsq fallback skipped: design matrix {design_bytes / 1e9:.2f} "
            f"GB exceeds {LSTSQ_FALLBACK_MAX_BYTES / 1e9:.2f} GB")
        return cores, diag
    cores, direct = _solve_inner_cores_lstsq(eri_dev,
                                             rotations_dev,
                                             xp,
                                             regularization,
                                             cg_tolerance,
                                             operator=operator)
    direct.preconditioner = preconditioner
    direct.iterations = diag.iterations
    direct.fallback = "lstsq"
    direct.message = (f"cg terminated with {diag.termination} at residual "
                      f"{diag.residual_norm:.3e} (threshold "
                      f"{diag.threshold:.3e}); cores from lstsq" +
                      (f"; {direct.message}" if direct.message else ""))
    return cores, direct


def _reconstruct_dev(rotations_dev: DeviceArray, cores_dev: DeviceArray,
                     xp: ArrayModule) -> DeviceArray:
    # Stack the leaves and reconstruct in a single batched contraction (the sum
    # over leaves t is folded into the einsum), instead of a Python loop with a
    # separate n^4 einsum and accumulation per leaf.
    rotations = rotations_dev if _is_stacked(rotations_dev) else xp.stack(
        list(rotations_dev))
    cores = cores_dev if _is_stacked(cores_dev) else xp.stack(list(cores_dev))
    return xp.einsum("tpk,tqk,tkl,trl,tsl->pqrs",
                     rotations,
                     rotations,
                     cores,
                     rotations,
                     rotations,
                     optimize=True)


def _is_stacked(arrays: object) -> bool:
    """True for a single stacked ``(num_leaves, n, n)`` array, False for a list
    of ``(n, n)`` leaf arrays."""
    return hasattr(arrays, "ndim") and arrays.ndim == 3


def _skew_to_vector(generator: np.ndarray, n: int) -> np.ndarray:
    rows, cols = np.tril_indices(n, k=-1)
    return generator[rows, cols]


def _vector_to_skew(vector: DeviceArray, n: int,
                    xp: ArrayModule) -> DeviceArray:
    generator = xp.zeros((n, n))
    rows, cols = np.tril_indices(n, k=-1)
    rows = xp.asarray(rows)
    cols = xp.asarray(cols)
    generator[rows, cols] = vector
    generator[cols, rows] = -vector
    return generator


def _initial_generators(eri: ArrayLike, num_leaves: int, n: int,
                        backend: str) -> List[np.ndarray]:
    """Warm-start antisymmetric generators from a truncated X-DF (identity pads
    any missing leaves)."""
    xdf = explicit_double_factorization(eri,
                                        threshold=0.0,
                                        max_num_leaves=num_leaves,
                                        backend=backend)
    generators = []
    for rotation in xdf.leaf_rotations:
        rotation = np.asarray(rotation, dtype=float).copy()
        # exp(antisymmetric) is always special orthogonal (det +1), but eigh
        # leaf rotations can be reflections (det -1), which have no real
        # antisymmetric logarithm. Flipping one column flips the determinant
        # without changing u_k u_k^T (hence without changing the factorization),
        # so force det +1 before taking the logarithm.
        if np.linalg.det(rotation) < 0.0:
            rotation[:, 0] *= -1.0
        skew = scipy.linalg.logm(rotation).real
        skew = 0.5 * (skew - skew.T)  # clean numerical asymmetry
        # logm sits on a branch cut for rotations with a (-1, -1)
        # eigenvalue pair (det +1), where taking .real silently yields
        # exp(skew) != rotation. The generators are only a warm start, so
        # fall back to a neutral zero generator instead of seeding the
        # optimizer with a wrong one.
        if not np.allclose(scipy.linalg.expm(skew), rotation, atol=1.0e-8):
            skew = np.zeros_like(skew)
        generators.append(skew)
    while len(generators) < num_leaves:
        generators.append(np.zeros((n, n)))
    return generators[:num_leaves]


def compressed_double_factorization(
        eri: ArrayLike,
        num_leaves: int,
        max_iterations: int = 2000,
        tolerance: float = 1.0e-10,
        regularization: float = 0.0,
        inner_solver: str = "lstsq",
        cg_tolerance: float = 1.0e-10,
        cg_max_iterations: Optional[int] = None,
        cg_warm_start: bool = True,
        cg_optimization_tolerance: Optional[float] = None,
        initial_generators: Optional[List[np.ndarray]] = None,
        backend: str = "auto",
        *,
        cg_preconditioner: str = "none",
        inner_fallback: str = "none",
        inner_solve_history: bool = False) -> DoubleFactorization:
    """Compressed double factorization (C-DF) by least-squares optimization.

    Minimizes ``O = 1/2 || eri - sum_t U^t Z^t (U^t)^T (congruence) ||_F^2`` over
    a fixed number of leaves. Uses the two-step scheme of arXiv:2104.08957: the
    leaf rotations are parameterized as ``U^t = exp(X^t)`` with antisymmetric
    ``X^t`` and optimized with L-BFGS (warm-started from X-DF), while the
    symmetric cores ``Z^t`` are solved exactly in closed form at each step.

    ``regularization`` (``rho``) enables RC-DF (arXiv:2212.07957, Eq. 17): the
    L2 penalty ``rho * sum_{t,k,l} (Z^t_kl)^2`` is added to the objective and,
    crucially, folded into the inner core solve as a ridge term. It shrinks the
    cores -- lowering the Hamiltonian one-norm ``lambda`` and the measurement
    variance -- and conditions the inner system. ``rho`` is an absolute
    coefficient (its useful scale depends on the integral magnitude; the paper
    uses ~1e-6 to 1e-3). ``rho = 0`` reproduces plain C-DF.

    ``inner_solver`` selects how the cores are solved each step: ``"lstsq"``
    (default) forms an explicit design matrix, while ``"cg"`` is the matrix-free
    conjugate-gradient solve (RC-DF Eqs. 25-30) that avoids the ``n^4``-row design
    matrix and scales to large orbital counts. ``cg_tolerance`` and
    ``cg_max_iterations`` control the CG solve. ``cg_tolerance`` is relative
    to the projected right-hand-side norm ``||R||_F``;
    ``cg_optimization_tolerance`` inherits the same meaning.

    For ``inner_solver="cg"`` two accelerators cut the per-step CG cost without
    changing the final accuracy: ``cg_warm_start`` (default ``True``) seeds each
    *new* L-BFGS evaluation's CG from the previous evaluation's cores -- which
    move slowly between steps -- while a re-evaluation of the same parameters
    (a rejected line-search point) reuses cores from a short LRU of recent
    solves so ``(f, g)`` stays a function of ``x``. ``cg_optimization_tolerance``
    (default ``max(cg_tolerance, 1e-6)``) solves the *in-loop* systems only
    loosely (an inexact inner solve; the gradient need only be approximate by
    the envelope theorem) while the single final solve is tightened to
    ``cg_tolerance``. ``cg_preconditioner`` selects ``"none"`` (plain CG) or
    ``"pairwise"`` (pairwise block-Jacobi preconditioning, which captures the
    coupling between the most-overlapping leaf pairs exactly). With
    ``inner_fallback="lstsq"`` a CG solve that does not reach its tolerance is
    redone with the direct solver when the explicit design matrix fits in
    ``LSTSQ_FALLBACK_MAX_BYTES``; the default ``"none"`` keeps the CG result
    and reports it.

    Every inner solve verifies its residual against the operator (see
    :class:`InnerSolveDiagnostics`), and a solve that misses its tolerance is
    reported, not hidden. The returned :class:`DoubleFactorization` carries
    the L-BFGS-B status (``optimizer_success``, ``optimizer_nit``,
    ``optimizer_status``, ``optimizer_message``, ``optimizer_grad_norm``), the
    objective recomputed with the final cores (``objective``), the final
    inner solve's diagnostics (``inner_solve``), and a bounded summary of the
    in-loop solves (``inner_solve_summary``); ``inner_solve_history=True``
    additionally keeps every solve's diagnostics. ``converged`` combines the
    outer and final-inner status.

    Warning policy: at most three ``RuntimeWarning`` s per call -- one if the
    optimizer does not converge (including when it hits ``max_iterations``),
    one if the final inner solve misses ``cg_tolerance``, and one summarizing
    any in-loop solves that missed ``cg_optimization_tolerance``. Individual
    in-loop misses never warn. In every case the best factorization found is
    still returned. When ``n == 1``, there are no free antisymmetric
    parameters, so each leaf rotation is the identity, the cores are the
    closed-form solution, and ``optimizer_nit`` is 0.

    Returns a :class:`DoubleFactorization` with NumPy arrays.
    """
    n = _validate_eri(eri)
    if num_leaves < 1:
        raise ValueError("num_leaves must be >= 1.")
    if inner_solver not in _INNER_SOLVERS:
        raise ValueError("inner_solver must be 'lstsq' or 'cg'.")
    if cg_preconditioner not in _PRECONDITIONERS:
        raise ValueError(
            f"cg_preconditioner must be one of {_PRECONDITIONERS}, got "
            f"{cg_preconditioner!r}.")
    if inner_fallback not in _INNER_FALLBACKS:
        raise ValueError(
            f"inner_fallback must be one of {_INNER_FALLBACKS}, got "
            f"{inner_fallback!r}.")
    xp, _ = resolve_backend(backend,
                            problem_size=n,
                            gpu_min_size=AUTO_GPU_MIN_ORBITALS_COMPRESSED)
    eri_dev = to_device(np.asarray(eri, dtype=float), xp)

    if initial_generators is None:
        initial_generators = _initial_generators(eri, num_leaves, n, backend)

    x0 = np.concatenate(
        [_skew_to_vector(np.asarray(g), n) for g in initial_generators])

    per_leaf = n * (n - 1) // 2
    lower = np.tril_indices(n, k=-1)

    # Inexact in-loop CG (forcing sequence) + warm start across L-BFGS steps.
    # recent is a 3-slot LRU of (parameter-vector, cores) so a line-search
    # backtrack to the same x does not reuse a rejected trial's Z, without
    # pinning a cores-stack per distinct x (device memory at production n).
    use_warm = inner_solver == "cg" and cg_warm_start
    loop_tolerance = (cg_optimization_tolerance if cg_optimization_tolerance
                      is not None else max(cg_tolerance, 1.0e-6))
    warm_state = {"z": None, "recent": []}
    warm_lru = 3
    summary = InnerSolveSummary()
    history: Optional[List[InnerSolveDiagnostics]] = ([] if inner_solve_history
                                                      else None)

    def solve_cores(rotations, inner_tolerance, guess):
        cores, diag = _solve_inner_cores(eri_dev,
                                         rotations,
                                         xp,
                                         regularization,
                                         inner_solver,
                                         inner_tolerance,
                                         cg_max_iterations,
                                         guess,
                                         preconditioner=cg_preconditioner,
                                         fallback=inner_fallback)
        if history is not None:
            history.append(diag)
        return cores, diag

    def unpack(parameter_vector):
        # Build all leaf generators, then exponentiate them in one batched
        # Hermitian eigendecomposition (single cuSOLVER call on GPU). Returns
        # stacked (num_leaves, n, n) arrays.
        skews = xp.stack([
            _vector_to_skew(
                to_device(parameter_vector[i * per_leaf:(i + 1) * per_leaf],
                          xp), n, xp) for i in range(num_leaves)
        ])
        rotations = expm_skew_symmetric_batched(skews, xp)
        return skews, rotations

    def objective_and_gradient(parameter_vector):
        # Two-step objective O(X) with the cores Z at their (regularized)
        # least-squares optimum. Because dO/dZ = 0 there, the envelope theorem
        # gives dO/dX = (dO/dU at fixed Z) chained through dU/dX; the L2 penalty
        # has no explicit X-dependence, so the gradient keeps the same form for
        # any regularization. dO/dU^t_ak = -4 sum_qrsl Delta_aqrs U_qk Z_kl U_rl
        # U_sl (Eq. 17 of arXiv:2104.08957).
        skews, rotations = unpack(parameter_vector)
        if use_warm:
            key = np.asarray(parameter_vector, dtype=float).tobytes()
            cores = None
            for index, (cached_key,
                        cached_cores) in enumerate(warm_state["recent"]):
                if cached_key == key:
                    cores = cached_cores
                    warm_state["recent"].append(
                        warm_state["recent"].pop(index))
                    break
            if cores is None:
                cores, diag = solve_cores(rotations, loop_tolerance,
                                          warm_state["z"])
                summary.record(diag)
                warm_state["z"] = xp.stack(cores)
                warm_state["recent"].append((key, cores))
                if len(warm_state["recent"]) > warm_lru:
                    del warm_state["recent"][0]
        else:
            cores, diag = solve_cores(rotations, loop_tolerance, None)
            summary.record(diag)
        residual = eri_dev - _reconstruct_dev(rotations, cores, xp)
        loss = 0.5 * xp.sum(residual * residual)
        if regularization > 0.0:
            loss = loss + regularization * sum(
                xp.sum(core * core) for core in cores)
        loss = float(to_numpy(loss))

        # dO/dU for all leaves in one batched contraction, then a single
        # device->host transfer (the leaf index t is the batch axis).
        cores_dev = cores if _is_stacked(cores) else xp.stack(list(cores))
        grad_u_all = to_numpy(-4.0 * xp.einsum("aqrs,tqk,tkl,trl,tsl->tak",
                                               residual,
                                               rotations,
                                               cores_dev,
                                               rotations,
                                               rotations,
                                               optimize=True))
        skews_host = to_numpy(skews)
        gradient_chunks = []
        for leaf_index in range(num_leaves):
            # Adjoint of d(exp)/dX: grad_X = L_exp(X^T, grad_U) (Frechet
            # derivative), then project onto the antisymmetric parameters. The
            # Frechet step stays on the host (it is ~2% of the step cost).
            grad_x = scipy.linalg.expm_frechet(skews_host[leaf_index].T,
                                               grad_u_all[leaf_index],
                                               compute_expm=False)
            grad_x = grad_x - grad_x.T
            gradient_chunks.append(grad_x[lower])
        return loss, np.concatenate(gradient_chunks)

    if per_leaf == 0:
        # No antisymmetric parameters: every leaf rotation is the identity.
        # The cores are the closed-form solve at the identity rotations.
        solved_x = np.zeros(0)
        optimizer_success = True
        optimizer_nit = 0
        grad_norm = 0.0
        optimizer_status = 0
        optimizer_message = "no free parameters (n == 1)"
    else:
        result = scipy.optimize.minimize(objective_and_gradient,
                                         x0,
                                         method="L-BFGS-B",
                                         jac=True,
                                         options={
                                             "maxiter": max_iterations,
                                             "ftol": tolerance,
                                             "gtol": tolerance,
                                         })

        grad_norm = (float(np.linalg.norm(np.asarray(result.jac))) if getattr(
            result, "jac", None) is not None else None)
        if not result.success:
            extra = (f", ||g||={grad_norm:.3e}"
                     if grad_norm is not None else "")
            warnings.warn(
                "double_factorization: compressed_double_factorization did "
                f"not converge (status={result.status}, nit={result.nit}/"
                f"{max_iterations}, nfev={result.nfev}{extra}): "
                f"{result.message}", RuntimeWarning)
        solved_x = result.x
        optimizer_success = bool(result.success)
        optimizer_nit = int(result.nit)
        optimizer_status = int(result.status)
        optimizer_message = str(result.message)

    # Final cores at the tight cg_tolerance (warm-started from the last step).
    _, rotations = unpack(solved_x)
    cores, final_diag = solve_cores(rotations, cg_tolerance,
                                    warm_state["z"] if use_warm else None)
    if not final_diag.converged:
        warnings.warn(
            "double_factorization: the final inner core solve did not reach "
            f"cg_tolerance={cg_tolerance:g} (solver={final_diag.solver}, "
            f"preconditioner={final_diag.preconditioner}, termination="
            f"{final_diag.termination}, iterations={final_diag.iterations}, "
            f"verified residual {final_diag.residual_norm:.3e} > threshold "
            f"{final_diag.threshold:.3e}"
            f"{'; ' + final_diag.message if final_diag.message else ''}); "
            "the returned cores are approximate -- see "
            "DoubleFactorization.inner_solve.", RuntimeWarning)
    if summary.num_unconverged > 0:
        warnings.warn(
            f"double_factorization: {summary.num_unconverged} of "
            f"{summary.num_solves} inner core solves during the optimization "
            f"missed cg_optimization_tolerance={loop_tolerance:g} (worst "
            f"residual/threshold {summary.worst_residual_to_threshold:.3g}, "
            f"terminations {summary.terminations}); the optimizer path may "
            "be inexact -- see DoubleFactorization.inner_solve_summary.",
            RuntimeWarning)

    cores_dev = xp.stack(cores)
    fit_residual = eri_dev - _reconstruct_dev(rotations, cores_dev, xp)
    objective = 0.5 * xp.sum(fit_residual * fit_residual)
    if regularization > 0.0:
        objective = objective + regularization * xp.sum(cores_dev * cores_dev)
    return DoubleFactorization(num_orbitals=n,
                               leaf_rotations=[to_numpy(r) for r in rotations],
                               leaf_cores=[to_numpy(c) for c in cores],
                               method="C-DF",
                               optimizer_success=optimizer_success,
                               optimizer_nit=optimizer_nit,
                               optimizer_grad_norm=grad_norm,
                               optimizer_status=optimizer_status,
                               optimizer_message=optimizer_message,
                               objective=float(to_numpy(objective)),
                               inner_solve=final_diag,
                               inner_solve_summary=summary,
                               inner_solve_history=history)


def reconstruct_eri(factorization: DoubleFactorization) -> np.ndarray:
    """Reconstruct the chemist-notation ERI tensor from a factorization."""
    return factorization.reconstruct_eri()


def factorization_error(eri: ArrayLike,
                        factorization: DoubleFactorization) -> float:
    """Frobenius norm of the ERI reconstruction residual."""
    return float(
        np.linalg.norm(
            np.asarray(eri, dtype=float) - factorization.reconstruct_eri()))


def modified_one_body_integrals(one_body: ArrayLike,
                                eri: ArrayLike) -> np.ndarray:
    """Return the DF-corrected one-body matrix ``kappa_pq = h_pq - 1/2 sum_r
    (pr|qr)`` (Eq. 3 of arXiv:2104.08957), used when assembling the full
    double-factorized Hamiltonian from the two-body factorization."""
    one_body = np.asarray(one_body, dtype=float)
    eri = np.asarray(eri, dtype=float)
    return one_body - 0.5 * np.einsum("prqr->pq", eri, optimize=True)


def centered_one_body(one_body: ArrayLike,
                      factorization: DoubleFactorization) -> np.ndarray:
    """One-body matrix ``kappa`` of the centered double-factorized
    Hamiltonian of ``factorization``.

    The DF Hamiltonian is written (von Burg et al., arXiv:2007.14460) as
    ``const + sum_k F_k N_k + 1/2 sum_t sum_kl Z^t_kl (N^t_k - 1)(N^t_l - 1)``
    with ``F`` the eigenvalues of ``kappa = h + sum_t U^t diag(sum_l Z^t_kl
    - 1/2 Z^t_kk) (U^t)^T``: the raw one-body integrals ``h`` plus the
    exchange correction ``-1/2 sum_r (pr|rq)`` *of the factorized tensor*
    and the one-body remainder from centering the leaf number operators.
    The eigenvalues of this matrix are the ``one_body_eigenvalues`` that
    :func:`double_factorization_one_norm` expects; the bare eigenvalues of
    ``h`` give a different (and for ``"lcu"`` smaller) number. A
    non-symmetric user-built core is symmetrized as the encoding does.
    """
    kappa = np.array(one_body, dtype=float, copy=True)
    if kappa.ndim != 2 or kappa.shape[0] != kappa.shape[1]:
        raise ValueError("one_body must be a square (n, n) matrix.")
    if kappa.shape[0] != factorization.num_orbitals:
        raise ValueError("one_body and factorization have different orbital "
                         "counts.")
    for rotation, core in zip(factorization.leaf_rotations,
                              factorization.leaf_cores):
        rotation = np.asarray(rotation, dtype=float)
        core = np.asarray(core, dtype=float)
        absorbed = (0.5 * (core.sum(axis=1) + core.sum(axis=0)) -
                    0.5 * np.diag(core))
        kappa += (rotation * absorbed) @ rotation.T
    return kappa


def double_factorization_one_norm(factorization: DoubleFactorization,
                                  one_body_eigenvalues: ArrayLike,
                                  convention: str = "lcu") -> float:
    """One-norm ``lambda`` of the double-factorized Hamiltonian (RC-DF,
    arXiv:2212.07957), used to assess factorization quality.

    ``convention="lcu"`` (Eq. 13) -- the LCU / Pauli-rotation norm
    ``sum_k |F_k| + sum_t (sum_{k<l} |Z^t_kl| + 1/4 sum_k |Z^t_kk|)``.

    ``convention="burg"`` -- the qubitization norm in the standard
    von Burg/Lee form: with each core eigendecomposed as
    ``Z^t = sum_i lambda^t_i v^t_i (v^t_i)^T``,
    ``sum_k |F_k| + 1/4 sum_t sum_i |lambda^t_i| (sum_k |v^t_ki|)^2``.
    (A factorization ``Z = W W^T`` leaves ``W`` free up to a right
    orthogonal gauge, and the column-norm formula is not gauge
    invariant; the eigenfactor is the standard, gauge-fixed choice and
    reduces to RC-DF Eq. 15 / von Burg's ``(1/4)(sum_k |gamma_k|)^2``
    for rank-one cores.)

    ``one_body_eigenvalues`` are the eigenvalues of the centered one-body
    matrix ``kappa`` of this factorization -- see :func:`centered_one_body`
    -- not of the bare one-body integrals.
    """
    one_body_eigenvalues = np.asarray(one_body_eigenvalues, dtype=float)
    lam = float(np.sum(np.abs(one_body_eigenvalues)))

    if convention == "lcu":
        for core in factorization.leaf_cores:
            core = np.asarray(core)
            # sum_{k<l} |Z_kl| (strict upper triangle) + 1/4 sum_k |Z_kk|
            lam += float(np.sum(np.abs(np.triu(core, k=1))))
            lam += 0.25 * float(np.sum(np.abs(np.diag(core))))
        return lam

    if convention == "burg":
        for core in factorization.leaf_cores:
            eigenvalues, vectors = np.linalg.eigh(np.asarray(core,
                                                             dtype=float))
            column_abs_sum = np.sum(np.abs(vectors), axis=0)
            lam += 0.25 * float(np.sum(
                np.abs(eigenvalues) * column_abs_sum**2))
        return lam

    raise ValueError("convention must be 'lcu' or 'burg'.")
