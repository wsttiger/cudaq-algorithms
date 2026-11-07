# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Joint gradient descent for compressed double factorization.

The joint strategy treats the antisymmetric leaf generators ``X^t`` and the
symmetric cores ``Z^t`` as one set of optimization variables and lowers

    F(X, Z) = 1/2 ||eri - sum_t U^t Z^t (U^t)^T (congruence)||_F^2
              + rho sum_t ||Z^t||_F^2,      U^t = exp(X^t),

by full-batch gradient descent with Armijo backtracking. Both blocks move in
the same step from the same accepted state; no inner core solve runs at any
point. The parameters live in Frobenius-preserving packed coordinates: the
strict lower triangle of each ``X^t`` with weight ``sqrt(2)``, and the upper
triangle of each ``Z^t`` with diagonal weight 1 and off-diagonal weight
``sqrt(2)``, so packed Euclidean norms equal matrix Frobenius norms and the
packed gradient is the exact Riesz representative of the matrix gradient.

The evaluator is backend-neutral: the same code runs on NumPy and on CuPy,
where every tensor (integrals, parameters, rotations, residual, gradients,
trial buffers) stays on the device and only scalar decisions (objective
values, gradient norms, the conditional-core residual) are read back.
"""
from __future__ import annotations

import math
import time
import warnings
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import scipy.linalg

from ._backend import (ArrayModule, DeviceArray, skew_spectral_factors,
                       to_numpy)

_SQRT2 = math.sqrt(2.0)

STRATEGY_VARIABLE_PROJECTION = "variable_projection"
STRATEGY_JOINT_GRADIENT_DESCENT = "joint_gradient_descent"
OPTIMIZATION_STRATEGIES = (STRATEGY_VARIABLE_PROJECTION,
                           STRATEGY_JOINT_GRADIENT_DESCENT)

_STEP_POLICIES = ("fixed", "adaptive")
_ROTATION_ADJOINTS = ("spectral", "scipy")

# Accepted updates between forced rechecks of the conditional-core residual
# while both gradient blocks are within tolerance.
CORE_RECHECK_INTERVAL = 100

# Termination reasons of the joint loop and their ``optimizer_status`` codes.
JOINT_TERMINATIONS = {
    "converged": 0,
    "max_iterations": 1,
    "line_search_failed": 2,
    "numerical_failure": 3,
    "stationary_core_residual_unmet": 4,
    "objective_stagnation": 5,
}


@dataclass(frozen=True)
class JointGradientDescentOptions:
    """Controls of the ``"joint_gradient_descent"`` strategy of
    :func:`compressed_double_factorization`.

    - ``generator_step_scale`` / ``core_step_scale`` (``s_x``, ``s_z``,
      positive): the descent direction is ``(-s_x g_x, -s_z g_z)`` in packed
      coordinates; the Armijo step ``alpha`` then scales both blocks
      together.
    - ``armijo_coefficient`` (``c1``), ``backtracking_factor``,
      ``max_backtracks``, ``initial_step``: a trial step ``alpha`` is
      accepted when ``F(trial) <= F + c1 alpha (g . d)``; otherwise it is
      multiplied by the factor, at most ``max_backtracks`` times, before the
      iteration is declared failed.
    - ``step_policy``: ``"fixed"`` starts every line search at
      ``initial_step``; ``"adaptive"`` starts at the previously accepted
      step, doubled when that step was accepted without backtracking, and
      capped at ``initial_step``.
    - ``core_residual_tolerance``: the relative conditional-core residual
      ``||R - (A + 2 rho I) Z||_F / ||R||_F`` the returned cores must meet
      for the fit to count as converged (recomputed with the normal
      operator, independently of the gradient evaluator). When ``R == 0``
      the residual is compared with ``core_residual_absolute_tolerance``
      instead.
    - ``objective_tolerance``: when positive, an accepted step whose
      relative objective decrease ``(F_old - F_new) / max(F_old, tiny)`` is
      below it ends the loop with ``"objective_stagnation"``; ``0`` disables
      the check.
    - ``history_length``: number of most recent per-iteration records kept
      in ``JointGradientDescentDiagnostics.history`` (``0`` keeps none).
    - ``rotation_adjoint``: ``"spectral"`` evaluates the Frechet adjoint of
      ``exp`` from the batched Hermitian eigendecomposition of ``i X`` (both
      backends); ``"scipy"`` uses ``scipy.linalg.expm_frechet`` per leaf on
      the host (NumPy backend only; reference/validation path).
    """

    generator_step_scale: float = 1.0
    core_step_scale: float = 1.0
    armijo_coefficient: float = 1.0e-4
    backtracking_factor: float = 0.5
    max_backtracks: int = 20
    initial_step: float = 1.0
    step_policy: str = "adaptive"
    core_residual_tolerance: float = 1.0e-10
    core_residual_absolute_tolerance: float = 0.0
    objective_tolerance: float = 0.0
    history_length: int = 0
    rotation_adjoint: str = "spectral"

    def validate(self) -> None:
        for name in ("generator_step_scale", "core_step_scale",
                     "initial_step"):
            value = getattr(self, name)
            if not (isinstance(value, (int, float)) and math.isfinite(value)
                    and value > 0.0):
                raise ValueError(
                    f"JointGradientDescentOptions.{name} must be a positive "
                    f"finite number, got {value!r}.")
        if not (0.0 < self.armijo_coefficient < 1.0):
            raise ValueError(
                "JointGradientDescentOptions.armijo_coefficient must lie in "
                f"(0, 1), got {self.armijo_coefficient!r}.")
        if not (0.0 < self.backtracking_factor < 1.0):
            raise ValueError(
                "JointGradientDescentOptions.backtracking_factor must lie in "
                f"(0, 1), got {self.backtracking_factor!r}.")
        if not (isinstance(self.max_backtracks, int)
                and self.max_backtracks >= 0):
            raise ValueError(
                "JointGradientDescentOptions.max_backtracks must be a "
                f"non-negative integer, got {self.max_backtracks!r}.")
        if self.step_policy not in _STEP_POLICIES:
            raise ValueError(
                "JointGradientDescentOptions.step_policy must be one of "
                f"{_STEP_POLICIES}, got {self.step_policy!r}.")
        for name in ("core_residual_tolerance",
                     "core_residual_absolute_tolerance",
                     "objective_tolerance"):
            value = getattr(self, name)
            if not (isinstance(value, (int, float)) and math.isfinite(value)
                    and value >= 0.0):
                raise ValueError(
                    f"JointGradientDescentOptions.{name} must be a "
                    f"non-negative finite number, got {value!r}.")
        if not (isinstance(self.history_length, int)
                and self.history_length >= 0):
            raise ValueError(
                "JointGradientDescentOptions.history_length must be a "
                f"non-negative integer, got {self.history_length!r}.")
        if self.rotation_adjoint not in _ROTATION_ADJOINTS:
            raise ValueError(
                "JointGradientDescentOptions.rotation_adjoint must be one of "
                f"{_ROTATION_ADJOINTS}, got {self.rotation_adjoint!r}.")

    def as_dict(self) -> Dict[str, object]:
        return {
            "generator_step_scale": self.generator_step_scale,
            "core_step_scale": self.core_step_scale,
            "armijo_coefficient": self.armijo_coefficient,
            "backtracking_factor": self.backtracking_factor,
            "max_backtracks": self.max_backtracks,
            "initial_step": self.initial_step,
            "step_policy": self.step_policy,
            "core_residual_tolerance": self.core_residual_tolerance,
            "core_residual_absolute_tolerance":
            self.core_residual_absolute_tolerance,
            "objective_tolerance": self.objective_tolerance,
            "history_length": self.history_length,
            "rotation_adjoint": self.rotation_adjoint,
        }


@dataclass
class JointGradientDescentDiagnostics:
    """Outcome of a ``"joint_gradient_descent"`` C-DF fit.

    ``gradient_norm`` is the Euclidean norm of the full packed ``(X, Z)``
    gradient at the returned state (the value reported as
    ``DoubleFactorization.optimizer_grad_norm``). ``success`` is ``True`` only when the complete convergence rule held at
    the returned state: finite objective, gradients, and parameters; both
    packed gradient infinity norms (``generator_gradient_inf``,
    ``core_gradient_inf``) at most ``gradient_tolerance``; and the
    independently recomputed conditional-core residual
    (``core_residual_norm`` against ``core_residual_threshold``) within its
    tolerance. ``termination`` is one of the keys of
    :data:`JOINT_TERMINATIONS`:

    - ``"converged"`` -- the complete rule held;
    - ``"max_iterations"`` -- the accepted-update budget ran out;
    - ``"line_search_failed"`` -- no Armijo-acceptable step was found within
      ``max_backtracks`` halvings;
    - ``"numerical_failure"`` -- a non-finite objective, gradient, or
      parameter at an accepted state;
    - ``"stationary_core_residual_unmet"`` -- the budget ran out (or the
      line search failed) while both gradient blocks were within tolerance
      but the conditional-core residual was not;
    - ``"objective_stagnation"`` -- the relative objective decrease fell
      below ``objective_tolerance``.

    ``iterations`` counts accepted simultaneous updates;
    ``objective_evaluations`` every objective evaluation (accepted and
    rejected trials, plus the initial and final audits);
    ``gradient_evaluations`` every gradient evaluation (one per accepted
    state plus the final audit); ``backtracks`` the rejected trials;
    ``host_syncs`` the scalar device-to-host reads the loop made on the
    CuPy backend (zero-dimensional only; the returned arrays are copied
    once at the end); ``core_residual_checks`` how many times the
    conditional-core residual was recomputed during the loop. ``core_norm`` is ``||Z||_F`` of the returned cores,
    which at ``rho = 0`` is otherwise unconstrained. ``generators_init``
    and ``cores_init`` say where the starting point came from. ``history``
    holds at most ``history_length`` of the most recent per-iteration
    records (iteration, objective, both gradient infinity norms, accepted
    step, backtracks).
    """

    strategy: str
    backend: str
    success: bool
    termination: str
    message: str
    iterations: int
    objective_evaluations: int
    gradient_evaluations: int
    backtracks: int
    host_syncs: int
    core_residual_checks: int
    final_step: Optional[float]
    generator_step_scale: float
    core_step_scale: float
    objective: float
    generator_gradient_inf: float
    core_gradient_inf: float
    gradient_norm: float
    gradient_tolerance: float
    core_residual_norm: float
    core_rhs_norm: float
    core_residual_threshold: float
    core_residual_tolerance: float
    core_residual_converged: bool
    core_norm: float
    generators_init: str
    cores_init: str
    elapsed_seconds: float
    options: Dict[str, object] = field(default_factory=dict)
    history: Optional[List[Dict[str, float]]] = None

    @property
    def core_residual_ratio(self) -> float:
        """``core_residual_norm / core_rhs_norm`` (``0.0`` for a zero
        right-hand side)."""
        return (self.core_residual_norm /
                self.core_rhs_norm if self.core_rhs_norm > 0.0 else 0.0)


# --------------------------------------------------------------------------- #
# Packed coordinates
# --------------------------------------------------------------------------- #


def _lower_indices(n: int, xp: ArrayModule):
    rows, cols = np.tril_indices(n, k=-1)
    return xp.asarray(rows), xp.asarray(cols)


def _upper_indices(n: int, xp: ArrayModule):
    rows, cols = np.triu_indices(n)
    weights = np.where(rows == cols, 1.0, _SQRT2)
    return xp.asarray(rows), xp.asarray(cols), xp.asarray(weights)


def skew_parameter_count(n: int) -> int:
    """Packed coordinates per antisymmetric ``n x n`` generator."""
    return n * (n - 1) // 2


def symmetric_parameter_count(n: int) -> int:
    """Packed coordinates per symmetric ``n x n`` core."""
    return n * (n + 1) // 2


def pack_skew(generators: DeviceArray, xp: ArrayModule) -> DeviceArray:
    """Pack a stack of antisymmetric ``(T, n, n)`` generators into one vector
    of length ``T n(n-1)/2``: the strict lower triangle scaled by
    ``sqrt(2)``, so that ``||packed||_2 == ||X||_F``."""
    generators = xp.asarray(generators)
    n = generators.shape[-1]
    rows, cols = _lower_indices(n, xp)
    return (generators[:, rows, cols] * _SQRT2).reshape(-1)


def unpack_skew(vector: DeviceArray, num_leaves: int, n: int,
                xp: ArrayModule) -> DeviceArray:
    """Inverse of :func:`pack_skew`: ``(T, n, n)`` antisymmetric generators
    (the upper triangle is the negated lower triangle)."""
    rows, cols = _lower_indices(n, xp)
    values = xp.asarray(vector).reshape(num_leaves, rows.shape[0]) / _SQRT2
    generators = xp.zeros((num_leaves, n, n), dtype=float)
    generators[:, rows, cols] = values
    generators[:, cols, rows] = -values
    return generators


def pack_skew_gradient(matrix_gradient: DeviceArray,
                       xp: ArrayModule) -> DeviceArray:
    """Packed-coordinate gradient from an ambient ``(T, n, n)`` matrix
    gradient ``G``: ``(G[i, j] - G[j, i]) / sqrt(2)`` for ``i > j``, the
    chain rule through :func:`unpack_skew`."""
    n = matrix_gradient.shape[-1]
    rows, cols = _lower_indices(n, xp)
    return ((matrix_gradient[:, rows, cols] - matrix_gradient[:, cols, rows]) /
            _SQRT2).reshape(-1)


def pack_symmetric_stack(cores: DeviceArray, xp: ArrayModule) -> DeviceArray:
    """Pack a stack of symmetric ``(T, n, n)`` cores into one vector of length
    ``T n(n+1)/2`` (diagonal weight 1, off-diagonal ``sqrt(2)``), the
    backend-neutral form of
    :func:`cudaq_algorithms.double_factorization.pack_symmetric`."""
    cores = xp.asarray(cores)
    n = cores.shape[-1]
    rows, cols, weights = _upper_indices(n, xp)
    return (cores[:, rows, cols] * weights).reshape(-1)


def unpack_symmetric_stack(vector: DeviceArray, num_leaves: int, n: int,
                           xp: ArrayModule) -> DeviceArray:
    """Inverse of :func:`pack_symmetric_stack`."""
    rows, cols, weights = _upper_indices(n, xp)
    values = xp.asarray(vector).reshape(num_leaves, rows.shape[0]) / weights
    cores = xp.zeros((num_leaves, n, n), dtype=float)
    cores[:, rows, cols] = values
    cores[:, cols, rows] = values
    return cores


def pack_symmetric_gradient(matrix_gradient: DeviceArray,
                            xp: ArrayModule) -> DeviceArray:
    """Packed-coordinate gradient from an ambient ``(T, n, n)`` matrix
    gradient: the symmetrized gradient in the symmetric packing (the chain
    rule through :func:`unpack_symmetric_stack`)."""
    symmetric = 0.5 * (matrix_gradient + matrix_gradient.transpose(0, 2, 1))
    return pack_symmetric_stack(symmetric, xp)


# --------------------------------------------------------------------------- #
# Rotation exponential and its adjoint
# --------------------------------------------------------------------------- #


def rotation_adjoint_spectral(eigenvalues: DeviceArray,
                              eigenvectors: DeviceArray,
                              rotation_gradient: DeviceArray,
                              xp: ArrayModule) -> DeviceArray:
    """Adjoint of the Frechet derivative of ``exp`` at the antisymmetric
    ``X`` from the spectral factors ``i X = V diag(lambda) V^H`` of
    :func:`~cudaq_algorithms.double_factorization._backend.skew_spectral_factors`:
    for a real ``(T, n, n)`` gradient ``G_U`` with respect to ``U = exp(X)``
    returns ``G_X = real(V (conj(K) * (V^H G_U V)) V^H)`` with the divided
    differences ``K[i, j] = exp(-i (lambda_i + lambda_j) / 2) sinc((lambda_i
    - lambda_j) / (2 pi))`` (``sinc(x) = sin(pi x) / (pi x)``, 1 at 0), the
    same matrix as ``scipy.linalg.expm_frechet(X.T, G_U)``. The
    ``sinc`` form has no removable singularity, so repeated and
    near-repeated eigenvalues need no guard, and eigenvectors are never
    differentiated."""
    mean = 0.5 * (eigenvalues[:, :, None] + eigenvalues[:, None, :])
    difference = eigenvalues[:, :, None] - eigenvalues[:, None, :]
    kernel = xp.exp(-1j * mean) * xp.sinc(difference / (2.0 * math.pi))
    v_h = xp.conj(eigenvectors).transpose(0, 2, 1)
    inner = v_h @ rotation_gradient.astype(complex) @ eigenvectors
    result = eigenvectors @ (xp.conj(kernel) * inner) @ v_h
    return xp.real(result)


def rotation_adjoint_scipy(generators: np.ndarray,
                           rotation_gradient: np.ndarray) -> np.ndarray:
    """Reference adjoint on the host: ``L_exp(X^T, G_U)`` per leaf through
    ``scipy.linalg.expm_frechet``."""
    generators = np.asarray(generators, dtype=float)
    rotation_gradient = np.asarray(rotation_gradient, dtype=float)
    return np.stack([
        scipy.linalg.expm_frechet(generators[t].T,
                                  rotation_gradient[t],
                                  compute_expm=False)
        for t in range(generators.shape[0])
    ])


# --------------------------------------------------------------------------- #
# Objective and gradients
# --------------------------------------------------------------------------- #


@dataclass
class JointState:
    """One evaluated point of the joint objective. ``x`` and ``z`` are the
    packed device vectors; ``objective`` is a host float. The gradients are
    filled by :meth:`JointEvaluator.gradient`; the spectral factors belong
    to this point only."""

    x: DeviceArray
    z: DeviceArray
    generators: DeviceArray
    eigenvalues: DeviceArray
    eigenvectors: DeviceArray
    rotations: DeviceArray
    cores: DeviceArray
    residual: DeviceArray
    objective: float
    gradient_x: Optional[DeviceArray] = None
    gradient_z: Optional[DeviceArray] = None
    gradient_x_inf: Optional[float] = None
    gradient_z_inf: Optional[float] = None


class JointEvaluator:
    """Stateless evaluator of the joint objective and its two packed
    gradient blocks on the backend ``xp``.

    ``evaluate`` returns the objective (and the intermediates the gradient
    reuses); ``gradient`` completes a state with its packed gradients. Both
    are pure functions of the packed vectors; nothing is cached across
    calls. Every contraction is a matrix product over the leaf column basis
    ``A[t, pq, k] = U^t_pk U^t_qk`` (shape ``(T, n^2, n)``): the
    reconstruction is one GEMM ``(n^2, T n) x (T n, n^2)`` and both
    gradients share one GEMM ``(n^2, n^2) x (n^2, T n)``, so no ``T n^4``
    intermediate is formed. Counters record evaluations and host reads for
    diagnostics.
    """

    def __init__(self,
                 eri_dev: DeviceArray,
                 num_leaves: int,
                 regularization: float,
                 xp: ArrayModule,
                 rotation_adjoint: str = "spectral"):
        self.xp = xp
        self.eri = eri_dev
        self.n = int(eri_dev.shape[0])
        self.num_leaves = int(num_leaves)
        self.regularization = float(regularization)
        if rotation_adjoint not in _ROTATION_ADJOINTS:
            raise ValueError(
                f"rotation_adjoint must be one of {_ROTATION_ADJOINTS}.")
        if rotation_adjoint == "scipy" and xp is not np:
            raise ValueError("rotation_adjoint='scipy' is available on the "
                             "NumPy backend only.")
        self.rotation_adjoint = rotation_adjoint
        self.objective_evaluations = 0
        self.gradient_evaluations = 0
        self.host_syncs = 0

    def _scalar(self, value) -> float:
        self.host_syncs += 1
        return float(value)

    def leaf_columns(self, rotations: DeviceArray) -> DeviceArray:
        """``A[t, pq, k] = U^t_pk U^t_qk``, shape ``(T, n^2, n)``."""
        n = self.n
        outer = rotations[:, :, None, :] * rotations[:, None, :, :]
        return outer.reshape(self.num_leaves, n * n, n)

    def reconstruct(self, rotations: DeviceArray,
                    cores: DeviceArray) -> DeviceArray:
        """``V_hat[pq, rs] = sum_t (A_t Z_t) A_t^T`` as one GEMM."""
        n = self.n
        columns = self.leaf_columns(rotations)
        weighted = columns @ cores  # (T, n^2, n) = A_t Z_t
        left = weighted.transpose(1, 0, 2).reshape(n * n, -1)  # (n^2, T n)
        right = columns.transpose(1, 0, 2).reshape(n * n, -1)  # (n^2, T n)
        return (left @ right.T).reshape(n, n, n, n)

    def residual_columns(self, tensor: DeviceArray,
                         columns: DeviceArray) -> DeviceArray:
        """``C[t, pq, l] = sum_rs tensor[pq, rs] A_t[rs, l]`` for every leaf
        (one GEMM ``(n^2, n^2) x (n^2, T n)``), shape ``(T, n^2, n)``."""
        n = self.n
        flat = columns.transpose(1, 0, 2).reshape(n * n, -1)  # (n^2, T n)
        product = tensor.reshape(n * n, n * n) @ flat  # (n^2, T n)
        return product.reshape(n * n, self.num_leaves, n).transpose(1, 0, 2)

    def project_into_leaves(self, tensor: DeviceArray,
                            rotations: DeviceArray) -> DeviceArray:
        """``P[t, k, l] = sum_pqrs U^t_pk U^t_qk tensor[p, q, r, s] U^t_rl
        U^t_sl`` for every leaf, i.e. ``A_t^T (tensor A_t)``."""
        columns = self.leaf_columns(rotations)
        return columns.transpose(0, 2, 1) @ self.residual_columns(
            tensor, columns)

    def evaluate(self, x: DeviceArray, z: DeviceArray) -> JointState:
        """Objective at the packed point ``(x, z)``; one host read."""
        xp = self.xp
        generators = unpack_skew(x, self.num_leaves, self.n, xp)
        cores = unpack_symmetric_stack(z, self.num_leaves, self.n, xp)
        eigenvalues, eigenvectors, rotations = skew_spectral_factors(
            generators, xp)
        residual = self.eri - self.reconstruct(rotations, cores)
        objective = 0.5 * xp.sum(residual * residual)
        if self.regularization > 0.0:
            objective = objective + self.regularization * xp.sum(cores * cores)
        self.objective_evaluations += 1
        return JointState(x=x,
                          z=z,
                          generators=generators,
                          eigenvalues=eigenvalues,
                          eigenvectors=eigenvectors,
                          rotations=rotations,
                          cores=cores,
                          residual=residual,
                          objective=self._scalar(objective))

    def _gradient_pieces(self, state: JointState):
        """The leaf columns ``A_t`` and the residual contraction ``C_t``
        that both partial gradients share."""
        columns = self.leaf_columns(state.rotations)
        return columns, self.residual_columns(state.residual, columns)

    def rotation_gradient(self, state: JointState, pieces=None) -> DeviceArray:
        """``dF/dU^t_ak = -4 sum_qrsl Delta_aqrs U_qk Z_kl U_rl U_sl``, i.e.
        ``-4 sum_q (C_t Z_t)[aq, k] U^t_qk`` with ``C_t`` of
        :meth:`residual_columns`."""
        n = self.n
        _, contracted = (pieces if pieces is not None else
                         self._gradient_pieces(state))
        weighted = contracted @ state.cores  # (T, n^2, k): sum_l C[aq, l] Z[k, l]
        weighted = weighted.reshape(self.num_leaves, n, n, n)  # (t, a, q, k)
        return -4.0 * self.xp.sum(weighted * state.rotations[:, None, :, :],
                                  axis=2)

    def core_gradient_matrix(self,
                             state: JointState,
                             pieces=None) -> DeviceArray:
        """``dF/dZ^t = -P_t(Delta) + 2 rho Z^t`` with ``P_t`` the projection
        of the fit residual into leaf ``t`` (the residual-projection form;
        the normal-operator form ``((A + 2 rho I) Z - R)^t`` is the same
        matrix and is used only for the independent audit)."""
        columns, contracted = (pieces if pieces is not None else
                               self._gradient_pieces(state))
        gradient = -(columns.transpose(0, 2, 1) @ contracted)
        if self.regularization > 0.0:
            gradient = gradient + (2.0 * self.regularization) * state.cores
        return gradient

    def generator_gradient_matrix(self,
                                  state: JointState,
                                  pieces=None) -> DeviceArray:
        """Ambient ``(T, n, n)`` gradient with respect to ``X``: the rotation
        gradient pulled back through the Frechet adjoint of ``exp``."""
        grad_u = self.rotation_gradient(state, pieces)
        if self.rotation_adjoint == "scipy":
            return rotation_adjoint_scipy(state.generators, grad_u)
        return rotation_adjoint_spectral(state.eigenvalues, state.eigenvectors,
                                         grad_u, self.xp)

    def gradient(self, state: JointState) -> JointState:
        """Fill both packed gradient blocks and their infinity norms (two
        host reads)."""
        xp = self.xp
        pieces = self._gradient_pieces(state)
        state.gradient_x = pack_skew_gradient(
            self.generator_gradient_matrix(state, pieces), xp)
        state.gradient_z = pack_symmetric_gradient(
            self.core_gradient_matrix(state, pieces), xp)
        state.gradient_x_inf = (self._scalar(xp.max(xp.abs(state.gradient_x)))
                                if state.gradient_x.size else 0.0)
        state.gradient_z_inf = self._scalar(xp.max(xp.abs(state.gradient_z)))
        self.gradient_evaluations += 1
        return state


def conditional_core_residual(eri_dev: DeviceArray, rotations: DeviceArray,
                              cores: DeviceArray, regularization: float,
                              xp: ArrayModule) -> tuple[float, float]:
    """``(||R - (A + 2 rho I) Z||_F, ||R||_F)`` of the fixed-rotation core
    normal equations, recomputed from the normal operator of
    :class:`~cudaq_algorithms.double_factorization._factorization._InnerOperator`
    (independent of the residual-projection form the evaluator uses)."""
    from ._factorization import _InnerOperator
    operator = _InnerOperator(eri_dev, rotations, xp, regularization)
    return operator.residual_norm(cores), operator.rhs_norm


def default_initial_cores(eri_dev: DeviceArray, rotations: DeviceArray,
                          regularization: float,
                          xp: ArrayModule) -> DeviceArray:
    """Solver-free starting cores at fixed rotations: the ERI projected into
    each leaf, ``Z^t = R^t / (T (1 + 2 rho))``. The normal operator's
    diagonal blocks are ``(1 + 2 rho) I`` and ``T`` leaves share the fit,
    so this is the diagonal-block solution with the leaf count as the
    overlap heuristic."""
    rotations = xp.asarray(rotations)
    num_leaves = int(rotations.shape[0])
    evaluator = JointEvaluator(eri_dev, num_leaves, regularization, xp)
    projected = evaluator.project_into_leaves(eri_dev, rotations)
    projected = 0.5 * (projected + projected.transpose(0, 2, 1))
    return projected / (num_leaves * (1.0 + 2.0 * float(regularization)))


# --------------------------------------------------------------------------- #
# The descent loop
# --------------------------------------------------------------------------- #


def run_joint_gradient_descent(
    eri_dev: DeviceArray,
    initial_x: DeviceArray,
    initial_z: DeviceArray,
    num_leaves: int,
    regularization: float,
    xp: ArrayModule,
    backend_name: str,
    tolerance: float,
    max_iterations: int,
    options: JointGradientDescentOptions,
    generators_init: str,
    cores_init: str,
) -> tuple[DeviceArray, DeviceArray, JointGradientDescentDiagnostics]:
    """Full-batch gradient descent with Armijo backtracking on the joint
    objective from the packed device vectors ``(initial_x, initial_z)``.

    Each iteration evaluates both gradient blocks at the accepted state,
    forms ``d = (-s_x g_x, -s_z g_z)``, and tries ``alpha`` along it until
    ``F(trial) <= F + c1 alpha (g . d)``; trials are independent evaluations
    that inherit nothing from rejected trials, and both blocks are committed
    together. Convergence is declared only when both packed gradient
    infinity norms are at most ``tolerance`` *and* the conditional-core
    residual, recomputed with the normal operator at the returned state,
    is within ``options.core_residual_tolerance`` (relative; absolute
    ``options.core_residual_absolute_tolerance`` when ``R == 0``). Any other
    end is reported as its own termination reason; the best finite accepted
    state is returned and its objective, gradients, and core residual are
    recomputed afresh for the diagnostics.

    Returns ``(rotations, cores, diagnostics)`` with device arrays.
    """
    options.validate()
    start = time.perf_counter()
    n = int(eri_dev.shape[0])
    num_leaves = int(num_leaves)
    tolerance = float(tolerance)
    if not (math.isfinite(tolerance) and tolerance > 0.0):
        raise ValueError("tolerance must be a positive finite number.")
    if max_iterations < 0:
        raise ValueError("max_iterations must be >= 0.")
    evaluator = JointEvaluator(eri_dev, num_leaves, regularization, xp,
                               options.rotation_adjoint)
    history: Optional[deque] = (deque(
        maxlen=options.history_length) if options.history_length > 0 else None)
    s_x, s_z = options.generator_step_scale, options.core_step_scale
    c1, factor = options.armijo_coefficient, options.backtracking_factor

    def core_check(state: JointState) -> tuple[float, float, float, bool]:
        residual_norm, rhs_norm = conditional_core_residual(
            eri_dev, state.rotations, state.cores, regularization, xp)
        evaluator.host_syncs += 2
        threshold = (options.core_residual_tolerance * rhs_norm if rhs_norm
                     > 0.0 else options.core_residual_absolute_tolerance)
        converged = math.isfinite(residual_norm) and residual_norm <= threshold
        return residual_norm, rhs_norm, threshold, converged

    def finite(state: JointState) -> bool:
        if not math.isfinite(state.objective):
            return False
        if not (math.isfinite(state.gradient_x_inf)
                and math.isfinite(state.gradient_z_inf)):
            return False
        return True

    def record(state: JointState, iteration: int, step: Optional[float],
               backtracks: int) -> None:
        if history is not None:
            history.append(
                dict(iteration=iteration,
                     objective=state.objective,
                     generator_gradient_inf=state.gradient_x_inf,
                     core_gradient_inf=state.gradient_z_inf,
                     step=step,
                     backtracks=backtracks))

    state = evaluator.gradient(
        evaluator.evaluate(xp.asarray(initial_x, dtype=float),
                           xp.asarray(initial_z, dtype=float)))
    record(state, 0, None, 0)
    best = state
    iterations = 0
    total_backtracks = 0
    last_step: Optional[float] = None
    last_backtracks = 0
    termination: Optional[str] = None
    message = ""

    core_checks = 0
    last_check_gz: Optional[float] = None
    last_check_iteration = -1
    stationary_message = ""

    def stationary_at(point: JointState) -> bool:
        return (point.gradient_x_inf <= tolerance
                and point.gradient_z_inf <= tolerance)

    while True:
        if not finite(state):
            termination = "numerical_failure"
            message = "non-finite objective or gradient at an accepted state"
            break
        stationary = stationary_at(state)
        if stationary:
            # The core residual scales with ||g_Z||_F, so it is rechecked
            # when that norm has halved since the last check (or every
            # CORE_RECHECK_INTERVAL accepted updates), and the descent
            # continues in between: the gradient blocks being within
            # tolerance does not stop the loop while the independent core
            # criterion is unmet.
            if (last_check_gz is None
                    or state.gradient_z_inf <= 0.5 * last_check_gz
                    or iterations - last_check_iteration
                    >= CORE_RECHECK_INTERVAL):
                residual_norm, _, threshold, core_converged = core_check(state)
                core_checks += 1
                last_check_gz = state.gradient_z_inf
                last_check_iteration = iterations
                if core_converged:
                    termination = "converged"
                    break
                stationary_message = (
                    f"both gradient blocks are within {tolerance:g} but the "
                    f"conditional-core residual {residual_norm:.3e} exceeds "
                    f"its threshold {threshold:.3e}")
        if iterations >= max_iterations:
            if stationary:
                termination = "stationary_core_residual_unmet"
                message = (f"{max_iterations} accepted updates; "
                           f"{stationary_message}")
            else:
                termination = "max_iterations"
                message = (f"{max_iterations} accepted updates without "
                           f"meeting the gradient tolerance {tolerance:g}")
            break
        direction_x = -s_x * state.gradient_x
        direction_z = -s_z * state.gradient_z
        slope = -evaluator._scalar(
            s_x * xp.sum(state.gradient_x * state.gradient_x) +
            s_z * xp.sum(state.gradient_z * state.gradient_z))
        if not (slope < 0.0):
            termination = "numerical_failure"
            message = f"descent slope {slope:.3e} is not negative"
            break
        if options.step_policy == "adaptive" and last_step is not None:
            # Start from the last accepted step; grow it only after an
            # iteration that accepted its first trial, so a well-sized step
            # is reused at one evaluation per update.
            alpha = min(options.initial_step,
                        2.0 * last_step if last_backtracks == 0 else last_step)
        else:
            alpha = options.initial_step
        accepted: Optional[JointState] = None
        backtracks = 0
        for _ in range(options.max_backtracks + 1):
            trial = evaluator.evaluate(state.x + alpha * direction_x,
                                       state.z + alpha * direction_z)
            if (math.isfinite(trial.objective) and trial.objective
                    <= state.objective + c1 * alpha * slope):
                accepted = trial
                break
            backtracks += 1
            alpha *= factor
        total_backtracks += backtracks
        if accepted is None:
            message = (f"no Armijo-acceptable step after "
                       f"{options.max_backtracks} backtracks (last trial "
                       f"step {alpha / factor:.3e}, slope {slope:.3e})")
            if stationary:
                termination = "stationary_core_residual_unmet"
                message = f"{stationary_message}; {message}"
            else:
                termination = "line_search_failed"
            break
        previous_objective = state.objective
        state = evaluator.gradient(accepted)
        iterations += 1
        last_step = alpha
        last_backtracks = backtracks
        record(state, iterations, alpha, backtracks)
        if finite(state):
            best = state
        if options.objective_tolerance > 0.0:
            decrease = (previous_objective - state.objective) / max(
                abs(previous_objective),
                np.finfo(float).tiny)
            if decrease < options.objective_tolerance:
                termination = "objective_stagnation"
                message = (f"relative objective decrease {decrease:.3e} below "
                           f"objective_tolerance "
                           f"{options.objective_tolerance:g}")
                break

    # Fresh audit of the returned state: objective, both gradients, and the
    # conditional-core residual are recomputed rather than read from the
    # loop's bookkeeping.
    final = evaluator.gradient(evaluator.evaluate(best.x, best.z))
    parameters_finite = bool(
        to_numpy(xp.isfinite(final.x).all() & xp.isfinite(final.z).all()))
    evaluator.host_syncs += 1
    residual_norm, rhs_norm, threshold, core_converged = core_check(final)
    core_checks += 1
    stationary = (final.gradient_x_inf <= tolerance
                  and final.gradient_z_inf <= tolerance)
    success = bool(termination == "converged" and finite(final)
                   and parameters_finite and stationary and core_converged)
    if termination == "converged" and not success:
        termination = "numerical_failure"
        message = "the final audit did not confirm the convergence rule"
    core_norm = evaluator._scalar(xp.sqrt(xp.sum(final.cores * final.cores)))
    gradient_norm = math.sqrt(
        evaluator._scalar(
            xp.sum(final.gradient_x * final.gradient_x) +
            xp.sum(final.gradient_z * final.gradient_z)))
    elapsed = time.perf_counter() - start
    if not success:
        warnings.warn(
            "double_factorization: joint gradient descent did not converge "
            f"(termination={termination}, iterations={iterations}/"
            f"{max_iterations}, ||g_X||_inf={final.gradient_x_inf:.3e}, "
            f"||g_Z||_inf={final.gradient_z_inf:.3e}, tolerance="
            f"{tolerance:g}, core residual {residual_norm:.3e} vs threshold "
            f"{threshold:.3e}{'; ' + message if message else ''}); the best "
            "accepted state is returned -- see DoubleFactorization.joint.",
            RuntimeWarning)
    diagnostics = JointGradientDescentDiagnostics(
        strategy=STRATEGY_JOINT_GRADIENT_DESCENT,
        backend=backend_name,
        success=success,
        termination=termination,
        message=message,
        iterations=iterations,
        objective_evaluations=evaluator.objective_evaluations,
        gradient_evaluations=evaluator.gradient_evaluations,
        backtracks=total_backtracks,
        host_syncs=evaluator.host_syncs,
        core_residual_checks=core_checks,
        final_step=last_step,
        generator_step_scale=s_x,
        core_step_scale=s_z,
        objective=final.objective,
        generator_gradient_inf=final.gradient_x_inf,
        core_gradient_inf=final.gradient_z_inf,
        gradient_norm=gradient_norm,
        gradient_tolerance=tolerance,
        core_residual_norm=residual_norm,
        core_rhs_norm=rhs_norm,
        core_residual_threshold=threshold,
        core_residual_tolerance=options.core_residual_tolerance,
        core_residual_converged=core_converged,
        core_norm=core_norm,
        generators_init=generators_init,
        cores_init=cores_init,
        elapsed_seconds=elapsed,
        options=options.as_dict(),
        history=list(history) if history is not None else None)
    return final.rotations, final.cores, diagnostics


# --------------------------------------------------------------------------- #
# Input validation shared with compressed_double_factorization
# --------------------------------------------------------------------------- #

# Relative asymmetry (against max(1, max |entry|)) treated as round-off and
# cleaned by (anti)symmetrization; anything larger is rejected.
ROUNDOFF_ASYMMETRY = 1.0e-8


def validate_stack(arrays, num_leaves: int, n: int, name: str,
                   antisymmetric: bool) -> np.ndarray:
    """Return a finite ``(num_leaves, n, n)`` float stack from a list or
    stacked array, antisymmetrized (``antisymmetric=True``) or
    symmetrized after checking that the asymmetry is round-off sized."""
    try:
        stack = np.array([np.asarray(a, dtype=float) for a in arrays],
                         dtype=float)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be {num_leaves} arrays of shape "
                         f"({n}, {n}).") from error
    if stack.shape != (num_leaves, n, n):
        raise ValueError(f"{name} must have shape ({num_leaves}, {n}, {n}), "
                         f"got {stack.shape}.")
    if not np.all(np.isfinite(stack)):
        raise ValueError(f"{name} must be finite.")
    flipped = stack.transpose(0, 2, 1)
    mismatch = stack + flipped if antisymmetric else stack - flipped
    scale = max(1.0, float(np.abs(stack).max()) if stack.size else 1.0)
    worst = float(np.abs(mismatch).max()) if stack.size else 0.0
    if worst > ROUNDOFF_ASYMMETRY * scale:
        kind = "antisymmetric" if antisymmetric else "symmetric"
        raise ValueError(f"{name} must be {kind} (max |deviation| {worst:.3e} "
                         f"at scale {scale:.3e}).")
    return 0.5 * (stack - flipped) if antisymmetric else 0.5 * (stack +
                                                                flipped)
