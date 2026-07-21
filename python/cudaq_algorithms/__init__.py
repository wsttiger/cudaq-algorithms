# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""CUDA-Q Algorithms.

A pure-Python package: the fermion-to-qubit transforms
(:mod:`.fermion`), the state-preparation kernels and operator pools
(:mod:`.stateprep`), the quantum primitives (:mod:`.pauli_lcu`,
:mod:`.qubitization`, :mod:`.qsvt`, :mod:`.trotter`,
:mod:`.sim_utils`), and the classical double-factorization
preprocessing (:mod:`.double_factorization`, NumPy/SciPy with optional
CuPy GPU acceleration) are implemented as CUDA-Q Python kernels and
host-side helpers.

Submodules load lazily, so the ``cudaq`` runtime is only required at
the point of use: the classical modules (:mod:`.double_factorization`,
:mod:`.block_encoding`, and the integral helpers in :mod:`.chemistry`)
import and run with NumPy/SciPy alone — including on platforms the
``cudaq`` wheels do not support, such as macOS. Everything that builds
or simulates circuits needs ``cudaq`` (Linux; elsewhere use the CUDA-Q
container).
"""


def _resolve_version() -> str:
    from importlib.metadata import PackageNotFoundError, version

    for distribution in ("cudaq-algorithms-cu12", "cudaq-algorithms-cu13"):
        try:
            return f"CUDA-Q Algorithms {version(distribution)}"
        except PackageNotFoundError:
            continue
    return "CUDA-Q Algorithms (source)"


__version__ = _resolve_version()
del _resolve_version

_SUBMODULES = (
    "block_encoding",
    "chemistry",
    "common_kernels",
    "df_encoding",
    "double_factorization",
    "fermion",
    "pauli_lcu",
    "qsvt",
    "qubitization",
    "sim_utils",
    "stateprep",
    "trotter",
)

# Root re-exports and the submodule each lazily resolves from. The
# composable device kernels keep their module namespaces
# (cudaq_algorithms.pauli_lcu.prepare, .select, .apply_phase_sequence, ...;
# cudaq_algorithms.common_kernels.reflect_about_zero, .signal_phase, ...):
# their names are too generic to export from the package root.
_ROOT_EXPORTS = {
    "BlockEncoding": "block_encoding",
    "state_from": "common_kernels",
    "DoubleFactorizedEncoding": "df_encoding",
    "PauliLCU": "pauli_lcu",
    "select_observable": "pauli_lcu",
    "ADJOINT": "qsvt",
    "FORWARD": "qsvt",
    "PhaseSequence": "qsvt",
    "QSVT": "qsvt",
    "recover_real_time_evolution": "qsvt",
    "Walk": "qubitization",
    "reflection_observable": "qubitization",
    "Trotter": "trotter",
    "TrotterOrdering": "trotter",
    "TrotterResourceEstimate": "trotter",
}

__all__ = list(_SUBMODULES) + list(_ROOT_EXPORTS)


def _import_submodule(name: str):
    from importlib import import_module

    try:
        return import_module(f".{name}", __name__)
    except ModuleNotFoundError as exc:
        if exc.name != "cudaq":
            raise
        raise ModuleNotFoundError(
            f"cudaq_algorithms.{name} requires the cudaq runtime, which is "
            "not installed (and has no wheels for this platform if pip "
            "skipped it). The classical modules — double_factorization, "
            "block_encoding, and the integral helpers in chemistry — work "
            "without it; everything that builds or simulates circuits needs "
            "cudaq (Linux natively, the CUDA-Q container elsewhere).",
            name="cudaq") from exc


def __getattr__(name: str):
    if name in _SUBMODULES:
        return _import_submodule(name)
    submodule = _ROOT_EXPORTS.get(name)
    if submodule is not None:
        return getattr(_import_submodule(submodule), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list:
    return sorted(set(globals()) | set(__all__))
