# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""Pure-Python Pauli LCU block encoding, qubitization, and QSVT.

Importing this package also registers it as ``cudaq.algorithms``, so the
public API is available under either name::

    import cudaq_algorithms
    from cudaq.algorithms import PauliLCU, Walk, QSVT, PhaseSequence
"""

import sys as _sys

import cudaq as _cudaq

from . import pauli_lcu, qsvt, qubitization
from .pauli_lcu import (PauliLCU, apply, controlled_select, prepare,
                        reflect_about_zero, select, state_from, unprepare,
                        walk)
from .qsvt import (ADJOINT, FORWARD, PhaseSequence, QSVT,
                   apply_controlled_phase_sequence, apply_phase_sequence,
                   controlled_signal_phase, recover_real_time_evolution,
                   signal_phase)
from .qubitization import (Walk, adjoint_walk, controlled_adjoint_walk,
                           controlled_reflect_about_prepare,
                           controlled_reflect_about_zero, controlled_walk,
                           reflect_about_prepare, reflection_observable,
                           select_observable)

__all__ = [
    "ADJOINT",
    "FORWARD",
    "PauliLCU",
    "PhaseSequence",
    "QSVT",
    "Walk",
    "adjoint_walk",
    "apply",
    "apply_controlled_phase_sequence",
    "apply_phase_sequence",
    "controlled_adjoint_walk",
    "controlled_reflect_about_prepare",
    "controlled_reflect_about_zero",
    "controlled_select",
    "controlled_signal_phase",
    "controlled_walk",
    "pauli_lcu",
    "prepare",
    "qsvt",
    "qubitization",
    "recover_real_time_evolution",
    "reflect_about_prepare",
    "reflect_about_zero",
    "reflection_observable",
    "select",
    "select_observable",
    "signal_phase",
    "state_from",
    "unprepare",
    "walk",
]

# Register the package under the cudaq.algorithms namespace.
_sys.modules["cudaq.algorithms"] = _sys.modules[__name__]
_sys.modules["cudaq.algorithms.pauli_lcu"] = pauli_lcu
_sys.modules["cudaq.algorithms.qubitization"] = qubitization
_sys.modules["cudaq.algorithms.qsvt"] = qsvt
_cudaq.algorithms = _sys.modules[__name__]
