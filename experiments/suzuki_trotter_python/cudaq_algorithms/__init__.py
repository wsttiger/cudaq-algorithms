# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                        #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""Pure-Python Suzuki-Trotter Hamiltonian simulation.

Importing this package also registers it under the ``cudaq.algorithms``
namespace, so the module is available under either name::

    import cudaq_algorithms
    from cudaq.algorithms import trotter

    plan = trotter.make_trotter_plan(hamiltonian, time=0.8, steps=4)
"""

import sys as _sys

import cudaq as _cudaq

from . import sim_utils, trotter

__all__ = [
    "sim_utils",
    "trotter",
]

# Register the package under the cudaq.algorithms namespace.
_sys.modules["cudaq.algorithms"] = _sys.modules[__name__]
_sys.modules["cudaq.algorithms.trotter"] = trotter
_sys.modules["cudaq.algorithms.sim_utils"] = sim_utils
_cudaq.algorithms = _sys.modules[__name__]
