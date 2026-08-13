# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Sparse-access block encodings (experimental).

Import this subpackage directly (``from cudaq_algorithms.sparse import
SparseOracleEncoding``); while experimental, nothing here is re-exported
from the package root.
"""

from ._alias_sampling import AliasSamplingPrepare
from ._banded import banded_oracles
from ._qrom import QROM
from ._sparse_lcu import SparseLCUEncoding
from ._sparse_oracle import OracleKernels, SparseOracleEncoding
from ._unary_iteration import UnaryIterationKernels, unary_iteration_kernels

__all__ = [
    "AliasSamplingPrepare",
    "OracleKernels",
    "QROM",
    "SparseLCUEncoding",
    "SparseOracleEncoding",
    "UnaryIterationKernels",
    "banded_oracles",
    "unary_iteration_kernels",
]
