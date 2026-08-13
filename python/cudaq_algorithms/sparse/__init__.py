# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Sparse-access block encodings (experimental).

Import this subpackage directly (``from cudaq_algorithms.sparse import
SparseOracleEncoding``); while experimental, nothing here is re-exported
from the package root.
"""

from ._banded import banded_oracles
from ._sparse_oracle import OracleKernels, SparseOracleEncoding

__all__ = [
    "OracleKernels",
    "SparseOracleEncoding",
    "banded_oracles",
]
