#!/usr/bin/env python3
"""Example 4 — Double factorization and the BlockEncoding protocol.

Two ideas that reinforce each other:

  A. The compression dial. Double factorization rewrites the two-electron
     tensor as a sum of low-rank "leaves". Keep fewer leaves and the
     block-encoding normalization `alpha` drops -- and since QSVT degree
     scales like alpha * t, a smaller alpha means fewer walk steps for the
     same simulated time. Compression is a resource knob you can see.

  B. Polymorphism through the protocol. `PauliLCU` and
     `DoubleFactorizedEncoding` are unrelated classes, but both satisfy
     the `BlockEncoding` protocol -- so `Walk` and `QSVT` consume either
     one with identical code. Bring an encoding, inherit the primitives.

H2/STO-3G integrals are hardcoded, so no PySCF is needed.
Run: python3 04_double_factorization_and_the_protocol.py
"""

from __future__ import annotations

import os

import cudaq
import numpy as np

from cudaq_algorithms import (BlockEncoding, DoubleFactorizedEncoding,
                              PauliLCU, QSVT, Walk, chemistry)
from cudaq_algorithms import double_factorization as df

# H2 / STO-3G MO integrals (chemist (pq|rs) convention).
ONE_BODY = np.array([[-1.25246357, 0.0], [0.0, -0.47594871]])
ERI = np.zeros((2, 2, 2, 2))
ERI[0, 0, 0, 0] = 0.67449876
ERI[1, 1, 1, 1] = 0.69716349
ERI[0, 0, 1, 1] = ERI[1, 1, 0, 0] = 0.66347258
ERI[0, 1, 0, 1] = ERI[1, 0, 1, 0] = 0.18128881
ERI[0, 1, 1, 0] = ERI[1, 0, 0, 1] = 0.18128881
E_NUCLEAR = 0.71375697


def ground_energy(operator) -> float:
    return float(np.min(np.linalg.eigvalsh(np.asarray(operator.to_matrix()))))


def main() -> int:
    cudaq.set_target(os.environ.get("CUDAQ_DEFAULT_SIMULATOR", "qpp-cpu"))

    exact = chemistry.qubit_hamiltonian(ONE_BODY, ERI, scalar_offset=E_NUCLEAR)
    exact_energy = ground_energy(exact)

    # -- A. The compression dial -----------------------------------------
    full = df.explicit_double_factorization(ERI, threshold=0.0)
    print(
        "A. double-factorization compression (fewer leaves -> smaller alpha)")
    print(f"   {'leaves':>6} {'tensor error':>13} {'alpha':>8} "
          f"{'dE_ground (Ha)':>15}")
    for num_leaves in range(1, full.num_leaves + 1):
        truncated = df.explicit_double_factorization(ERI,
                                                     max_num_leaves=num_leaves)
        reconstructed = chemistry.qubit_hamiltonian(
            ONE_BODY, df.reconstruct_eri(truncated), scalar_offset=E_NUCLEAR)
        print(
            f"   {num_leaves:>6} {df.factorization_error(ERI, truncated):>13.2e}"
            f" {PauliLCU(reconstructed).alpha:>8.4f}"
            f" {ground_energy(reconstructed) - exact_energy:>+15.2e}")

    # -- B. One protocol, two encodings ----------------------------------
    lcu = PauliLCU(exact)
    dfe = DoubleFactorizedEncoding(ONE_BODY, ERI, scalar_offset=E_NUCLEAR)
    print(
        "\nB. PauliLCU and DoubleFactorizedEncoding both *are* BlockEncodings")
    for name, encoding in [("PauliLCU", lcu),
                           ("DoubleFactorizedEncoding", dfe)]:
        assert isinstance(encoding, BlockEncoding)
        # The exact same two calls work on either encoding:
        walk = Walk(encoding)
        transformer = QSVT(encoding)
        print(f"   {name:26s} system={encoding.num_system} "
              f"ancilla={encoding.num_ancilla} alpha={encoding.alpha:.4f} "
              f"-> Walk + QSVT built")

    print("\nOK — compression is a visible cost knob, and Walk/QSVT are "
          "generic over any BlockEncoding (see example 6 to add your own).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
