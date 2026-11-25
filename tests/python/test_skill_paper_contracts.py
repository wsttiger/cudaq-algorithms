"""Resolve the new skill's claimed public contracts against the runtime checkout."""
import importlib
import inspect
import json
from pathlib import Path
import numpy as np
from cudaq_algorithms import PauliLCU,PhaseEstimation,UnaryPhaseEstimation


def test_paper_record_symbols_resolve_to_runtime_api():
    root=Path(__file__).resolve().parents[2]
    registry=json.loads((root/'skills-dev/cudaq-algorithms-dev/coverage/features.json').read_text())
    chosen={'basis-change-plan','basis-change-programmable','basis-change-phase-gradient',
            'primitives-phase-gradient','primitives-qrom-erasure','thc-domain','thc-index',
            'thc-pair-prepare','qubitization-phase-estimation','qubitization-unary-phase-estimation'}
    for row in registry['features']:
        if row['id'] not in chosen:continue
        for symbol in row['symbols']:
            parts=symbol.split('.')
            for end in range(len(parts),0,-1):
                try:obj=importlib.import_module('.'.join(parts[:end]))
                except ModuleNotFoundError:continue
                for name in parts[end:]:obj=getattr(obj,name)
                assert callable(obj),symbol
                break
            else:raise AssertionError(symbol)


def test_documented_qpe_factories_and_signed_count_decoder():
    encoding=PauliLCU([(1.,'Z')])
    binary=PhaseEstimation(encoding);unary=UnaryPhaseEstimation(encoding,queries=2)
    assert 'num_bits' in inspect.signature(binary.kernel).parameters
    assert 'state_prep' in inspect.signature(unary.kernel).parameters
    assert unary.num_states==6 and unary.resources['total_select_queries']==3
    # Explicit CUDA-Q bit-zero-first strings at signed cosine endpoints.
    binary_values=binary.energies_from_counts({'000':3,'001':1},3,top=2)
    unary_values=unary.energies_from_counts({'000':3,'001':1},top=2)
    np.testing.assert_allclose(binary_values,unary_values,atol=1e-12)
    np.testing.assert_allclose(binary_values,[[-1.,.75],[1.,.25]],atol=1e-12)
