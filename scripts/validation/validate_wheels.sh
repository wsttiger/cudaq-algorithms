#!/bin/bash

# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

# QA validation of the cudaq-algorithms wheel.
#
# The library ships ONE CUDA-agnostic py3-none-any wheel that depends on the
# `cudaq` metapackage, which selects the right cuda-quantum-cuNN for the
# machine at install time. Tests and examples are not packaged, so this
# script must run from a repository checkout at the release tag, with the
# wheel(s) under --wheels-dir.
#
# For every Python version it creates a fresh conda environment, installs
# the wheel (released cuda-quantum from pypi.nvidia.com by default, or a
# from-source cuda-quantum via --cudaq-wheels), runs the full pytest suite
# against every applicable simulator, and executes every example.
#
#   bash scripts/validation/validate_wheels.sh \
#       [--wheels-dir <dir>]        # default /root/wheels
#       [--cudaq-wheels <dir>]      # unreleased cuda-quantum wheels (Custom
#                                   # mode: installs cuda-quantum-cuNN from
#                                   # here, then the wheel with --no-deps)
#       [--cudaq-version <ver>]     # required with --cudaq-wheels (e.g. 0.15.99)
#       [--cuda-version <X.Y.Z>]    # default 12.6.0 (conda toolkit for GPU legs)
#       [--python-versions "3.11 3.12 3.13"]
#
# Simulator coverage:
#   * qpp-cpu       always. The suite's dense-reference tolerances
#                   (1e-10..1e-12) require fp64; this is the default the
#                   test conftest selects.
#   * nvidia-fp64   added automatically when a GPU is visible (nvidia-smi).
#   * nvidia (fp32) is deliberately NOT run against the pytest suite: the
#                   fp64 tolerances fail on fp32 by design, not by bug.
#
# Optional integrations exercised: pyscf and qsppack are installed (two
# examples need them); psi4 is NOT installed - its tests importorskip, and
# that skip is expected in the pytest summary. On CUDA-13 machines the
# cuda-quantum wheel pulls CuPy transitively, which needs a CUDA toolkit -
# the conda environment provides one on GPU legs.

set -uo pipefail

WHEELS_DIR=/root/wheels
CUDAQ_WHEELS=""
CUDAQ_VERSION=""
CUDA_VERSION="12.6.0"
PYTHON_VERSIONS=(3.11 3.12 3.13)

while [[ $# -gt 0 ]]; do
    case $1 in
        --wheels-dir)      WHEELS_DIR=$2; shift 2 ;;
        --cudaq-wheels)    CUDAQ_WHEELS=$2; shift 2 ;;
        --cudaq-version)   CUDAQ_VERSION=$2; shift 2 ;;
        --cuda-version)    CUDA_VERSION=$2; shift 2 ;;
        --python-versions) read -r -a PYTHON_VERSIONS <<< "$2"; shift 2 ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--wheels-dir <dir>] [--cudaq-wheels <dir>] \\"
            echo "          [--cudaq-version <ver>] [--cuda-version <X.Y.Z>] \\"
            echo "          [--python-versions \"3.11 3.12 3.13\"]"
            exit 1
            ;;
    esac
done

if [[ -n "$CUDAQ_WHEELS" && -z "$CUDAQ_VERSION" ]]; then
    echo "--cudaq-wheels requires --cudaq-version (e.g. 0.15.99)"
    exit 1
fi
if [[ ! -f pyproject.toml || ! -d tests/python ]]; then
    echo "Run this script from the repository root (tests/ and examples/ are"
    echo "not packaged in the wheel)."
    exit 1
fi

CURRENT_ARCH=$(uname -m)
CUDA_MAJOR=$(echo "$CUDA_VERSION" | cut -d . -f 1)
shopt -s nullglob
algorithm_wheels=("$WHEELS_DIR"/cudaq_algorithms-*.whl)
shopt -u nullglob
if [[ ${#algorithm_wheels[@]} -eq 0 ]]; then
    echo "No cudaq_algorithms wheel found under $WHEELS_DIR"
    exit 1
fi
ALGORITHMS_VERSION=$(basename "${algorithm_wheels[0]}" | cut -d - -f 2)

# fp64 simulators only (see header). GPU coverage via nvidia-fp64.
SIMULATORS=("qpp-cpu")
if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi >/dev/null 2>&1; then
    SIMULATORS+=("nvidia-fp64")
    HAVE_GPU=1
else
    HAVE_GPU=0
fi

# OpenBLAS can get bogged down on some machines if using too many threads.
export OMP_NUM_THREADS=8

num_failures=0
declare -a failure_log=()

record_failure() {
    num_failures=$((num_failures + 1))
    failure_log+=("$1")
    echo "FAILED: $1"
}

run_suite_and_examples() {
    local python_version=$1
    local env_name=cudaq-algorithms-env-${python_version}

    echo "=================================================================="
    echo "Python ${python_version}"
    echo "=================================================================="

    conda create -y -n "$env_name" "python=${python_version}" pip
    if [[ "$HAVE_GPU" == "1" ]]; then
        # The cu13 cuda-quantum wheel pulls CuPy, which needs a CUDA toolkit.
        conda install -y -n "$env_name" \
            -c "nvidia/label/cuda-${CUDA_VERSION}" cuda
    fi
    conda activate "$env_name"

    # --- install the wheel under test -----------------------------------
    if [[ -n "$CUDAQ_WHEELS" ]]; then
        # Custom mode: from-source cuda-quantum, then the wheel without its
        # resolver (cuda-quantum + numpy/scipy provided explicitly).
        pip install numpy scipy
        pip install --find-links "$CUDAQ_WHEELS" \
            "cuda-quantum-cu${CUDA_MAJOR}==${CUDAQ_VERSION}"
        pip install --no-deps --find-links "$WHEELS_DIR" \
            "cudaq-algorithms==${ALGORITHMS_VERSION}"
    else
        # PyPI mode: the `cudaq` metapackage resolves a released
        # cuda-quantum for this machine.
        pip install --extra-index-url https://pypi.nvidia.com/ \
            "cuda_toolkit[cudart]==${CUDA_VERSION%.*}.*" || true
        pip install --find-links "$WHEELS_DIR" \
            --extra-index-url https://pypi.nvidia.com/ \
            "cudaq-algorithms==${ALGORITHMS_VERSION}"
    fi

    # Test/example dependencies. psi4 is intentionally absent (importorskip).
    pip install pytest pyscf qsppack

    # The wheel must be importable and versions logged.
    python3 -c 'import cudaq; print("cudaq:", cudaq.__version__)'
    python3 -c 'import cudaq_algorithms; print(cudaq_algorithms.__version__)'
    if ! python3 -m pip show cudaq-algorithms >/dev/null 2>&1; then
        record_failure "py${python_version}: cudaq-algorithms not installed"
        conda deactivate
        return
    fi

    # --- the pytest suite, per fp64 simulator ---------------------------
    for simulator in "${SIMULATORS[@]}"; do
        echo "--- pytest tests/python on ${simulator} (Python ${python_version}) ---"
        if ! CUDAQ_DEFAULT_SIMULATOR="$simulator" \
                python3 -m pytest tests/python -v; then
            record_failure "py${python_version}: pytest on ${simulator}"
        fi
    done

    # --- every example, self-verifying ----------------------------------
    # Examples assert their own results and exit nonzero on failure. They
    # run on the fp64 CPU simulator: several pin it themselves, and the
    # env var covers the rest (fp32 defaults would fail their checks).
    shopt -s nullglob
    for example in examples/*/*.py; do
        case "$example" in
            # The encoding under demonstration is imported by its sibling
            # demo, not run standalone.
            examples/bring_your_own_encoding/df_encoding.py) continue ;;
        esac
        echo "--- ${example} (Python ${python_version}) ---"
        if ! CUDAQ_DEFAULT_SIMULATOR=qpp-cpu python3 "$example"; then
            record_failure "py${python_version}: ${example}"
        fi
    done
    shopt -u nullglob

    conda deactivate
}

echo "Validating cudaq-algorithms ${ALGORITHMS_VERSION} wheels on ${CURRENT_ARCH}"
echo "  wheels: ${WHEELS_DIR}  cuda: ${CUDA_VERSION}  gpu: ${HAVE_GPU}"
echo "  pythons: ${PYTHON_VERSIONS[*]}  simulators: ${SIMULATORS[*]}"

eval "$(conda shell.bash hook)"
for python_version in "${PYTHON_VERSIONS[@]}"; do
    run_suite_and_examples "$python_version"
done

echo "=================================================================="
if [[ $num_failures -gt 0 ]]; then
    echo "Validation FAILED on ${CURRENT_ARCH}: ${num_failures} failure(s)"
    for entry in "${failure_log[@]}"; do
        echo "  - $entry"
    done
    exit 1
fi
echo "Validation completed successfully on ${CURRENT_ARCH}!"
