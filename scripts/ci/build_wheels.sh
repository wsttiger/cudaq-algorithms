#!/bin/bash

# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

set -euo pipefail

show_help() {
    echo "Usage: $0 [options]"
    echo "Options:"
    echo "  --build-type      Build type (default: Release)"
    echo "  --cuda-version    CUDA version or major version to build for (default: 12)"
    echo "  --cudaq-prefix    CUDA-Q install prefix (default: \$HOME/.cudaq)"
    echo "  --python-version  Python version to build wheel for (default: 3.11)"
    echo "  --devdeps         Build wheels suitable for internal testing"
    echo "  --version         Version of wheels to produce (default: 0.0.0)"
}

build_type=Release
cuda_version=12
cudaq_prefix="$HOME/.cudaq"
python_version=3.11
devdeps=false
wheels_version=0.0.0

while (( $# > 0 )); do
    case "$1" in
        --build-type)
            build_type="$2"
            shift 2
            ;;
        --cuda-version)
            cuda_version="$2"
            shift 2
            ;;
        --cudaq-prefix)
            cudaq_prefix="$2"
            shift 2
            ;;
        --python-version)
            python_version="$2"
            shift 2
            ;;
        --devdeps)
            devdeps=true
            shift
            ;;
        --version)
            wheels_version="$2"
            shift 2
            ;;
        -h|--help)
            show_help
            exit 0
            ;;
        *)
            echo "Error: Unknown argument $1" >&2
            show_help
            exit 1
            ;;
    esac
done

cuda_major=$(echo "$cuda_version" | cut -d . -f 1)
if [[ "$cuda_major" != "12" && "$cuda_major" != "13" ]]; then
    echo "Error: unsupported CUDA major version '$cuda_major'" >&2
    exit 1
fi

python=python${python_version}
arch=$(uname -m)
plat_args=()

if $devdeps; then
    if [[ "$arch" == "x86_64" ]]; then
        plat_args=(--plat manylinux_2_34_x86_64)
    elif [[ "$arch" == "aarch64" ]]; then
        plat_args=(--plat manylinux_2_34_aarch64)
    fi
elif [[ -f /opt/rh/gcc-toolset-12/enable ]]; then
    source /opt/rh/gcc-toolset-12/enable
fi

export CC=${CC:-gcc}
export CXX=${CXX:-g++}
export SETUPTOOLS_SCM_PRETEND_VERSION=$wheels_version
export CUDAQ_ALGORITHMS_VERSION=$wheels_version

if [[ ! -d "$cudaq_prefix/lib/cmake/cudaq" ]]; then
    echo "Error: CUDA-Q CMake package not found at $cudaq_prefix/lib/cmake/cudaq" >&2
    exit 1
fi

if [[ ! -f "pyproject.toml.cu${cuda_major}" ]]; then
    echo "Error: pyproject.toml.cu${cuda_major} not found" >&2
    exit 1
fi

rm -rf dist _skbuild pyproject.toml
cp "pyproject.toml.cu${cuda_major}" pyproject.toml

# Stage the experimental pure-Python modules as a cudaq_algorithms.experimental
# subpackage. The prototype is already a relative-import package
# (experiments/lcu_python/cudaq_algorithms), so its modules work unchanged as a
# subpackage; its __init__ also registers the cudaq.algorithms namespace. The
# Suzuki-Trotter module is appended as .trotter. Tests, conftest, and READMEs
# stay in experiments/; sim_utils and the runnable examples ship in the wheel.
experimental_pkg=python/cudaq_algorithms/experimental
rm -rf "$experimental_pkg"
if [[ -d experiments ]]; then
    mkdir -p "$experimental_pkg"
    cp experiments/lcu_python/cudaq_algorithms/*.py "$experimental_pkg/"
    cp experiments/suzuki_trotter_python/cudaq_algorithms/trotter.py \
       "$experimental_pkg/trotter.py"
    # Merge the Trotter simulation helper into the staged sim_utils (both
    # experiments ship a sim_utils module; the wheel carries one).
    {
        echo ""
        sed -n '/^def evolve/,$p' \
            experiments/suzuki_trotter_python/cudaq_algorithms/sim_utils.py
        echo ""
        echo '__all__ = list(__all__) + ["evolve"]'
    } >> "$experimental_pkg/sim_utils.py"
    cat >> "$experimental_pkg/__init__.py" <<'PYEOF'

# Staged addition for the wheel build: the Suzuki-Trotter module.
from . import trotter
_sys.modules["cudaq.algorithms.trotter"] = trotter
PYEOF
    # Stage the runnable examples, rewriting the in-tree imports (local
    # package on sys.path) to the installed-wheel form (experimental
    # subpackage registering cudaq.algorithms).
    mkdir -p "$experimental_pkg/examples"
    touch "$experimental_pkg/examples/__init__.py"
    sed -e '/^sys\.path\.insert/d' \
        -e 's|^import cudaq_algorithms .*|import cudaq_algorithms.experimental  # noqa: F401 — registers cudaq.algorithms|' \
        -e 's|^from cudaq_algorithms import sim_utils as sim$|from cudaq.algorithms import sim_utils as sim|' \
        -e 's|PYTHONPATH=/path/to/cudaq python3 example_hamiltonian_simulation.py|python3 -m cudaq_algorithms.experimental.examples.example_hamiltonian_simulation|' \
        experiments/lcu_python/example_hamiltonian_simulation.py \
        > "$experimental_pkg/examples/example_hamiltonian_simulation.py"
    sed -e '/^sys\.path\.insert/d' \
        -e 's|^from cudaq_algorithms import sim_utils, trotter$|import cudaq_algorithms.experimental  # noqa: F401 — registers cudaq.algorithms\nfrom cudaq.algorithms import sim_utils, trotter|' \
        -e 's|PYTHONPATH=/path/to/cudaq python3 example_trotter_chemistry.py|python3 -m cudaq_algorithms.experimental.examples.example_trotter_chemistry|' \
        experiments/suzuki_trotter_python/example_trotter_chemistry.py \
        > "$experimental_pkg/examples/example_trotter_chemistry.py"
    trap 'rm -rf "$experimental_pkg"' EXIT
fi

skbuild_args="-DCUDAQ_DIR=$cudaq_prefix/lib/cmake/cudaq;-DCMAKE_BUILD_TYPE=$build_type"
toolchain_dir="/opt/rh/gcc-toolset-12/root/usr/lib/gcc/${arch}-redhat-linux/12/"
if ! $devdeps && [[ -d "$toolchain_dir" ]]; then
    skbuild_args="$skbuild_args;-DCMAKE_CXX_COMPILER_EXTERNAL_TOOLCHAIN=$toolchain_dir"
fi
export SKBUILD_CMAKE_ARGS=$skbuild_args

echo "Building cudaq-algorithms-cu${cuda_major} $wheels_version for Python $python_version"
$python -m pip install --no-cache-dir build auditwheel
$python -m build --wheel

exclude_args=()
if [[ -d "$cudaq_prefix/lib" ]]; then
    while IFS= read -r lib; do
        exclude_args+=(--exclude "$lib")
    done < <(find "$cudaq_prefix/lib" -name "*.so" -printf "%P\n" | sort)
fi

mkdir -p /wheels
LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}:$(pwd)/_skbuild/lib" \
$python -m auditwheel -v repair dist/*.whl \
    "${exclude_args[@]}" \
    --wheel-dir /wheels \
    "${plat_args[@]}"

ls -la /wheels
