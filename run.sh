#!/usr/bin/env bash
# Run anything in this repo against an interpreter that has torch + CUDA.
#
#   ./run.sh -m pytest tests/ -q
#   ./run.sh examples/run_geometry.py
#   ./run.sh examples/run_multiplication.py --quick
#
# Why this exists: `python` on this machine is conda base (3.8, no torch), so a
# bare `python examples/...` fails partway through a run. Rather than install a
# second multi-gigabyte copy of torch on a disk that is 97% full, this points at
# the pixi environment next door, which already has exactly the torch/CUDA pair
# this code was developed against (2.3.1 / cu121, RTX 4090).
#
# On any other machine, build the environment properly instead:
#   conda env create -f environment.yml && conda activate dynamicmultinet
#
# Override the interpreter with DMN_PYTHON=/path/to/python ./run.sh ...

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_PY="$HERE/../RL_training/rl_training/.pixi/envs/default/bin/python"
PY="${DMN_PYTHON:-$DEFAULT_PY}"

if [[ ! -x "$PY" ]]; then
    echo "no usable interpreter at $PY" >&2
    echo "set DMN_PYTHON to one that has torch, or: conda env create -f environment.yml" >&2
    exit 1
fi

cd "$HERE"
exec "$PY" "$@"
