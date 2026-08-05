#!/usr/bin/env bash
set -euo pipefail
srun --nodes=4 --ntasks-per-node=64 --cpus-per-task=1 vasp_std

