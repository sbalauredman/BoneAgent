#!/usr/bin/env bash
set -euo pipefail
python -m boneagent.console.train --config configuration/pretraining.yaml
python -m boneagent.console.train --config configuration/finetuning.yaml
