#!/usr/bin/env bash
set -euo pipefail

rm -rf venv
python -m venv venv
. venv/bin/activate

pip install .
