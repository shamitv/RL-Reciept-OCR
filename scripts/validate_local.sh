#!/usr/bin/env bash
set -euo pipefail

pytest
python scripts/smoke_test.py
python inference.py --format text
