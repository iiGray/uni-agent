#!/usr/bin/env bash
# Compatibility entrypoint. The canonical quickstart lives alongside other training scripts.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"

exec bash "${REPO_ROOT}/examples/quickstart/training/train_mem_agent.sh" "$@"
