#!/usr/bin/env bash
set -euo pipefail

# Cache guard: run the byte-stability / effect tests that pin the prompt-cache
# contract (Reasonix scripts/cache-guard.sh 移植). These are the tests that must
# pass before any cache-sensitive change merges:
#   - system prompt byte stability (incl. cross-PYTHONHASHSEED)
#   - tools schema byte stability (incl. cross-cwd / cross-process)
#   - wire payload allowlist (no local-only field leaks into provider bytes)
#   - use_capability proxy schema stability (deferred MCP churn must not drift schema)

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$root"

tests=(
  tests/agent/test_prompt_byte_stability.py
  tests/agent/test_tools_schema_stability.py
  tests/model/test_wire_payload_allowlist.py
  tests/tools/test_use_capability.py
)

python -m pytest "${tests[@]}" -v --tb=short
