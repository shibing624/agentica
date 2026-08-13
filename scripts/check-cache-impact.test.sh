#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
check="$repo_root/scripts/check-cache-impact.sh"

# 1. cache-sensitive file + filled fields -> pass
PR_BODY='Cache-impact: medium - reorders system prompt sections
Cache-guard: tests/agent/test_prompt_byte_stability.py
System-prompt-review: reviewed by xuming' \
	"$check" agentica/agent/prompts.py >/dev/null

# 2. non-sensitive file -> pass with no fields required
PR_BODY='' "$check" tests/agent/test_foo.py >/dev/null

# 3. missing both fields on a cache-sensitive file -> fail
if PR_BODY='' "$check" agentica/tools/base.py >/dev/null 2>&1; then
	echo "missing cache-impact declaration unexpectedly passed" >&2
	exit 1
fi

# 4. TODO placeholder -> fail
if PR_BODY='Cache-impact: TODO
Cache-guard: TODO' "$check" agentica/model/base.py >/dev/null 2>&1; then
	echo "TODO placeholder unexpectedly passed" >&2
	exit 1
fi

# 5. missing System-prompt-review on a system-prompt-sensitive file -> fail
if PR_BODY='Cache-impact: medium - changed
Cache-guard: tests/agent/test_prompt_byte_stability.py' \
	"$check" agentica/agent/prompts.py >/dev/null 2>&1; then
	echo "missing system-prompt-review unexpectedly passed" >&2
	exit 1
fi

# 6. n/a System-prompt-review on a system-prompt-sensitive file -> fail
if PR_BODY='Cache-impact: medium - changed
Cache-guard: tests/agent/test_prompt_byte_stability.py
System-prompt-review: n/a' \
	"$check" agentica/agent/prompts.py >/dev/null 2>&1; then
	echo "n/a system-prompt-review unexpectedly passed" >&2
	exit 1
fi

echo "cache impact contract tests: PASS"
