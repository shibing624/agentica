#!/usr/bin/env bash
set -euo pipefail

# Cache-impact PR gate for agentica (Reasonix scripts/check-cache-impact.sh 移植).
#
# Guards the "cache stability is a first-class, test-guarded contract" discipline
# that v1 (docs/learn_cc/reasonix_v1.md) established: PRs touching cache-sensitive
# prompt/tool surfaces must declare an explicit Cache-impact and Cache-guard in
# the PR body. Sensitive-file detection is path-prefix based and repo-local.

usage() {
  cat <<'USAGE'
Usage: scripts/check-cache-impact.sh [changed-file ...]

Checks that PRs touching cache-sensitive prompt or tool surfaces include an
explicit cache-impact note and guard-test note in the pull request body.

Inputs:
  CACHE_IMPACT_PR_BODY or PR_BODY          Pull request body text.
  CACHE_IMPACT_PR_BODY_FILE               File containing the pull request body.
  CACHE_IMPACT_CHANGED_FILES              Newline-separated changed files.
  CACHE_IMPACT_CHANGED_FILES_FILE         File containing newline-separated changed files.
  CACHE_IMPACT_BASE_SHA / BASE_SHA        Diff base when files are not supplied.
  CACHE_IMPACT_HEAD_SHA / HEAD_SHA        Diff head when files are not supplied.
USAGE
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

body="${CACHE_IMPACT_PR_BODY:-${PR_BODY:-}}"
if [[ -n "${CACHE_IMPACT_PR_BODY_FILE:-}" ]]; then
  body="$(cat "$CACHE_IMPACT_PR_BODY_FILE")"
fi

changed_input=""
if [[ "$#" -gt 0 ]]; then
  changed_input="$(printf '%s\n' "$@")"
elif [[ -n "${CACHE_IMPACT_CHANGED_FILES_FILE:-}" ]]; then
  changed_input="$(cat "$CACHE_IMPACT_CHANGED_FILES_FILE")"
elif [[ -n "${CACHE_IMPACT_CHANGED_FILES:-}" ]]; then
  changed_input="$CACHE_IMPACT_CHANGED_FILES"
else
  base="${CACHE_IMPACT_BASE_SHA:-${BASE_SHA:-}}"
  head="${CACHE_IMPACT_HEAD_SHA:-${HEAD_SHA:-HEAD}}"
  if [[ -z "$base" ]]; then
    base="$(git merge-base origin/main "$head" 2>/dev/null || git merge-base main "$head" 2>/dev/null || git merge-base origin/master "$head" 2>/dev/null || git merge-base master "$head")"
  fi
  diff_base="$base"
  if merge_base="$(git merge-base "$base" "$head" 2>/dev/null)"; then
    diff_base="$merge_base"
  fi
  changed_input="$(git diff --name-only "$diff_base" "$head")"
fi

changed_files=()
while IFS= read -r file; do
  [[ -z "$file" ]] && continue
  changed_files+=("$file")
done <<< "$changed_input"

cache_sensitive=()
system_prompt_sensitive=()

# Cache-sensitive surfaces: prompt composition, tool schema serialisation,
# provider wire/cache-control, compression, cost accounting, and this gate itself.
for file in "${changed_files[@]:-}"; do
  case "$file" in
    agentica/agent/prompts.py|\
    agentica/agent/base.py|\
    agentica/agent/tools.py|\
    agentica/agent/config.py|\
    agentica/tools/base.py|\
    agentica/tools/mcp_tool.py|\
    agentica/tools/use_capability_tool.py|\
    agentica/model/base.py|\
    agentica/model/message.py|\
    agentica/model/usage.py|\
    agentica/model/openai/*|\
    agentica/model/anthropic/*|\
    agentica/compression/*|\
    agentica/memory/session_log.py|\
    agentica/aux_session.py|\
    agentica/cost_tracker.py|\
    scripts/check-cache-impact.sh|\
    scripts/cache-guard.sh)
      cache_sensitive+=("$file")
      ;;
  esac

  case "$file" in
    agentica/agent/prompts.py|\
    agentica/agent/base.py|\
    agentica/model/message.py|\
    agentica/compression/*|\
    agentica/memory/session_log.py)
      system_prompt_sensitive+=("$file")
      ;;
  esac
done

if [[ "${#cache_sensitive[@]}" -eq 0 ]]; then
  echo "No cache-sensitive prompt/tool files changed."
  exit 0
fi

failures=()

trim() {
  local s="$1"
  s="${s#"${s%%[![:space:]]*}"}"
  s="${s%"${s##*[![:space:]]}"}"
  printf '%s' "$s"
}

field_value() {
  local label="$1"
  local line
  line="$(printf '%s\n' "$body" | grep -Eim1 "^[[:space:]>#*_-]*${label}[[:space:]]*:" || true)"
  [[ -z "$line" ]] && return 1
  trim "${line#*:}"
}

require_field() {
  local label="$1"
  local value
  if ! value="$(field_value "$label")"; then
    failures+=("missing ${label}: line")
    return
  fi
  if [[ -z "$value" || "$value" =~ ^[Tt][Oo][Dd][Oo]($|[[:space:]:-]) || "$value" =~ ^[Tt][Bb][Dd]($|[[:space:]:-]) ]]; then
    failures+=("${label}: must be filled out")
  fi
}

require_review_field() {
  local label="$1"
  local value
  if ! value="$(field_value "$label")"; then
    failures+=("missing ${label}: line")
    return
  fi
  local lower
  lower="$(printf '%s' "$value" | tr '[:upper:]' '[:lower:]')"
  if [[ -z "$value" || "$lower" =~ ^todo($|[[:space:]:-]) || "$lower" =~ ^tbd($|[[:space:]:-]) || "$lower" =~ ^n/?a($|[[:space:]:-]) || "$lower" =~ ^none($|[[:space:]:-]) ]]; then
    failures+=("${label}: must name the explicit system-prompt review/approval")
  fi
}

require_field "Cache-impact"
require_field "Cache-guard"

if [[ "${#system_prompt_sensitive[@]}" -gt 0 ]]; then
  require_review_field "System-prompt-review"
fi

if [[ "${#failures[@]}" -gt 0 ]]; then
  {
    echo "Cache impact check failed."
    echo
    echo "Cache-sensitive files changed:"
    printf '  - %s\n' "${cache_sensitive[@]}"
    if [[ "${#system_prompt_sensitive[@]}" -gt 0 ]]; then
      echo
      echo "System-prompt-sensitive files changed:"
      printf '  - %s\n' "${system_prompt_sensitive[@]}"
    fi
    echo
    echo "Required PR body lines:"
    echo "  Cache-impact: <none|low|medium|high> - <reason>"
    echo "  Cache-guard: <focused guard test/command or existing guard rationale>"
    if [[ "${#system_prompt_sensitive[@]}" -gt 0 ]]; then
      echo "  System-prompt-review: <reviewer/approval note>"
    fi
    echo
    echo "Failures:"
    printf '  - %s\n' "${failures[@]}"
  } >&2
  exit 1
fi

echo "Cache impact check passed."
echo "Cache-sensitive files:"
printf '  - %s\n' "${cache_sensitive[@]}"
