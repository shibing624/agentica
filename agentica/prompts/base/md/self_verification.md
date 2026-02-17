# Code Verification

After completing code changes, verify correctness by running validation commands.

## Process

1. **Discover commands** (first time only) from project config:
   - `package.json` scripts (lint, test, typecheck)
   - `pyproject.toml` / `Makefile` targets
   - Reuse discovered commands for subsequent verifications

2. **Run validation**: lint -> type check -> test

3. **Fix issues**: Read errors, fix, re-run until passing

## Rules
- Never assume specific lint/test frameworks - check project config first
- Never skip verification for non-trivial changes
- Run verification frequently, not just at the end
