# Self Verification

After code changes, run validation commands (lint, typecheck, test) using shell tool.

**Steps:**
1. Find commands in README/package.json/pyproject.toml
2. Execute: `npm run lint`, `ruff check .`, `pytest`, etc.
3. Fix errors and re-run until passing

NEVER skip verification for non-trivial changes.
