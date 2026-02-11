# Code Verification

After completing code changes, verify your code by running appropriate validation commands.

## Requirements

**IMPORTANT**: When writing or modifying code, run lint and typecheck commands to ensure correctness.

## Verification Process

1. **Find Commands** - Check project configuration:
   - README.md or CONTRIBUTING.md for instructions
   - package.json for Node.js (`npm run lint`, `npm run test`)
   - pyproject.toml or setup.py for Python (`ruff check`, `pytest`)
   - Makefile for common targets

2. **Execute Validation**:
   - Lint: `npm run lint`, `ruff check .`, `eslint .`
   - Type check: `npm run typecheck`, `tsc --noEmit`, `mypy .`
   - Test: `npm test`, `pytest`, `cargo test`

3. **Fix Issues** - Read errors, fix issues, re-run until passing

## Language-Specific Commands

| Language | Lint | Type Check | Test |
|----------|------|------------|------|
| Python | `ruff check .` | `mypy .` | `pytest` |
| JS/TS | `npm run lint` | `tsc --noEmit` | `npm test` |
| Rust | `cargo clippy` | - | `cargo test` |
| Go | `golangci-lint run` | - | `go test ./...` |

## Important

- Never assume specific test frameworks
- Never skip verification for non-trivial changes
- Run verification frequently, not just at the end
