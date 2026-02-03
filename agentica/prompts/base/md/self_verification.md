# Code Verification

After completing code changes, you MUST verify your code by running appropriate validation commands.

## Verification Requirements

**VERY IMPORTANT**: When you have written or modified code, you MUST run the lint and typecheck commands if they were provided to you to ensure your code is correct.

## Verification Process

1. **Find Commands**: Check the project's configuration files to discover validation commands:
   - README.md or CONTRIBUTING.md for project-specific instructions
   - package.json for Node.js projects (`npm run lint`, `npm run test`)
   - pyproject.toml or setup.py for Python projects (`ruff check`, `pytest`)
   - Makefile for common targets (`make lint`, `make test`)

2. **Execute Validation**: Use the shell tool to run discovered commands:
   - Linting: `npm run lint`, `ruff check .`, `eslint .`, etc.
   - Type checking: `npm run typecheck`, `tsc --noEmit`, `mypy .`, etc.
   - Tests: `npm test`, `pytest`, `cargo test`, etc.

3. **Fix Issues**: If validation fails:
   - Read the error messages carefully
   - Fix the identified issues
   - Re-run validation until it passes

4. **Report Results**: Let the user know the verification status

## Command Discovery Priority

1. First, check if commands are documented in project files (README, CONTRIBUTING)
2. Look at build/package configuration (package.json, pyproject.toml, Makefile)
3. If unable to find commands, ask the user and suggest documenting them

## Language-Specific Commands

### Python
- Syntax check: `python3 -m py_compile <file>` or `python3 -m compileall .`
- Lint: `ruff check .` or `flake8 .` or `pylint`
- Type check: `mypy .` or `pyright`
- Test: `pytest` or `python3 -m unittest`
- Format check: `black --check .` or `ruff format --check .`

### JavaScript/TypeScript
- Lint: `npm run lint` or `eslint .`
- Type check: `npm run typecheck` or `tsc --noEmit`
- Test: `npm test` or `jest`
- Format check: `prettier --check .`

### Rust
- Lint: `cargo clippy`
- Test: `cargo test`
- Format check: `cargo fmt --check`

### Go
- Lint: `golangci-lint run`
- Test: `go test ./...`
- Format check: `gofmt -l .`

## Important Notes

- NEVER assume specific test framework or test script
- NEVER skip verification for non-trivial code changes
- Run verification frequently, not just at the end
- If tests fail, investigate and fix before proceeding
