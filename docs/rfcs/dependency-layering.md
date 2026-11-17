# RFC: Dependency Layering

Status: Deferred

## Problem

`pip install agentica` currently installs both SDK core dependencies and product defaults needed by CLI and `DeepAgent`. This preserves the one-command experience, but light SDK users also receive terminal UI, crawler, parser, and product-surface dependencies they may not need.

## Current Decision

Keep the current dependency set for the 1.6.x line. The CLI and `DeepAgent` must remain usable after a plain `pip install agentica`, and past attempts to move crawler dependencies into extras caused the default CLI path to crash.

## Candidate Future Shape

```text
agentica-core
    Agent / Runner / RunConfig / Tool / Model / RunResponse

agentica[workspace]
    Workspace / memory files / skills parsing

agentica[cli]
    prompt-toolkit / rich / DeepAgent product preset / builtin product tools

agentica[gateway]
    FastAPI service / channels / scheduled daily tasks

agentica[tools-*]
    Search / browser / research / media / finance tool bundles
```

## Migration Constraints

- Do not break `agentica` CLI for existing users.
- Do not make `DeepAgent()` crash after plain installation.
- Keep `Agent` and custom tool users on the smallest stable import path possible.
- Any split must include tests that install the minimal package and verify imports for `Agent`, `RunConfig`, `Tool`, and core providers.
- Any split must include tests that install the CLI extra and verify `agentica --query` can construct a `DeepAgent` with builtin tools.

## Recommendation

Do not split packages in this PR. First finish API layering documentation and preset boundaries (daily task failure visibility has shipped). Revisit this RFC when preparing a major version or when packaging telemetry shows dependency conflicts from light SDK users.
