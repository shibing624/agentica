#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
No-Docker coding-agent benchmarks for agentica.

  Aider Polyglot  — agent loop (file edit + pytest). Primary coding score.
  DABench         — data-analysis agent loop (CSV + closed-form tags).
  LiveCodeBench   — single-shot generation. Bare-model baseline.
  BigCodeBench    — function-level + real libraries.
  EvalPlus        — HumanEval+ smoke / pipeline check.

Usage:
  python evaluation/code_benchmark/run.py --bench polyglot --max-samples 1
  python evaluation/code_benchmark/run.py --bench dabench --max-samples 10
  python evaluation/code_benchmark/run.py --dry-run
  python evaluation/code_benchmark/run.py --bench all --max-samples 2 --model hy3
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evaluation.code_benchmark.cli_agents import (  # noqa: E402
    resolve_codex_reasoning_effort,
)
from evaluation.code_benchmark.common import (  # noqa: E402
    OUTPUT_DIR,
    build_model,
    now_tag,
    print_summary,
    resolve_responses_reasoning,
    summarize,
    write_json,
    write_jsonl,
)


BENCHES = ("polyglot", "dabench", "livecodebench", "bigcodebench", "evalplus")
CLI_AGENT_BENCHES = ("polyglot", "dabench")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="No-Docker coding-agent benchmarks")
    parser.add_argument(
        "--bench",
        default="polyglot",
        choices=(*BENCHES, "all"),
        help="Which benchmark to run (default: polyglot)",
    )
    parser.add_argument("--model", default=os.environ.get("CODE_BENCH_MODEL", "gpt-5.1"))
    parser.add_argument("--base-url", default=os.environ.get("OPENAI_BASE_URL"))
    parser.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY"))
    parser.add_argument("--max-samples", type=int, default=1, help="0 = all")
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--dry-run", action="store_true", help="Judge pipeline only, no LLM")
    parser.add_argument("--tries", type=int, default=2, help="Polyglot: feed pytest failures back")
    parser.add_argument("--tool-call-limit", type=int, default=40)
    parser.add_argument("--language", default="python", help="Polyglot language subset")
    parser.add_argument("--keywords", default=None, help="Polyglot exercise name filter")
    parser.add_argument("--polyglot-repo", type=Path, default=None)
    parser.add_argument("--lcb-release", default="release_v1")
    parser.add_argument("--lcb-start-date", default=None, help="YYYY-MM-DD contamination cut")
    parser.add_argument("--bcb-subset", default="full", help="full | hard")
    parser.add_argument("--bcb-split", default="instruct", help="instruct | complete")
    parser.add_argument("--no-plus", action="store_true", help="EvalPlus: skip extra tests")
    parser.add_argument("--timeout", type=int, default=90, help="Per-sample exec timeout (s)")
    parser.add_argument(
        "--agent",
        default="agentica",
        choices=("agentica", "claude", "codex"),
        help="Polyglot / DABench: which coding agent edits the files",
    )
    parser.add_argument(
        "--agent-timeout",
        type=int,
        default=300,
        help="Per-try wall-clock timeout for --agent claude|codex (s)",
    )
    parser.add_argument(
        "--extra-body",
        default=os.environ.get("CODE_BENCH_EXTRA_BODY"),
        help='JSON extra_body. Agentica/Codex Responses: {"reasoning": {"effort": "none"}} '
        'to turn thinking off. Chat Completions leftover: {"thinking_enabled": false}.',
    )
    parser.add_argument(
        "--http-timeout",
        type=float,
        default=float(os.environ.get("CODE_BENCH_HTTP_TIMEOUT", "600")),
        help="Per-request HTTP timeout in seconds (thinking models need this high)",
    )
    parser.add_argument(
        "--wire-api",
        default="responses",
        choices=("chat_completions", "responses"),
        help="Agentica only: OpenAIResponses (default) vs OpenAIChat",
    )
    return parser.parse_args()


def _output_dir(args: argparse.Namespace, bench: str) -> Path:
    root = args.output_dir or (OUTPUT_DIR / f"{now_tag()}-{bench}")
    root.mkdir(parents=True, exist_ok=True)
    return root


def run_dry(args: argparse.Namespace) -> int:
    selected = BENCHES if args.bench == "all" else (args.bench,)
    failed = 0
    for name in selected:
        print(f"\n--- dry-run {name} ---")
        try:
            if name == "polyglot":
                from evaluation.code_benchmark.polyglot import dry_run_polyglot
                payload = dry_run_polyglot(max_samples=1, repo=args.polyglot_repo)
                ok = payload["stub_failed"] and payload["example_passed"]
            elif name == "dabench":
                from evaluation.code_benchmark.dabench import dry_run_dabench
                payload = dry_run_dabench()
                ok = payload["gold_passed"] and payload["broken_failed"] and payload["empty_failed"]
            elif name == "livecodebench":
                from evaluation.code_benchmark.livecodebench import dry_run_lcb
                payload = dry_run_lcb()
                ok = payload["broken_failed"] and payload["n_tests"] > 0
            elif name == "bigcodebench":
                from evaluation.code_benchmark.bigcodebench import dry_run_bigcodebench
                payload = dry_run_bigcodebench()
                ok = payload["canonical_passed"] and payload["broken_failed"]
            else:
                from evaluation.code_benchmark.evalplus_he import dry_run_evalplus
                payload = dry_run_evalplus()
                ok = payload["canonical_passed"] and payload["broken_failed"]
        except Exception as exc:
            print(f"DRY-RUN ERROR: {exc}")
            failed += 1
            continue
        print(payload)
        print("OK" if ok else "PIPELINE MISMATCH")
        if not ok:
            failed += 1
    return 1 if failed else 0


def _cli_env(args: argparse.Namespace, extra_body, out: Path):
    from evaluation.code_benchmark.cli_agents import (
        claude_env,
        resolve_codex_reasoning_effort,
        write_isolated_codex_home,
    )

    if args.agent == "codex" and args.base_url:
        effort = resolve_codex_reasoning_effort(extra_body)
        codex_home = write_isolated_codex_home(
            out / "codex-home",
            model_id=args.model,
            base_url=args.base_url,
            reasoning_effort=str(effort),
        )
        env = {"CODEX_HOME": str(codex_home)}
        if args.api_key:
            env["OPENAI_API_KEY"] = args.api_key
        return env
    if args.agent == "claude" and args.base_url and args.api_key:
        effort = resolve_codex_reasoning_effort(extra_body)
        env = claude_env(
            config_dir=out / "claude-home",
            base_url=args.base_url,
            api_key=args.api_key,
            model_id=args.model,
        )
        env["CODE_BENCH_CLAUDE_EFFORT"] = effort
        return env
    return None


async def run_live(args: argparse.Namespace) -> int:
    extra_body = json.loads(args.extra_body) if args.extra_body else None
    if args.agent != "agentica":
        if args.bench == "all":
            selected = CLI_AGENT_BENCHES
        elif args.bench not in CLI_AGENT_BENCHES:
            print("--agent claude|codex only works with --bench polyglot|dabench")
            return 1
        else:
            selected = (args.bench,)
    else:
        selected = BENCHES if args.bench == "all" else (args.bench,)
    model = None
    if args.agent == "agentica":
        model = build_model(
            args.model,
            base_url=args.base_url,
            api_key=args.api_key,
            extra_body=extra_body,
            timeout=args.http_timeout,
            wire_api=args.wire_api,
        )
    exit_code = 0
    for name in selected:
        out = _output_dir(args, name) if args.bench != "all" else (args.output_dir or OUTPUT_DIR) / f"{now_tag()}-{name}"
        out.mkdir(parents=True, exist_ok=True)
        cli_env = _cli_env(args, extra_body, out) if name in CLI_AGENT_BENCHES else None
        if name == "polyglot":
            from evaluation.code_benchmark.polyglot import run_polyglot

            results = await run_polyglot(
                model=model,
                max_samples=args.max_samples,
                offset=args.offset,
                tries=args.tries,
                tool_call_limit=args.tool_call_limit,
                language=args.language,
                keywords=args.keywords,
                repo=args.polyglot_repo,
                output_dir=out,
                agent_kind=args.agent,
                agent_timeout=args.agent_timeout,
                cli_env=cli_env,
                cli_model=args.model if args.agent in ("codex", "claude") else "",
            )
        elif name == "dabench":
            from evaluation.code_benchmark.dabench import run_dabench

            results = await run_dabench(
                model=model,
                max_samples=args.max_samples,
                offset=args.offset,
                tool_call_limit=args.tool_call_limit,
                output_dir=out,
                agent_kind=args.agent,
                agent_timeout=args.agent_timeout,
                cli_env=cli_env,
                cli_model=args.model if args.agent in ("codex", "claude") else "",
            )
        elif name == "livecodebench":
            from evaluation.code_benchmark.livecodebench import run_livecodebench
            results = await run_livecodebench(
                model=model,
                max_samples=args.max_samples,
                offset=args.offset,
                release=args.lcb_release,
                start_date=args.lcb_start_date,
                timeout=args.timeout,
            )
        elif name == "bigcodebench":
            from evaluation.code_benchmark.bigcodebench import run_bigcodebench
            results = await run_bigcodebench(
                model=model,
                max_samples=args.max_samples,
                offset=args.offset,
                subset=args.bcb_subset,
                split=args.bcb_split,
                timeout=max(args.timeout, 30),
            )
        else:
            from evaluation.code_benchmark.evalplus_he import run_evalplus
            results = await run_evalplus(
                model=model,
                max_samples=args.max_samples,
                offset=args.offset,
                plus=not args.no_plus,
                timeout=args.timeout,
            )
        if args.agent == "codex":
            wire_api = "responses"
            reasoning_effort = resolve_codex_reasoning_effort(extra_body)
        elif args.agent == "agentica":
            wire_api = args.wire_api
            reasoning_effort = resolve_responses_reasoning(extra_body)
        else:
            wire_api = None
            reasoning_effort = None
        payload = summarize(
            name,
            args.model,
            results,
            extra={
                "agent": args.agent,
                "extra_body": extra_body,
                "wire_api": wire_api,
                "reasoning_effort": reasoning_effort,
            },
        )
        write_jsonl(out / "predictions.jsonl", [r.to_dict() for r in results])
        write_json(out / "summary.json", payload)
        print_summary(payload)
        print(f"wrote {out}")
        if payload["total"] == 0:
            exit_code = 1
    return exit_code


def main() -> int:
    args = parse_args()
    if args.dry_run:
        return run_dry(args)
    return asyncio.run(run_live(args))


if __name__ == "__main__":
    raise SystemExit(main())
