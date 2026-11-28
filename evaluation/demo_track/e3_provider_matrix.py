#!/usr/bin/env python3
"""E3 — the same tool-use tasks on several OpenAI-compatible endpoints.

Each profile gets its own throwaway gateway (see e1_crash_recovery). Every task
is one fresh session and one chat turn, checked against the files it should
have produced and the tool rows it left in the session log — never against the
model's own claim that it succeeded.

    python3 evaluation/demo_track/e3_provider_matrix.py --smoke
    python3 evaluation/demo_track/e3_provider_matrix.py --profiles deepseek-flash,hy4,venus-glm-5.3-flash
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable

sys.path.insert(0, str(Path(__file__).resolve().parent))
import e1_crash_recovery as e1  # noqa: E402

Check = Callable[[str, list[dict], Path], tuple[bool, str]]

NOTES = '第一行 line one\n"quoted" line two\nline three'
JSON_TEXT = '{"k": "a\\"b", "arr": [1, 2, 3], "path": "C:\\\\tmp\\\\x"}'
LONG_LINES = 300


def norm(text: str) -> str:
    return text.strip().strip("`").strip().lower()


def tool_rows(entries: list[dict]) -> list[list[Any]]:
    return [[e["tool_name"], bool(e["is_error"])] for e in entries if e.get("type") == "tool"]


def max_calls_per_round(entries: list[dict]) -> int:
    return max((len(e.get("tool_calls") or []) for e in entries if e.get("type") == "assistant"), default=0)


def read(work: Path, name: str) -> str:
    return (work / name).read_text(encoding="utf-8").rstrip("\n")


def has(rows: list[list[Any]], name: str, error: bool = False) -> bool:
    return [name, error] in rows


# ---- tasks ----------------------------------------------------------------

def check_no_tool(reply: str, entries: list[dict], work: Path) -> tuple[bool, str]:
    rows = tool_rows(entries)
    if rows:
        return False, f"called tools: {rows}"
    return norm(reply) == "pong", reply


def check_execute(reply: str, entries: list[dict], work: Path) -> tuple[bool, str]:
    rows = tool_rows(entries)
    if not has(rows, "execute"):
        return False, f"no successful execute call: {rows}"
    return norm(reply) == "391", reply


def check_notes(reply: str, entries: list[dict], work: Path) -> tuple[bool, str]:
    rows = tool_rows(entries)
    problems = []
    if not has(rows, "write_file"):
        problems.append(f"write_file {rows}")
    if not has(rows, "read_file"):
        problems.append(f"read_file {rows}")
    if not (work / "notes.txt").exists() or read(work, "notes.txt").split("\n") != NOTES.split("\n"):
        problems.append(f"file={read(work, 'notes.txt')!r}" if (work / "notes.txt").exists() else "file missing")
    if norm(reply) != "3":
        problems.append(f"reply={reply!r}")
    return not problems, "; ".join(problems)


def setup_bug(work: Path) -> None:
    (work / "app.py").write_text("def add(a, b):\n    return a - b\n", encoding="utf-8")


def check_bug(reply: str, entries: list[dict], work: Path) -> tuple[bool, str]:
    rows = tool_rows(entries)
    text = read(work, "app.py")
    problems = []
    if not has(rows, "apply_patch"):
        problems.append(f"apply_patch {rows}")
    if "return a + b" not in text or "a - b" in text:
        problems.append(f"app.py={text!r}")
    if norm(reply) != "fixed":
        problems.append(f"reply={reply!r}")
    return not problems, "; ".join(problems)


def setup_needle(work: Path) -> None:
    (work / "alpha.py").write_text("x = 1\n# needle_here\n", encoding="utf-8")
    (work / "beta.md").write_text("nothing to see\n", encoding="utf-8")
    (work / "gamma.py").write_text("y = 2\n# needle_here too\n", encoding="utf-8")


def check_needle(reply: str, entries: list[dict], work: Path) -> tuple[bool, str]:
    rows = tool_rows(entries)
    names = set(re.findall(r"[A-Za-z]+\.py", reply))
    problems = []
    if names != {"alpha.py", "gamma.py"}:
        problems.append(f"reply={reply!r}")
    if has(rows, "read_file") or has(rows, "read_file", error=True):
        problems.append(f"read_file was used: {rows}")
    return not problems, "; ".join(problems)


def check_parallel(reply: str, entries: list[dict], work: Path) -> tuple[bool, str]:
    problems = []
    for name, body in (("x.txt", "xx"), ("y.txt", "yy")):
        if not (work / name).exists() or read(work, name) != body:
            problems.append(f"{name} wrong")
    if norm(reply) != "done":
        problems.append(f"reply={reply!r}")
    return not problems, "; ".join(problems)


def check_recover(reply: str, entries: list[dict], work: Path) -> tuple[bool, str]:
    rows = tool_rows(entries)
    problems = []
    if not has(rows, "read_file", error=True):
        problems.append(f"no failed read_file: {rows}")
    if has(rows, "write_file"):
        problems.append("created the file anyway")
    if (work / "missing.txt").exists():
        problems.append("missing.txt exists")
    if norm(reply) != "recovered":
        problems.append(f"reply={reply!r}")
    return not problems, "; ".join(problems)


def check_long(reply: str, entries: list[dict], work: Path) -> tuple[bool, str]:
    want = [f"line {i}" for i in range(1, LONG_LINES + 1)]
    problems = []
    if not (work / "lines.txt").exists() or read(work, "lines.txt").split("\n") != want:
        got = read(work, "lines.txt").split("\n") if (work / "lines.txt").exists() else []
        problems.append(f"file has {len(got)} lines, first wrong: "
                        f"{next((f'{i}: {l!r}' for i, l in enumerate(got, 1) if l != f'line {i}'), 'none')}")
    if norm(reply) != "done":
        problems.append(f"reply={reply!r}")
    return not problems, "; ".join(problems)


def check_chinese(reply: str, entries: list[dict], work: Path) -> tuple[bool, str]:
    problems = []
    if not (work / "说明.txt").exists() or read(work, "说明.txt") != "代理（agent）运行时「测试」":
        problems.append(f"file={read(work, '说明.txt')!r}" if (work / "说明.txt").exists() else "file missing")
    if norm(reply) != "完成":
        problems.append(f"reply={reply!r}")
    return not problems, "; ".join(problems)


def check_sequence(reply: str, entries: list[dict], work: Path) -> tuple[bool, str]:
    rows = tool_rows(entries)
    order = [name for name, _ in rows if not name.startswith("read")]
    problems = []
    if not has(rows, "execute"):
        problems.append(f"no execute: {rows}")
    if not (work / "b.txt").exists() or read(work, "b.txt") != "AB":
        problems.append(f"b.txt={read(work, 'b.txt')!r}" if (work / "b.txt").exists() else "b.txt missing")
    if norm(reply) != "ab":
        problems.append(f"reply={reply!r}")
    if order[:3] != ["write_file", "write_file", "execute"]:
        problems.append(f"order={order}")
    return not problems, "; ".join(problems)


def check_todos(reply: str, entries: list[dict], work: Path) -> tuple[bool, str]:
    rows = tool_rows(entries)
    problems = []
    if not has(rows, "write_todos"):
        problems.append(f"write_todos {rows}")
    if norm(reply) != "recorded":
        problems.append(f"reply={reply!r}")
    return not problems, "; ".join(problems)


def check_json(reply: str, entries: list[dict], work: Path) -> tuple[bool, str]:
    problems = []
    if not (work / "data.json").exists() or read(work, "data.json") != JSON_TEXT:
        problems.append(f"file={read(work, 'data.json')!r}" if (work / "data.json").exists() else "file missing")
    if norm(reply) != "saved":
        problems.append(f"reply={reply!r}")
    return not problems, "; ".join(problems)


TASKS: list[dict] = [
    {"name": "no_tool", "message": "Reply with only the word PONG.", "check": check_no_tool},
    {"name": "execute", "message": "Use the execute tool to run this exact command: echo $((17*23))\n"
                                   "Then reply with only the number it printed.", "check": check_execute},
    {"name": "write_read", "message": "Use write_file to create notes.txt with exactly these three lines and nothing else:\n"
                                      + NOTES + "\nThen use read_file to read notes.txt back, and reply with only the "
                                      "number of lines the file has.", "check": check_notes},
    {"name": "apply_patch", "setup": setup_bug, "message": "The file app.py in the current directory has a bug: its add "
                                                          "function returns the difference. Fix it with a single "
                                                          "apply_patch call so that it returns the sum. Then reply with "
                                                          "only the word FIXED.", "check": check_bug},
    {"name": "glob_grep", "setup": setup_needle, "message": "Find which .py files in the current directory contain the "
                                                           "text needle_here. Use glob and grep; do not read files with "
                                                           "read_file. Reply with only the file names, comma-separated, "
                                                           "sorted alphabetically.", "check": check_needle},
    {"name": "parallel", "message": "In one response, call write_file twice without waiting between the calls: create "
                                    "x.txt containing exactly xx, and y.txt containing exactly yy. Then reply with only "
                                    "the word DONE.", "check": check_parallel},
    {"name": "recover", "message": "Use read_file to read missing.txt in the current directory. The read will fail. "
                                   "Do not call any other tool and do not create the file. After the failure, reply "
                                   "with only the word RECOVERED.", "check": check_recover},
    {"name": "long_write", "message": f"Use write_file once to create lines.txt containing exactly {LONG_LINES} lines, "
                                      f"line i being: line i\nfor i from 1 to {LONG_LINES}. Do not use any other tool. "
                                      "Then reply with only the word DONE.", "check": check_long},
    {"name": "chinese", "message": "用 write_file 创建 说明.txt，内容恰好为一行：代理（agent）运行时「测试」\n"
                                   "然后只回复两个字：完成", "check": check_chinese},
    {"name": "sequence", "message": "Do these three steps strictly one at a time, each with its own tool call, waiting "
                                    "for each result before the next: (1) use write_file to create a.txt containing "
                                    "exactly A; (2) use write_file to create b.txt containing exactly AB; (3) use the "
                                    "execute tool to run: cat b.txt\nThen reply with only the output of step (3).",
     "check": check_sequence},
    {"name": "todos", "message": "Use write_todos once to record three tasks named alpha, beta and gamma, all with "
                                 "status pending. Then reply with only the word RECORDED.", "check": check_todos},
    {"name": "json", "message": "Use write_file to create data.json with exactly this content and nothing else:\n"
                                + JSON_TEXT + "\nThen reply with only the word SAVED.", "check": check_json},
]


# ---- run ------------------------------------------------------------------

def entries_of(home: Path, session_id: str) -> list[dict]:
    path = next((home / "projects").rglob(f"{session_id}.jsonl"), None)
    if path is None:
        return []
    entries = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            entries.append(json.loads(line))
        except ValueError:
            continue
    return entries


def run_profile(python: str, src: Path, profile: str, dotenv: Path, turn_timeout: float) -> list[dict]:
    home = e1.make_home(profile)
    if dotenv.exists():
        os.symlink(dotenv, home / ".env")
    gw = e1.Gateway(python, src, home, {})
    results: list[dict] = []
    try:
        gw.start()
        for i, task in enumerate(TASKS, 1):
            sid = f"s_e3_{i:02d}_{task['name']}"
            work = e1.mkwork(home, sid)
            task.get("setup", lambda w: None)(work)
            t0 = time.monotonic()
            ok, reply = e1.chat(gw, sid, work, task["message"], turn_timeout)
            r: dict[str, Any] = {
                "profile": profile, "task": task["name"], "wall_s": round(time.monotonic() - t0, 1),
                "transport_error": None if ok else reply,
            }
            entries = entries_of(home, sid)
            passed, detail = task["check"](reply or "", entries, work)
            r.update(ok=ok and passed, detail=detail, reply=(reply or "")[:160], tool_rows=tool_rows(entries),
                     n_tool_calls=sum(len(e.get("tool_calls") or []) for e in entries if e.get("type") == "assistant"),
                     max_calls_per_round=max_calls_per_round(entries))
            try:
                usage = e1.http("GET", gw.url(f"/api/sessions/{sid}/usage"), timeout=60)
                r["usage"] = {k: usage.get(k) for k in ("api_calls", "input_tokens", "output_tokens",
                                                        "cache_read_tokens", "cache_hit_percent", "cost_usd")}
                r["usage"]["steps"] = usage["stats"]["steps"]
            except Exception as exc:  # noqa: BLE001 - a missing usage payload is a result, not a crash
                r["usage"] = None
                r["usage_error"] = f"{type(exc).__name__}: {exc}"
            results.append(r)
            print(profile, task["name"], "ok" if r["ok"] else "FAIL", f"{r['wall_s']}s",
                  "" if r["ok"] else r["detail"][:150], flush=True)
    finally:
        gw.sweep()
        shutil.rmtree(home, ignore_errors=True)
    return results


def self_test() -> None:
    """Each checker must reject the failure it exists to catch."""
    import tempfile

    good_rows = [{"type": "tool", "tool_name": "write_file", "is_error": False, "tool_call_id": "c"}]
    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp)
        cases = [
            (check_no_tool, "PONG", [], True), (check_no_tool, "pong", good_rows, False),
            (check_execute, "391", [{"type": "tool", "tool_name": "execute", "is_error": False}], True),
            (check_execute, "391", [], False),
            (check_recover, "RECOVERED", [{"type": "tool", "tool_name": "read_file", "is_error": True}], True),
            (check_recover, "RECOVERED", [{"type": "tool", "tool_name": "read_file", "is_error": False}], False),
        ]
        for check, reply, entries, want in cases:
            got = check(reply, entries, work)[0]
            if got != want:
                sys.exit(f"self-test failed: {check.__name__}({reply!r}) -> {got}, expected {want}")
        (work / "lines.txt").write_text("\n".join(f"line {i}" for i in range(1, LONG_LINES)) + "\n", encoding="utf-8")
        if check_long("done", [], work)[0]:
            sys.exit("self-test failed: check_long accepted a file one line short")
        if max_calls_per_round([{"type": "assistant", "tool_calls": [{}, {}]}]) != 2:
            sys.exit("self-test failed: max_calls_per_round")
    print("self-test ok: task checkers")


def summarize(results: list[dict], profiles: list[str]) -> None:
    print("\n=== E3 summary ===")
    print("| profile | tasks ok | unexpected tool errors | p50 s | max s | in tok | out tok | cache hit % | cost USD |")
    print("|---|---|---|---|---|---|---|---|---|")
    for profile in profiles:
        rs = [r for r in results if r["profile"] == profile]
        if not rs:
            continue
        ok = sum(r["ok"] for r in rs)
        errors = 0
        for r in rs:
            failed = sum(1 for _, err in r["tool_rows"] if err)
            errors += failed if r["task"] != "recover" else abs(failed - 1)
        times = [r["wall_s"] for r in rs]
        usages = [r["usage"] for r in rs if r["usage"]]
        inp = sum(u["input_tokens"] or 0 for u in usages)
        outp = sum(u["output_tokens"] or 0 for u in usages)
        cached = sum(u["cache_read_tokens"] or 0 for u in usages)
        hit = f"{100 * cached / inp:.0f}" if inp else "-"
        cost = sum(u["cost_usd"] or 0 for u in usages)
        print(f"| {profile} | {ok}/{len(rs)} | {errors} | {statistics.median(times):.1f} | {max(times):.1f} | "
              f"{inp} | {outp} | {hit} | {cost:.4f} |")
    print("\n| task | " + " | ".join(profiles) + " |")
    print("|---|" + "---|" * len(profiles))
    for task in TASKS:
        cells = []
        for profile in profiles:
            r = next((r for r in results if r["profile"] == profile and r["task"] == task["name"]), None)
            cells.append("-" if r is None else ("ok" if r["ok"] else "**fail**"))
        print(f"| {task['name']} | " + " | ".join(cells) + " |")
    print("\nfailures:")
    for r in results:
        if not r["ok"]:
            print(f"  {r['profile']} / {r['task']}: transport={r['transport_error']} detail={r['detail'][:160]!r} "
                  f"reply={r['reply'][:60]!r}")
    parallel = [(r["profile"], r["max_calls_per_round"]) for r in results if r["task"] == "parallel"]
    print(f"\nparallel task, most tool calls in one assistant message: {parallel}")
    longs = [(r["profile"], sum(1 for name, _ in r["tool_rows"] if name == "write_file"))
             for r in results if r["task"] == "long_write"]
    print(f"long_write task, write_file calls made: {longs}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--smoke", action="store_true", help="first profile, two tasks")
    ap.add_argument("--profiles", default="deepseek-flash,hy4",
                    help="profile names from ~/.agentica/config.yaml, comma-separated")
    ap.add_argument("--python", help="interpreter that can import agentica and agentica_gateway")
    ap.add_argument("--gateway-src", default=str(e1.REPO / "gateway" / "src"))
    ap.add_argument("--dotenv", default=str(Path.home() / ".agentica" / ".env"),
                    help="dotenv the profiles' keys live in; symlinked into the throwaway data root")
    ap.add_argument("--turn-timeout", type=float, default=300.0)
    ap.add_argument("--out", help="results file (jsonl)")
    args = ap.parse_args(argv)

    self_test()
    global TASKS
    if args.smoke:
        TASKS = [TASKS[0], TASKS[1]]
    profiles = [p.strip() for p in args.profiles.split(",") if p.strip()]
    python = e1.resolve_python(args.python)
    out = Path(args.out) if args.out else e1.REPO / "evaluation" / "demo_track" / "outputs" / f"e3-{time.strftime('%Y%m%d-%H%M%S')}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=e1.REPO, capture_output=True, text=True).stdout.strip()
    results: list[dict] = []
    print(f"out={out}\nprofiles={profiles} python={python}\ngateway_src={args.gateway_src} commit={head}", flush=True)
    with open(out, "a", encoding="utf-8") as fh:
        fh.write(json.dumps({"meta": True, "commit": head, "profiles": profiles, "python": python,
                             "gateway_src": args.gateway_src, "tasks": [t["name"] for t in TASKS]}) + "\n")
        for profile in profiles:
            for r in run_profile(python, Path(args.gateway_src), profile, Path(args.dotenv), args.turn_timeout):
                results.append(r)
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
                fh.flush()
    summarize(results, profiles)
    return 0


if __name__ == "__main__":
    sys.exit(main())
