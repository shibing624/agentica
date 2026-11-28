#!/usr/bin/env python3
"""E2 — one session across surfaces, and an approval decided while a turn is parked.

A throwaway gateway (see e1_crash_recovery) plays the Web/Desktop surface. The
terminal surface is the real CLI — `agentica resume <session> --query ... --print`
— pointed at the same data root and working directory. One session per repetition:

  G1  gateway turn creates a file
  C1  CLI turn on the same session is asked what G1 did      (transcript: gateway -> CLI)
  C2  CLI turn creates a second file
  G2  gateway turn is asked which files exist                 (transcript: CLI -> gateway)
  AP  a gateway turn parks on a tool approval; the approval is allowed through
      the approvals API while the turn is parked, and the turn must finish

The IM surfaces are not driven here: they need a logged-in account and a phone.

    python3 evaluation/demo_track/e2_continuity.py --reps 5
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import shutil
import statistics
import subprocess
import sys
import threading
import time
import urllib.request
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
import e1_crash_recovery as e1  # noqa: E402

G1 = "Use write_file to create log.txt containing exactly: from-gateway\nThen reply with only the word DONE."
C1 = ("Do not call any tool. In one short sentence: which file did you create earlier in this conversation, "
      "and what was written in it?")
C2 = "Use write_file to create cli.txt containing exactly: from-cli\nThen reply with only the word DONE."
G2 = "Do not call any tool. In one short sentence: which two files were created in this conversation?"
AP = "Use write_file to create approved.txt containing exactly: yes. Then reply with only the word DONE."


def cli_turn(python: str, home: Path, work: Path, session_id: str, message: str, timeout: float) -> tuple[bool, str, float]:
    """One headless CLI turn on an existing session. Returns (exited 0, stdout, seconds)."""
    env = {
        "PATH": os.environ["PATH"],
        "HOME": os.environ["HOME"],
        "LANG": os.environ.get("LANG", "en_US.UTF-8"),
        "TMPDIR": os.environ.get("TMPDIR", "/tmp"),
        "AGENTICA_HOME": str(home),
        "PYTHONPATH": str(e1.REPO),
        "NO_COLOR": "1",
    }
    cmd = [python, "-c", "from agentica.cli.main import main; main()", "resume", session_id,
           "--query", message, "--print", "--work_dir", str(work), "--permissions", "allow-all",
           "--no-experience", "--no-enable-diagnostics"]
    t0 = time.monotonic()
    try:
        proc = subprocess.run(cmd, cwd=work, env=env, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return False, f"CLI timed out after {timeout}s", round(time.monotonic() - t0, 1)
    out = proc.stdout.strip() or proc.stderr.strip()[-300:]
    return proc.returncode == 0, out, round(time.monotonic() - t0, 1)


class Events(threading.Thread):
    """Reads one run's SSE stream into a list; each item is the published envelope."""

    def __init__(self, url: str, timeout: float) -> None:
        super().__init__(daemon=True)
        self.url, self.timeout = url, timeout
        self.items: list[dict] = []
        self.error: str | None = None
        self.t0 = time.monotonic()

    def run(self) -> None:
        try:
            req = urllib.request.Request(self.url, headers={"Accept": "text/event-stream"})
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                for raw in resp:
                    line = raw.decode("utf-8", errors="replace").strip()
                    if not line.startswith("data: "):
                        continue
                    payload = line[6:]
                    if payload == "[DONE]":
                        break
                    self.items.append({"t": round(time.monotonic() - self.t0, 2), **json.loads(payload)})
        except Exception as exc:  # noqa: BLE001 - the error is the measurement
            self.error = f"{type(exc).__name__}: {exc}"


def file_is(work: Path, name: str, body: str) -> bool:
    return (work / name).exists() and (work / name).read_text(encoding="utf-8").rstrip("\n") == body


def user_entry_keys(entries: list[dict], start: int) -> list[str]:
    """Keys of the first user entry from index ``start`` on, beyond the base ones."""
    base = {"type", "uuid", "parent_uuid", "session_id", "cwd", "timestamp", "version", "git_branch", "content"}
    for e in entries[start:]:
        if e.get("type") == "user":
            return sorted(k for k in e if k not in base)
    return []


def entry_point(sess: "e1.Session") -> dict:
    """What the session's meta file says about where it was last entered from."""
    return json.loads(sess.log_path.with_suffix(".meta.json").read_text(encoding="utf-8"))


def approval_flow(gw: e1.Gateway, session_id: str, work: Path) -> dict:
    r: dict[str, Any] = {}
    run = e1.http("POST", gw.url("/api/chat/runs"), body={
        "message": AP, "session_id": session_id, "work_dir": str(work), "approval_mode": "read-only",
    })
    events = Events(gw.url(f"/api/chat/runs/{run['run_id']}/events"), timeout=300)
    events.start()

    def parked() -> bool:
        return any(item.get("event") == "approval_request" for item in events.items)

    r["t_park"] = e1.wait_for(parked, 180, 0.1)
    if r["t_park"] is None:
        r["approval_error"] = "the turn never parked on an approval"
        events.join(5)
        return r
    request = next(item for item in events.items if item.get("event") == "approval_request")
    t_allow = time.monotonic()
    e1.http("POST", gw.url(f"/api/sessions/{session_id}/approvals/{request['data']['tool_call_id']}"),
            body={"decision": "allow"})
    events.join(240)
    r["t_allow_to_stream_end"] = None if events.is_alive() else round(time.monotonic() - t_allow, 2)
    r["event_counts"] = dict(collections.Counter(item.get("event") for item in events.items))
    r["sse_error"] = events.error
    r["file_ok"] = file_is(work, "approved.txt", "yes")
    r["run_still_active"] = e1.http(
        "GET", gw.url(f"/api/chat/runs/active?session_id={session_id}"))["run"] is not None
    return r


def repetition(python: str, home: Path, run: str, i: int, gw: e1.Gateway) -> dict:
    sid = f"s_e2_{run}_{i:02d}"
    work = e1.mkwork(home, sid)
    gw.start()
    sess = e1.Session(home, sid)
    r: dict[str, Any] = {"rep": i, "session_id": sid}

    ok, reply = e1.chat(gw, sid, work, G1, 180)
    r["g1_ok"] = ok and file_is(work, "log.txt", "from-gateway")
    r["source_after_g1"] = entry_point(sess)["source"]
    r["origin"] = entry_point(sess)["origin"]
    n1 = len(sess.entries())
    r["gateway_user_entry_keys"] = user_entry_keys(sess.entries(), 0)

    ok, out, secs = cli_turn(python, home, work, sid, C1, 240)
    r.update(c1_ok=ok, c1_s=secs, c1_out=out[:200])
    r["c1_saw_g1"] = ok and "log.txt" in out.lower() and "from-gateway" in out.lower()
    entries = sess.entries()
    r["log_grew_c1"] = len(entries) > n1
    r["cli_user_entry_keys"] = user_entry_keys(entries, n1)
    r["source_after_c1"] = entry_point(sess)["source"]

    ok, out, secs = cli_turn(python, home, work, sid, C2, 240)
    r.update(c2_ok=ok, c2_s=secs, c2_file_ok=file_is(work, "cli.txt", "from-cli"))
    r["source_after_c2"] = entry_point(sess)["source"]
    n3 = len(sess.entries())

    ok, reply = e1.chat(gw, sid, work, G2, 180)
    r["g2_ok"] = ok
    r["g2_saw_both"] = ok and "cli.txt" in reply.lower() and "log.txt" in reply.lower()
    r["log_grew_g2"] = len(sess.entries()) > n3
    r["source_after_g2"] = entry_point(sess)["source"]

    r.update(approval_flow(gw, sid, work))
    return r


def summarize(results: list[dict]) -> None:
    n = len(results)
    print(f"\n=== E2 summary ({n} repetitions) ===")
    for key, label in (
        ("g1_ok", "G1 gateway turn wrote the file"),
        ("c1_ok", "C1 CLI turn finished"),
        ("c1_saw_g1", "C1 CLI turn saw what the gateway turn did"),
        ("log_grew_c1", "C1 appended to the same session log"),
        ("c2_ok", "C2 CLI turn finished"),
        ("c2_file_ok", "C2 CLI turn wrote the file"),
        ("g2_ok", "G2 gateway turn finished"),
        ("g2_saw_both", "G2 gateway turn saw the CLI turn's file"),
        ("log_grew_g2", "G2 appended to the same session log"),
    ):
        print(f"  {label}: {sum(bool(r.get(key)) for r in results)}/{n}")
    parked = [r for r in results if r.get("t_park") is not None]
    print(f"  AP turn parked on an approval: {len(parked)}/{n}; "
          f"write finished after allow: {sum(bool(r.get('file_ok')) for r in parked)}/{len(parked)}; "
          f"run still listed active afterwards: {sum(bool(r.get('run_still_active')) for r in parked)}")
    for key in ("c1_s", "c2_s", "t_park", "t_allow_to_stream_end"):
        values = [r[key] for r in results if isinstance(r.get(key), (int, float))]
        if values:
            print(f"  median {key}: {statistics.median(values):.1f} s (max {max(values):.1f})")
    print(f"  user-entry keys, gateway: {sorted({k for r in results for k in r.get('gateway_user_entry_keys', [])})}")
    print(f"  user-entry keys, CLI:     {sorted({k for r in results for k in r.get('cli_user_entry_keys', [])})}")
    for key in ("origin", "source_after_g1", "source_after_c1", "source_after_c2", "source_after_g2"):
        print(f"  session meta {key}: {sorted({str(r.get(key)) for r in results})}")
    for r in results:
        if not (r.get("c1_ok") and r.get("c2_ok") and r.get("g2_ok") and r.get("file_ok")):
            print(f"  rep {r['rep']}: c1={r.get('c1_out')!r} c2_ok={r.get('c2_ok')} g2_ok={r.get('g2_ok')} "
                  f"approval={r.get('approval_error') or r.get('sse_error')}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--profile", default="hy4")
    ap.add_argument("--python", help="interpreter that can import agentica and agentica_gateway")
    ap.add_argument("--gateway-src", default=str(e1.REPO / "gateway" / "src"))
    ap.add_argument("--out", help="results file (jsonl)")
    ap.add_argument("--keep-home", action="store_true")
    args = ap.parse_args(argv)

    python = e1.resolve_python(args.python)
    home = e1.make_home(args.profile)
    run = time.strftime("%Y%m%d-%H%M%S")
    out = Path(args.out) if args.out else e1.REPO / "evaluation" / "demo_track" / "outputs" / f"e2-{run}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=e1.REPO, capture_output=True, text=True).stdout.strip()
    gw = e1.Gateway(python, Path(args.gateway_src), home, {})
    results: list[dict] = []
    print(f"home={home}\nout={out}\nprofile={args.profile} python={python}\ngateway_src={args.gateway_src} commit={head}", flush=True)
    with open(out, "a", encoding="utf-8") as fh:
        fh.write(json.dumps({"meta": True, "commit": head, "profile": args.profile, "reps": args.reps}) + "\n")
        try:
            for i in range(1, args.reps + 1):
                try:
                    r = repetition(python, home, run, i, gw)
                except Exception as exc:  # noqa: BLE001 - recorded, the run goes on
                    r = {"rep": i, "harness_error": f"{type(exc).__name__}: {exc}"}
                results.append(r)
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
                fh.flush()
                print("rep", i, json.dumps({k: v for k, v in r.items() if k not in ("event_counts", "session_id")},
                                          ensure_ascii=False)[:700], flush=True)
        finally:
            gw.sweep()
            if not args.keep_home:
                shutil.rmtree(home, ignore_errors=True)
    summarize(results)
    return 0


if __name__ == "__main__":
    sys.exit(main())
