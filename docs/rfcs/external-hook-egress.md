# RFC: External Hook Egress

Status: Proposed (step 1 of the agent-status integration)

Scope: **agentica side only.** The desktop side (VPet) is specified separately, in
`VPetMac/docs/agentica-hook-adapter.md`. The two documents are one contract; a
change to either side's field names has to change both.

## Problem

agentica can already tell a desktop app what a run is doing, but only through a
channel it invented for exactly one consumer: an HTTP-over-UDS sink whose default
socket is `~/Library/Application Support/VPet/notify.sock`
(`agentica/notify/config.py:31`). Those two lines are the only place VPet appears
in the package — the coupling is not deep, but it is **a private protocol with a
single client**: anyone else (another pet, another notifier, a user's own script)
has to write code against agentica's envelope instead of using a mechanism every
CLI already agrees on.

The alternative is the one every other coding CLI offers: **the user declares a
command, we run it at lifecycle points and hand it JSON on stdin**. Claude Code,
codex, opencode and Vibe Island all work this way. An integration between two
such programs is then a three-line config entry, not a protocol implementation.

This RFC specifies agentica's version of that mechanism. It is **not** a
compatibility layer that mimics Claude Code — see "Naming" below for the reason,
which is a decision, not an oversight.

## Non-goals

- **Not** copying Claude Code's field names. (`hookSpecificOutput.decision.behavior`
  is a Claude Code output encoding; `hook_event_name` values are its vocabulary.)
  We send our own names and document them. A consumer adds one adapter.
- **Not** deleting the notify sink in this change. The sink keeps working exactly
  as it does today while the new egress is verified; see "Phases".
- **Not** adding per-tool events (`PreToolUse` / `PostToolUse`). `RunEventType`
  has four lifecycle values on purpose and `run_events.py` states the discipline:
  *"every value listed here MUST have a real `_emit_event` call somewhere"*. There
  is no emit site for per-tool events today and adding one is a separate decision
  with its own failure mode (see "Explicitly not in step 1").
- **Not** an outbound webhook. Remote delivery is a different trust question
  (auth, retries, what leaves the machine) and a separate change.

## Warning: two different things are both called "hooks"

This RFC is about the **executable** kind (`settings.hooks` below). It is not the
Python callback classes already in the package: `AgentHooks` / `RunHooks`
(`agentica/hooks.py:37,62`) are in-process observers and stay as they are. Naming
the config block `settings.hooks` is deliberate (it is what users search for), but
the docstring for `agentica/hooks.py` should point here so the two are not confused.

## Design

### Config

```yaml
# ~/.agentica/config.yaml
settings:
  hooks:
    enabled: false            # default off: this runs the user's command on every run
    command: ["/abs/path/to/notifier", "--from-agentica"]
    events:
      run.started: true
      run.completed: true
      run.failed: true
      run.cancelled: true
      needs.approval: true
      needs.input: true
```

- `command` is an **argv list**, not a shell string: no quoting rules, no
  accidental shell expansion of the JSON, and no `shell=True` process group. A
  user who wants a shell writes `["/bin/sh","-c","…"]` explicitly.
- Env overrides follow the existing convention: `AGENTICA_HOOKS_ENABLED`,
  `AGENTICA_HOOKS_COMMAND`. Read **once at install time**, like the sink
  (`notify/config.py` docstring), so a mid-run config flip cannot leave a
  half-wired channel.
- `enabled: false` wires nothing — no thread, no process. Same "decided once"
  rule as `install_sink` (`notify/sink.py:412`).

### Events (six, in step 1)

| event | when | emitted from |
|---|---|---|
| `run.started` | a run begins | `run_events.py` `run_started`, already emitted (`runner/loop.py:605`) |
| `run.completed` | a run ends **and no goal lap is coming** | the sink's existing deferral logic (`notify/sink.py:482-498`) |
| `run.failed` | a run raised | `runner/loop.py:1630` |
| `run.cancelled` | user interrupt | `run_events.py` `run_cancelled` |
| `needs.approval` | a tool call is parked, waiting for the user | the `publish` hook of `make_approve` (`cli/approvals.py:270`) |
| `needs.input` | `ask_user_question` is parked | `wrap_ask_callback` (`notify/questions.py:71`) |

The schema gate refuses anything else, so the vocabulary cannot drift away from
what `RunEventType` actually emits.

**`goal.*` stays off this wire.** The reason recorded at `runner/core.py:93-103`
holds here too: one request becoming N runs is an agentica implementation detail,
and the `run.completed` deferral already carries the only part a consumer needs.

### Wire: stdin

```json
{
  "hook_event_name": "needs.approval",
  "session_id": "e549ef44-4836-47e4-81e5-92fe60e561f7",
  "prompt": "把这个模块的测试补上",
  "cwd": "/Users/xuming/Documents/Codes/VPetMac",
  "run_id": "…",
  "tool_name": "execute",
  "tool_call_id": "call_00_wNzJ…",
  "options": ["allow", "allow_prefix", "deny", "deny_prefix"],
  "question": "允许运行 `rm -rf build`？",
  "preview": "rm -rf build",
  "similar_label": "以后同类命令"
}
```

Field rules that are decisions, not defaults:

- **`prompt` is the run's anchor text**, exactly as the sink defines it
  (`notify/sink.py:668-690`): the user's message on an ordinary turn, the **goal
  objective** in a goal-driven session. A consumer may display it, but must not
  treat it as "what the user just typed" — the sink already documents that trap and
  a second transport must not reintroduce it. Clipped on our side (`_clip_text`).
- **The event name is in the JSON, not in argv.** Claude Code passes it as an
  argument because it does not put it on stdin; ours is one document with no
  second channel, so a consumer reads one thing and cannot get them out of sync.
- Optional fields are **omitted, not null** — a missing key and an empty string
  are different things, and a consumer that has to test both will eventually test
  one.
- `needs.input` carries `question` + optional `options`, and **no `decision`
  vocabulary**: the reply is a free string. Keeping a question out of the four
  approval words is the same distinction the sink draws at `notify/approvals.py:46`.
- **`options` is what the CLI actually offered**, i.e. `PendingApproval.options`
  (`agent/approvals.py:96`) after the CLI narrowed it via
  `visible_approval_decisions` (`cli/approvals.py:46-51`). Send the subset, not
  the four-word superset: a consumer that renders exactly what it is given cannot
  then show a button the terminal would reject.
- Payload discipline is unchanged from the sink: metadata only (`question`,
  `preview`, `tool`, `answer` clipped by the caller). **No full prompt, no tool
  output, no conversation history, no file contents.** A hook is user config, not
  a licence to widen the payload.

### Reply: stdout

Exactly one JSON document, nothing else:

```json
{"decision": "allow"}            // allow | allow_prefix | deny | deny_prefix
{"answer": "date-fns"}           // needs.input only
```

Exit codes, all three meanings fixed:

| exit | stdout | meaning |
|---|---|---|
| 0 | one JSON document | that is the user's answer |
| 0 | empty | **no decision** — the terminal prompt is still the answer path |
| non-zero | ignored | **no decision**, same as above |

There is no third meaning and no "error" channel: a hook that fails to reply and a
hook that declines to reply are the same thing to us, because in both cases the
user has not answered.

### Two semantics kept from the sink (do not drop these)

1. **No wait cap invented by us.** The harness never imposes a deadline on a
   `needs.*` reply. This is the rule already written at `notify-sink.md:18`
   ("用户还没答不是事件"). A number baked into this layer would mean a desktop
   answer got less time than a typed one, so there is no `settings.hooks.timeout`
   either. A command may set its own internal limit (the VPet bridge does); that
   is the command's business and it is visible in the user's own config.
2. **The terminal and the hook race; whoever answers first wins.** The tool call
   is *not* parked on the hook process. `publish` offers the request and returns;
   the approval machinery parks on the `ApprovalRegistry` future as it does today.
   If the user answers in the terminal first, the registry resolves and we
   **kill the hook process group** — its answer arrives second and is a race, not
   an error. The existing branch for exactly this is `notify/approvals.py:172`
   (*"False means the id is unknown or was already decided — normally the user
   answered in the terminal first. That is a race, not an error."*) and is reused
   as-is.

## Implementation notes

**Wiring point.** `Runner._emit_event` already dispatches side-mounted, after the
in-process callback, wrapped so a broken consumer cannot take a run down
(`runner/core.py:104-116`). Add the hook egress next to `notify_sink_dispatch`
there, under the same "observation must never break a run" rule. Do **not** put it
inside the sink: the sink is one consumer, and step 1 is about the transport.

**Process handling** (`needs.*` only; the four `run.*` events are fire-and-forget
with the same shape as a `POST /event`):

```python
proc = subprocess.Popen(argv, stdin=PIPE, stdout=PIPE, stderr=DEVNULL,
                        start_new_session=True)          # own process group
proc.stdin.write(json.dumps(payload).encode()); proc.stdin.close()
```

Run it on a daemon thread, wait on `proc.stdout` **and** on the registry future;
when the future resolves first, `os.killpg(os.getpgid(proc.pid), SIGKILL)`. The
package already relies on this pattern — `execute_tool.py:146-147` and
`utils/async_utils.py:73-82` both spawn with `start_new_session=True` and kill by
process group, for the reason recorded there: a child that spawned its own
children would otherwise survive. Teardown (turn cancelled, CLI exit) reaches the
thread through the same path that resolves the registry today (`deny_all`), so the
thread wakes and kills the child rather than leaking it.

**No token, and that is not a hole.** The sink needed a bearer token because it
listens on a socket any local process can reach (`notify-sink.md` §6.2). Here the
user supplies the command in their own config: there is no shared endpoint to
forge a request into. This is an escape hatch, not a service.

**A shell wrapper is a supported shape, on purpose.** `command` is argv, so a
consumer that needs things we do not send (its own pid chain, its controlling
terminal) writes `["/bin/sh","-c","… exec the-real-binary"]` and reads them from
the shell. That keeps such needs out of our wire format; it is also why the
`ppid`/`tty` question below is an open question rather than an omission.

**Failure ladder is the sink's, unchanged**: disabled, command missing, spawn
error, non-zero exit, empty stdout, unparseable JSON, unknown decision word — all
mean "no decision", and the terminal prompt remains. **Never synthesize `allow`.**

## Explicitly not in step 1

- **`PreToolUse` / `PostToolUse`.** `RunEventType` has no per-tool lifecycle value
  and `run_events.py`'s scope discipline forbids adding one before its emit site
  exists. Adding those means new `_emit_event` calls in the runner's tool loop and
  putting tool names on the wire — the latter conflicts with the metadata-only
  rule above and would have to be argued first. Revisit only when a consumer has a
  concrete need that `run.started` → `run.completed` cannot serve. **No step-1
  work depends on this.**
- **Deleting the sink.** See Phases.
- **Mimicking Claude Code's output encoding.** If a consumer wants agentica events
  to arrive in Claude Code's shape, the right place for that translation is an
  adapter on the consumer's side (VPet is adding one), not a lie in our payload.
- **`allow_prefix` downgrade.** The sink accepts four decision words
  (`_ALLOWED_DECISIONS`, `notify/sink.py:80`) and the reply contract keeps all
  four. A consumer that hides two of them is making a UI decision, not a protocol
  change.

## Phases

1. **This RFC**: hook egress behind `settings.hooks.enabled`, six events, race
   implementation. The notify sink is untouched and keeps running; both can be
   enabled at once, and a consumer may listen to both.
2. **Verification**: the real desktop app, over the real binary, must round-trip
   y/n — acceptance is below. Only after that is the question "does the sink's
   `/await` half still earn its place" a real question, and even then the thing to
   remove is **agentica's direct `/await` call site**, not the endpoint (the
   endpoint is the desktop app's, shared with the Claude Code bridge).
3. **Optional**: outbound webhook, separate RFC.

## Acceptance

- With `settings.hooks.enabled: true` and a command pointing at the desktop app's
  bridge, approving on the desktop resolves an approval that the terminal is also
  waiting on; answering in the terminal first kills the hook process and leaves no
  zombie (`ps` shows neither the bridge nor a stray process group).
- A hook that exits non-zero, or prints nothing, or prints junk → **the terminal
  prompt still answers the request**. Same for a bridge that is not installed.
- No deadline in agentica's layer: a reply that takes longer than any number the
  bridge might have chosen still lands, as long as the bridge itself is still
  waiting.
- `enabled: false` spawns nothing (no thread, no process) — verify by process
  count, not by reading the config.
- Existing suites stay green: `tests/notify/*` (2,719 lines) must not need edits
  for step 1.

## Open questions

- **Does `run.completed` still need the goal deferral here, or should the hook see
  every lap?** The sink holds it back (`notify/sink.py:482`). Keeping the same
  behaviour is the conservative choice and is what VPet already expects; a
  consumer that wants laps can be given a separate event later.
- **No `settings.hooks.timeout`.** A field that is parsed but never applied is a
  lie; a number that *is* applied would give a desktop answer less time than a
  typed one. The command sets its own limit if it wants one.
- **Does the replied-to question need correlation for `needs.input`?** The sink
  keys questions by position and applies the first usable answer
  (`notify/questions.py:53`); the approval path has `tool_call_id`. If a consumer
  ever renders two questions at once, questions need an id too. Out of step 1.
- **Do we add `ppid` / `tty` to the wire?** The Claude Code bridge sends both, and
  a desktop uses them to jump back to the terminal that owns the session. A
  consumer can recover them today with the shell wrapper described above
  (`$PPID` is the pid of the process that forked the hook, i.e. this agent), so
  the field is a convenience, not a capability. Decide only if a consumer reports
  the wrapper as a real problem.
