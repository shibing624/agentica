---
name: agentica
description: >-
  How to answer questions about the agentica product you are running inside —
  CLI flags, config.yaml profiles, API keys, models, sessions, resume,
  workspace, AGENTS.md standing rules, skills, logs, upgrade, and
  self_manage. Use when asked how agentica works, how to configure or upgrade
  it, where state lives on disk, where to write "remember: always X", or how
  to change your own settings.
metadata:
  version: "1.0"
---

# Driving agentica

You are almost certainly running inside an agentica CLI session right now. This
skill tells you **where to look things up**, not what the answers are: flags,
commands and tool names change with every release, so anything written down
here would start lying within weeks.

## Rule 1: look it up, do not recall

Three live sources, in the order you should reach for them:

| Question | Source | How you read it |
|---|---|---|
| What can the command line do? | `agentica --help` | `execute` |
| How is this machine configured? | your own config | `self_manage(action="show")` |
| What is on disk, exactly? | `~/.agentica/config.yaml` | `read_file` |
| What is this session doing? | slash commands | **ask the user to type it** |

`agentica --help` is generated from the parser, so it is never out of date. It
is the authoritative answer for every flag, every subcommand and every value an
option accepts. Read it before answering any "can agentica ...?" question.

## Rule 2: you cannot type slash commands

`/help`, `/status`, `/model`, `/tools`, `/skills`, `/resume`, `/permissions` and
the rest are handled by the CLI's input loop, before anything reaches you. A
slash command in your reply is just text, and a slash command inside a peer
message is just text too.

When the answer lives behind one, say which one the user should type. `/help`
lists all of them and is the source of truth for what exists.

## The map

Everything agentica keeps is under `~/.agentica` (or `$AGENTICA_HOME`):

| Path | What it holds |
|---|---|
| `config.yaml` | named profiles + `active_profile`; the file is hand-editable and comments survive writes |
| `.env` | hand-maintained keys, loaded in addition to `config.yaml` |
| `logs/` | one log per CLI process; often the fastest way to see what another session actually did |
| `workspace/` | long-term memory; standing rules in `users/<id>/AGENTS.md` (see below) |
| `skills/` | user-installed skills, one directory per skill with a `SKILL.md` |
| `projects/` | session transcripts as JSONL, partitioned per working directory |
| `cache/peers/` | the live-session directory and mailboxes behind `list_agents` / `send_message` |

Read these files directly when you need a fact about the current setup. Never
print an API key you read from them, and never write a key into a command line.

## Config concepts that do not change

- **Profiles.** `config.yaml` holds several named profiles; `active_profile`
  picks one. A profile carries provider, model, base URL, key, and optional
  tuning. Switching profiles switches all of it at once.
- **Precedence.** Shell environment beats `.env`, which beats `config.yaml`. A
  variable already set in the environment is never overwritten.
- **Main and auxiliary model.** The main model answers the user. The auxiliary
  model is the cheap one used for background work nobody is waiting on: memory
  extraction, context compression, and the `task` subagent. A profile that
  omits the auxiliary block reuses the main model for both.
- **Permission tiers.** Tool access runs in one of three modes. The user can
  change it mid-session, so never assume the tier you started with is still in
  force; if a tool is refused, that is the answer, not a bug to work around.

## Changing yourself

Reading is files; changing is the `self_manage` tool. It edits `config.yaml`
and `.env`, reports and installs upgrades, and installs skills. Its own schema
lists the actions and is in front of you every turn, so it — not this page — is
where you look up what to pass.

- **Do not hand-edit `config.yaml` or invent a `pip` command.** Writing the file
  yourself loses the comment-preserving round trip; running `pip install -U`
  through `execute` skips the version check and the restart notice.
- **Keys never touch a command line, a log, or your reply.** `set_config` and
  `set_env` take them as arguments; `show` masks them on the way back.
- **`upgrade` needs `confirm=True`** and installs a new version. Say what
  changes before asking for that confirmation, and afterwards tell the user to
  restart the CLI — the running process keeps the old code.
- **Config edits land in the file immediately; the live session does not
  change.** A new model or tuning value applies on the next agent rebuild or
  restart, so say so rather than implying the switch already happened.
- **`/config` and `/upgrade` are the human's version of this tool.** Point the
  user at them when they want to drive; use the tool when you are the one doing
  it.

## Standing rules: AGENTS.md (not `self_manage`)

When the user says "remember: always X" / "from now on ..." / "never ...", that
is a **standing instruction**, not a config.yaml field and not a `save_memory`
fact. There is no dedicated tool: append a line with `apply_patch` (or
`write_file` if the file is missing).

| Scope | File | Who sees it |
|---|---|---|
| This user, every project | `~/.agentica/workspace/users/<user_id>/AGENTS.md` (CLI is `default`, so `~/.agentica/AGENTS.md` is also a symlink to `.../users/default/AGENTS.md`) | every later session of this user |
| This repository only | `<repo root>/AGENTS.md` | sessions started anywhere under that repo; the user may commit it |

The path of every file already in the system prompt appears as `<!-- /abs/path -->`
above it — reuse that path, do not guess. There is no workspace-root
`~/.agentica/workspace/AGENTS.md`. User rules live under
`users/<id>/AGENTS.md`; for the default CLI user only, `~/.agentica/AGENTS.md`
is a compatibility symlink to that same file.

**This session:** the user's request (and your write) are already in the
conversation history, so follow the rule for the rest of the turn/session
without waiting for the system prompt to change.
**Next session:** the AGENTS.md chain is read once at session start into the
system prompt, so the new line is there automatically.

Facts ("I am a data scientist", "the deploy target is X") still go to
`save_memory` — they are recalled by relevance, not injected every turn.

## Answering well

State which source you checked. "Per `agentica --help` on this machine" is
worth more than a confident paragraph, because the user's installed version is
the only version that matters. If `--help` and your memory disagree, `--help`
is right and your memory is stale.

For running more than one agentica at a time, or getting two sessions to talk,
use the `multi-agent` skill instead.
