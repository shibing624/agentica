---
name: agentica
description: How to answer questions about the agentica product you are running inside - flags, config profiles, models, sessions, tools, skills, self-management, and where state lives on disk. Use when asked how agentica works, how to configure it, upgrade it, or where something is stored.
when_to_use: agentica, cli flag, config.yaml, profile, api key, switch model, session, resume, workspace, skills, logs, where is, upgrade, self_manage
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
| How is this machine configured? | `~/.agentica/config.yaml` | `read_file` |
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
| `workspace/` | long-term memory (`AGENT.md`, `MEMORY.md`, daily notes) |
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

## Answering well

State which source you checked. "Per `agentica --help` on this machine" is
worth more than a confident paragraph, because the user's installed version is
the only version that matters. If `--help` and your memory disagree, `--help`
is right and your memory is stale.

For running more than one agentica at a time, or getting two sessions to talk,
use the `multi-agent` skill instead.
