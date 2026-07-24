# -*- coding: utf-8 -*-
"""
@author: XuMing(xuming624@qq.com)
@description: Command safety detection and secret redaction.

Dangerous command patterns (31 regex) detect risky shell operations.
Read-only classification decides whether a command merely inspects state.
Secret redaction removes API keys, tokens, passwords from tool output.
"""
import logging
import os
import re
import shlex
from typing import List, Tuple

from agentica.security.redact import redact_sensitive_text

logger = logging.getLogger(__name__)

# ============== Dangerous Command Patterns ==============
# Adapted from hermes-agent tools/approval.py
# Each tuple: (regex_pattern, human_description)

DANGEROUS_PATTERNS: List[Tuple[str, str]] = [
    # Destructive file operations
    (r"\brm\s+(-[^\s]*\s+)*/\s*$", "delete root filesystem"),
    (r"\brm\s+(-[^\s]*\s+)*/\*", "delete all files in root"),
    (r"\bsudo\s+rm\b", "sudo rm"),
    (r"\brm\s+-[^\s]*r", "recursive delete"),
    (r"\brm\s+--recursive\b", "recursive delete (long flag)"),
    (r"\brm\s+-[^\s]*f[^\s]*\s+~", "force delete in home directory"),
    # Git history / working-tree destruction
    (r"\bgit\s+reset\s+--hard\b", "git reset --hard"),
    (r"\bgit\s+clean\s+-[a-z]*f[a-z]*d", "git clean -fd"),
    (r"\bgit\s+checkout\s+\.\s*$", "git checkout . (discard all)"),
    (r"\bgit\s+push\s+(-f|--force)\b", "git push --force"),
    # Overwrite redirection to sensitive locations
    (r">\s*/etc/", "overwrite system config under /etc"),
    (r">\s*~/\.ssh/", "overwrite SSH config"),
    (r">\s*/dev/(?!null\b|stderr\b|stdout\b)", "overwrite device file"),
    # Permissions
    (r"\bchmod\s+(-[^\s]*\s+)*(777|666|o\+[rwx]*w|a\+[rwx]*w)\b", "world-writable permissions"),
    (r"\bchmod\s+--recursive\b.*(777|666)", "recursive world-writable"),
    (r"\bchown\s+(-[^\s]*)?R\s+root", "recursive chown to root"),
    # Filesystem
    (r"\bmkfs\b", "format filesystem"),
    (r"\bdd\s+.*if=", "disk copy"),
    (r">\s*/dev/sd", "write to block device"),
    # SQL
    (r"\bDROP\s+(TABLE|DATABASE)\b", "SQL DROP"),
    (r"\bDELETE\s+FROM\b(?!.*\bWHERE\b)", "SQL DELETE without WHERE"),
    (r"\bTRUNCATE\s+TABLE\b", "SQL TRUNCATE"),
    # Remote code execution
    (r"curl\s+[^\n]*\|\s*(bash|sh|python|ruby|perl)", "pipe remote code to shell"),
    (r"wget\s+[^\n]*\|\s*(bash|sh|python|ruby|perl)", "pipe remote code to shell"),
    (r"curl\s+[^\n]*-o\s+/tmp/[^\s]*\s*&&\s*(bash|sh|chmod)", "download and execute"),
    # Process manipulation
    (r"\bkill\s+-9\s+-1\b", "kill all processes"),
    (r"\bkillall\s+-9\b", "kill all by name"),
    # Fork bomb
    (r":\(\)\s*\{\s*:\|:&\s*\}\s*;:", "fork bomb"),
    # System modification
    (r"/etc/sudoers\b", "sudoers modification"),
    (r"\bvisudo\b", "sudoers edit"),
    (r"\bpasswd\s+root\b", "change root password"),
    # SSH
    (r"authorized_keys", "SSH authorized_keys modification"),
    (r"\bssh-keygen\b.*-f\s+/", "SSH key generation in system path"),
    # Persistence
    (r"\bcrontab\s+-r\b", "clear all cron jobs"),
    (r"\.bashrc|\.zshrc|\.profile", "shell config modification"),
    # Exfiltration
    (r"curl\s+[^\n]*\$\{?\w*(KEY|TOKEN|SECRET|PASSWORD|CREDENTIAL|API)", "potential credential exfiltration via curl"),
    (r"wget\s+[^\n]*\$\{?\w*(KEY|TOKEN|SECRET|PASSWORD|CREDENTIAL|API)", "potential credential exfiltration via wget"),
    # Sensitive file access
    (r"\bcat\s+[^\n]*(\.env|credentials|\.netrc|\.pgpass)", "read sensitive files"),
    # Network scanning
    (r"\bnmap\b", "network scanning"),
    # Container escape
    (r"\bnsenter\b", "namespace enter (container escape)"),
]

_COMPILED_PATTERNS = [(re.compile(p, re.IGNORECASE), desc) for p, desc in DANGEROUS_PATTERNS]

_BLOCK_DESCRIPTIONS = frozenset({
    "delete root filesystem", "delete all files in root",
    "format filesystem", "fork bomb",
    "kill all processes", "disk copy", "write to block device",
})

# Top-level shell separators that chain independent commands.
# Order matters: longer tokens (&&, ||) must be tried before single chars (&, |).
_SEPARATOR_TOKENS = ("&&", "||", ";", "|", "&")


def split_compound_command(command: str) -> List[str]:
    """Best-effort split of a shell command on top-level separators.

    Splits on ``&&``, ``||``, ``;``, ``|``, and ``&`` while honoring single/
    double quotes via ``shlex``. This is NOT a full shell parser — its purpose
    is to surface dangerous *sub-commands* in patterns like
    ``cd / && rm -rf *`` or ``echo x; sudo rm /etc/foo`` so each segment can
    be scanned independently. Subshells ``$(...)`` / ``(...)`` and backticks
    are scanned as part of their enclosing segment (regex patterns still see
    them).

    On any tokenization error (unbalanced quotes, etc.) returns ``[command]``
    so the caller still scans the whole string.
    """
    if not command or not command.strip():
        return []

    try:
        lex = shlex.shlex(command, posix=True, punctuation_chars=True)
        lex.whitespace_split = True
        tokens = list(lex)
    except ValueError:
        return [command]

    segments: List[List[str]] = [[]]
    for tok in tokens:
        if tok in _SEPARATOR_TOKENS:
            segments.append([])
        else:
            segments[-1].append(tok)

    out: List[str] = []
    for seg in segments:
        if not seg:
            continue
        # Re-quote tokens that contain whitespace or shell metacharacters so
        # scanning sees a faithful representation. Plain words pass through
        # without quoting (regex patterns expect bare `rm -rf /`).
        joined = " ".join(
            shlex.quote(t) if (any(c.isspace() for c in t) or '"' in t or "'" in t) else t
            for t in seg
        )
        out.append(joined.strip())
    return out


# ============== Read-only Command Classification ==============
# Separate axis from DANGEROUS_PATTERNS above: that answers "is this
# destructive?", this answers "does this change any state?". `git commit` is
# not dangerous but is not read-only either.
#
# BEST-EFFORT, NOT A SANDBOX. Test runners and linters are allowed (see
# _RUNNER_COMMANDS) because a review subagent that cannot run the test suite is
# largely useless — but they execute arbitrary project code (conftest.py, npm
# scripts, Makefile targets) and therefore CAN write files. Real enforcement
# needs OS-level sandboxing. Treat this as a guardrail against the model
# casually running `git commit`, not as a security boundary against an
# adversary.

# Commands that cannot change state regardless of their arguments.
_READ_ONLY_COMMANDS = frozenset({
    "ls", "pwd", "echo", "date", "whoami", "hostname", "uname", "id",
    "which", "type", "env", "printenv", "cat", "head", "tail", "wc",
    "diff", "stat", "file", "du", "df", "tree", "basename", "dirname",
    "sort", "uniq", "cut", "grep", "rg", "jq", "ps", "true", "false",
})

# Commands whose read-only-ness depends on the subcommand.
_GUARDED_SUBCOMMANDS = {
    "git": frozenset({
        "diff", "diff-tree", "diff-index", "log", "show", "status", "blame",
        "rev-parse", "rev-list", "describe", "ls-files", "ls-tree", "shortlog",
        "cat-file", "show-ref", "name-rev", "merge-base", "count-objects",
        "whatchanged", "reflog", "grep",
    }),
    "npm": frozenset({"run", "run-script", "test", "t", "ls", "list", "outdated", "view", "why"}),
    "yarn": frozenset({"run", "test", "list", "why", "outdated", "info"}),
    "pnpm": frozenset({"run", "test", "list", "why", "outdated", "view"}),
    "cargo": frozenset({"test", "check", "clippy", "tree", "metadata", "bench"}),
    "go": frozenset({"test", "vet", "list", "doc", "env"}),
}

# Test runners, linters and type checkers. These execute project-controlled
# code, so they are read-only only by convention — see the caveat above.
_RUNNER_COMMANDS = frozenset({
    "pytest", "tox", "nox", "make", "jest", "vitest", "mocha", "rspec",
    "phpunit", "mypy", "pyright", "flake8", "pylint", "eslint", "tsc", "ruff",
})

# Interpreter `-m` modules treated as runners (`python -m pytest ...`).
_RUNNER_MODULES = frozenset({
    "pytest", "unittest", "tox", "nox", "mypy", "ruff", "flake8", "pylint",
    "pyright", "coverage",
})

_INTERPRETERS = frozenset({"python", "python3", "py"})

# Flags/subcommands that turn an otherwise read-only runner into a file writer.
# Deliberately long-form only: `-i` means "ignore case" to git grep/log/diff,
# so treating it as in-place-edit would refuse very common read-only commands.
_WRITE_FLAGS = frozenset({"--fix", "--write", "--in-place", "--apply", "--save"})
_WRITE_SUBCOMMANDS = frozenset({"fmt", "format"})

# `git -C path diff` — global options that consume the following token.
_GIT_GLOBAL_OPTS_WITH_VALUE = frozenset({"-C", "-c", "--git-dir", "--work-tree", "--namespace"})

# Redirections that do not write to a file: `2>&1`, `>&2`, `>/dev/null`.
_BENIGN_REDIRECT = re.compile(r"\d*>&\d*|&>\s*/dev/null|\d*>>?\s*/dev/null")


def _git_subcommand(tokens: List[str]) -> str:
    """Return the git subcommand, skipping global options like `-C <path>`."""
    i = 1
    while i < len(tokens):
        tok = tokens[i]
        if tok in _GIT_GLOBAL_OPTS_WITH_VALUE:
            i += 2
            continue
        if tok.startswith("-"):
            i += 1
            continue
        return tok
    return ""


def _segment_is_read_only(segment: str) -> Tuple[bool, str]:
    """Classify one already-split command segment."""
    try:
        tokens = shlex.split(segment)
    except ValueError:
        return False, f"cannot parse command: {segment!r}"
    if not tokens:
        return True, ""

    name = os.path.basename(tokens[0]).lower()

    if name in _READ_ONLY_COMMANDS:
        return True, ""

    if name in _GUARDED_SUBCOMMANDS:
        sub = _git_subcommand(tokens) if name == "git" else (tokens[1] if len(tokens) > 1 else "")
        if not sub:
            return False, f"`{name}` needs a read-only subcommand"
        if sub not in _GUARDED_SUBCOMMANDS[name]:
            return False, f"`{name} {sub}` is not a read-only subcommand"
        return _check_write_flags(name, tokens)

    if name in _RUNNER_COMMANDS:
        return _check_write_flags(name, tokens)

    if name in _INTERPRETERS:
        if len(tokens) > 2 and tokens[1] == "-m" and tokens[2] in _RUNNER_MODULES:
            return _check_write_flags(name, tokens)
        return False, f"`{name}` can run arbitrary code; only `-m {{{','.join(sorted(_RUNNER_MODULES))}}}` is allowed"

    return False, f"`{name}` is not a read-only command"


def _check_write_flags(name: str, tokens: List[str]) -> Tuple[bool, str]:
    """Reject runner invocations that rewrite files (`ruff --fix`, `cargo fmt`)."""
    for tok in tokens[1:]:
        if tok in _WRITE_FLAGS:
            return False, f"`{name} {tok}` rewrites files"
        if tok in _WRITE_SUBCOMMANDS:
            return False, f"`{name} {tok}` rewrites files"
    return True, ""


def is_read_only_command(command: str) -> Tuple[bool, str]:
    """Decide whether a shell command only inspects state.

    Every segment of a compound command is checked independently, so
    ``git log && rm -rf build`` is rejected on its second segment rather than
    passing on the benign first token.

    Returns:
        ``(True, "")`` if read-only, else ``(False, reason)``.
    """
    if not command or not command.strip():
        return False, "empty command"

    # Command substitution hides an unchecked command inside a benign one.
    if "$(" in command or "`" in command:
        return False, "command substitution is not allowed in read-only mode"

    # Any redirection that is not `2>&1` / `>/dev/null` writes a file.
    if ">" in _BENIGN_REDIRECT.sub("", command):
        return False, "output redirection is not allowed in read-only mode"

    segments = split_compound_command(command) or [command]
    for segment in segments:
        ok, reason = _segment_is_read_only(segment)
        if not ok:
            return False, f"{reason} (in: {segment!r})"
    return True, ""


def _scan_single(command: str) -> dict:
    """Scan one (already-split) command against DANGEROUS_PATTERNS."""
    for compiled, description in _COMPILED_PATTERNS:
        if compiled.search(command):
            if description in _BLOCK_DESCRIPTIONS:
                return {
                    "action": "block",
                    "reason": f"Blocked: {description}",
                    "pattern": description,
                }
            return {
                "action": "warn",
                "reason": f"Warning: {description}",
                "pattern": description,
            }
    return {"action": "allow", "reason": "", "pattern": ""}


def check_command_safety(command: str) -> dict:
    """Check a shell command for dangerous patterns.

    Two-pass scan:
      1. Whole-command scan — catches patterns that span shell separators
         (e.g. ``curl ... | bash``, fork bomb ``:(){ :|:& };:``, etc.) where
         the danger is in the *combination* of segments.
      2. Per-segment scan — splits on ``&&`` / ``||`` / ``;`` / ``|`` / ``&``
         and scans each sub-command, so attacks that hide a destructive
         segment behind a benign prefix (``echo ok && rm -rf /``) cannot
         slip past a clean whole-command scan.

    Severity policy: any ``block`` hit in EITHER pass dominates and wins.
    Otherwise the first ``warn`` (whole-command takes precedence over
    per-segment for stable reasons).

    Returns:
        {
            "action": "allow" | "warn" | "block",
            "reason": str,      # Empty if allowed
            "pattern": str,     # Pattern description if matched
        }
    """
    # Pass 1: whole command (preserves pipeline / fork-bomb context)
    whole = _scan_single(command)
    if whole["action"] == "block":
        return whole

    # Pass 2: per-segment
    sub_commands = split_compound_command(command) or [command]
    worst = whole if whole["action"] == "warn" else {"action": "allow", "reason": "", "pattern": ""}
    for sub in sub_commands:
        if sub == command:
            continue  # already scanned in pass 1
        result = _scan_single(sub)
        if result["action"] == "block":
            return {
                "action": "block",
                "reason": f"{result['reason']} (in: {sub!r})",
                "pattern": result["pattern"],
            }
        if result["action"] == "warn" and worst["action"] == "allow":
            worst = {
                "action": "warn",
                "reason": f"{result['reason']} (in: {sub!r})",
                "pattern": result["pattern"],
            }
    return worst
