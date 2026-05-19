# -*- coding: utf-8 -*-
"""
@author: XuMing(xuming624@qq.com)
@description: Command safety detection and secret redaction.

Dangerous command patterns (31 regex) detect risky shell operations.
Secret redaction removes API keys, tokens, passwords from tool output.
"""
import logging
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
