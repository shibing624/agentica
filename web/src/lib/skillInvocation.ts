/**
 * Recognise a skill turn in a user message, so the bubble can show a chip
 * instead of the whole SKILL.md.
 *
 * Two shapes reach the transcript:
 * - the typed line, `/trigger args` (a live web turn before the log is read);
 * - the expanded prompt the log stores — `Skill.render_invocation()` in
 *   `agentica/skills/skill.py` (CLI and gateway both persist it):
 *
 *       Loading: <name>
 *       Base directory: <path>
 *
 *       <SKILL.md body>
 *       [\n\n---\n\n## Skill invocation\n\nThe user invoked this skill as: `…`\n\nArgument: <args>\n…]
 *
 * A bare "Loading: / Base directory:" prefix is NOT enough: users paste
 * SKILL.md bodies, logs and other people's prompts into the composer. The
 * expanded shape is only accepted when it carries the invocation trailer or
 * names an installed skill. Callers that render the chip must still keep the
 * original text reachable (see `SkillTurn`).
 *
 * Keep the markers below in step with render_invocation.
 */
export type SkillTurn = { name: string; args: string };

const HEADER_RE = /^Loading: ([^\n]+)\nBase directory: ([^\n]*)\n/;
const INVOCATION_MARK = "\n## Skill invocation\n";
const ARGUMENT_MARK = "\nArgument: ";
const ARGUMENT_ENDS = ["\nExpected argument: ", "\n\nThe argument names the target"];

export function parseSkillTurn(
  text: string,
  skills: Array<{ cmd: string; name: string }> = [],
): SkillTurn | null {
  if (!text) return null;
  const head = HEADER_RE.exec(text);
  if (head) {
    const name = head[1]!.trim();
    const baseDir = head[2]!.trim();
    const inv = text.lastIndexOf(INVOCATION_MARK);
    const installed = skills.some((s) => s.name === name || s.cmd === `/${name}`);
    const plausibleDir = /skills[\\/]/i.test(baseDir) || /\.agentica[\\/]/i.test(baseDir);
    // No arguments means render_invocation() returns the bare prompt with no
    // trailer. Replay parses without a skills list, so accept that shape when
    // the header's directory is this skill's own folder (its last segment is
    // the name). A pasted document rarely has that correspondence.
    const dirLeaf = baseDir.split(/[\\/]/).filter(Boolean).pop() || "";
    const ownDir = plausibleDir && dirLeaf.toLowerCase() === name.toLowerCase();
    // Pasted documents can start with the same two header lines. Require a real
    // invocation trailer, an installed skill name, or a matching skill folder
    // before hiding the user's text behind a chip.
    if (!((inv >= 0 && plausibleDir) || installed || ownDir)) return null;

    let args = "";
    if (inv >= 0) {
      const a = text.indexOf(ARGUMENT_MARK, inv);
      if (a >= 0) {
        const start = a + ARGUMENT_MARK.length;
        let end = -1;
        for (const marker of ARGUMENT_ENDS) {
          const at = text.indexOf(marker, start);
          if (at >= 0 && (end < 0 || at < end)) end = at;
        }
        // Without a terminator we cannot tell where the argument stops. Return
        // no argument rather than swallowing the rest of the message.
        if (end > start) args = text.slice(start, end).trim();
      }
    }
    return { name, args };
  }
  const typed = /^\s*(\/\S+)(?:\s+([\s\S]*))?$/.exec(text);
  if (!typed) return null;
  const cmd = typed[1]!.toLowerCase();
  const hit = skills.find((s) => s.cmd === cmd);
  return hit ? { name: hit.name, args: (typed[2] || "").trim() } : null;
}

export type SkillToken = { start: number; end: number; name: string };

/**
 * The leading `/trigger` of a composer draft when it names an installed skill.
 * Only the first token counts: the gateway expands a skill only when the
 * message starts with its trigger, so a `/word` later in the text is plain text.
 */
export function leadingSkillToken(
  text: string,
  skills: Array<{ cmd: string; name: string }>,
): SkillToken | null {
  const m = /^(\s*)(\/[^\s]+)(?=\s|$)/.exec(text);
  if (!m) return null;
  const cmd = m[2]!.toLowerCase();
  const hit = skills.find((s) => s.cmd === cmd);
  if (!hit) return null;
  const start = m[1]!.length;
  return { start, end: start + m[2]!.length, name: hit.name };
}
