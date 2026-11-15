/**
 * The three approval tiers, and what a stored value means today.
 *
 * `ask` is the pre-rename name of exactly the same tier as `read-only`: file
 * writes and mutating commands need approval. It is mapped forward rather than
 * treated as junk. Falling back to the default would be `auto` (Workspace
 * Write), which lets in-workspace writes and ordinary commands run unasked —
 * a saved Read Only choice must not restart one tier looser.
 *
 * Anything that was never a tier (a typo, a hand-edit) still falls back to the
 * `auto` default: interpreting it would be a guess about intent.
 */
export const APPROVAL_MODES = ["read-only", "auto", "allow-all"] as const;

export type ApprovalMode = (typeof APPROVAL_MODES)[number];

const PRE_RENAME: Record<string, ApprovalMode> = { ask: "read-only" };

export function normalizeApprovalMode(stored: unknown): ApprovalMode {
  if (typeof stored !== "string") return "auto";
  const value = stored.trim().toLowerCase();
  const migrated = PRE_RENAME[value];
  if (migrated) return migrated;
  return (APPROVAL_MODES as readonly string[]).includes(value) ? (value as ApprovalMode) : "auto";
}

/**
 * The tier one chat runs with.
 *
 * A chat that stored its own tier keeps it. A conversation that existed before
 * tiers were per-chat has no recorded value to restore, so it uses the gateway
 * default (`auto` — what the server applies when nothing overrides it); it must
 * NOT inherit the account's current pick, which answers "what should a new chat
 * start on" and is not a decision about that conversation.
 *
 * A brand-new chat (`hasLog === false`, or no session id yet) starts on the
 * pick. `hasLog` is the server-side log size, not the in-memory message list:
 * the first send pushes the user bubble before this is resolved, so the message
 * list cannot tell a new chat from an old one.
 */
export function resolveChatTier(
  sessionId: string | null,
  stored: unknown,
  currentPick: ApprovalMode,
  hasLog: boolean,
): ApprovalMode {
  if (stored) return normalizeApprovalMode(stored);
  if (!sessionId || !hasLog) return currentPick;
  return "auto";
}
