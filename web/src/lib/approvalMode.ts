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
