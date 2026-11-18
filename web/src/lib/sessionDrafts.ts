/** A newly created, never-sent conversation. `createSession` uses this to keep
 *  at most one "New Chat" per workspace: clicking a workspace's + repeatedly
 *  used to leave a pile of identical empty rows in the sidebar. */
export type DraftSession = {
  title: string;
  msgs: readonly unknown[];
  dir: string;
  ts?: number;
  archived?: boolean;
  running?: boolean;
  /** Sessions that came from the server have a log; a local draft does not. */
  sizeBytes?: number;
  userCount?: number;
};

function normDir(dir: string) {
  const d = (dir || "").trim();
  return d.replace(/[/\\]+$/, "") || d;
}

/**
 * Ids of unsent "New Chat" rows in this workspace, newest first.
 *
 * Only local drafts qualify: a server session has `sizeBytes` / `userCount`
 * and must never be collapsed into a draft even if it has no messages.
 */
export function emptyChatIds<T extends DraftSession>(
  sessions: Record<string, T>, dir: string,
): string[] {
  const target = normDir(dir);
  return Object.entries(sessions)
    .filter(([, sess]) =>
      !sess.archived
      && !sess.running
      && sess.title === "New Chat"
      && sess.msgs.length === 0
      && !sess.sizeBytes
      && !sess.userCount
      && normDir(sess.dir) === target)
    .sort((a, b) => (b[1].ts || 0) - (a[1].ts || 0))
    .map(([id]) => id);
}

/**
 * Local sessions the server catalog cannot list yet, which a refresh must keep.
 *
 * A "New Chat" with text typed into it, or a chat whose first turn has just
 * started, has no log on the server: `GET /api/sessions` cannot return it.
 * Merging the catalog over the local sessions used to drop everything the
 * catalog did not mention except the open chat, so such a chat vanished if a
 * refresh (focus sync, pulse, fork) landed while another chat was open. A chat
 * the server has seen (`sizeBytes`) and no longer lists was removed elsewhere
 * and goes.
 */
export function localOnlySessionIds<T extends { sizeBytes?: number }>(
  local: Record<string, T>,
  catalogIds: ReadonlySet<string>,
  deletedIds: ReadonlySet<string>,
): string[] {
  return Object.entries(local)
    .filter(([id, sess]) => !catalogIds.has(id) && !deletedIds.has(id) && !sess.sizeBytes)
    .map(([id]) => id);
}
