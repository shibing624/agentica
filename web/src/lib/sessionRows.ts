export const COLLAPSED_SESSION_LIMIT = 5;

export type SessionRow = {
  id: string;
  session: {
    title: string;
    preview?: string;
    running?: boolean;
    pinned?: boolean;
  };
};

export function sessionMatches(row: SessionRow, projectName: string, query: string) {
  const q = query.trim().toLocaleLowerCase();
  if (!q) return true;
  return [row.session.title, row.session.preview || "", projectName]
    .some((value) => value.toLocaleLowerCase().includes(q));
}

export function collapseSessionRows<T extends SessionRow>(
  rows: T[],
  currentId: string | null,
  expanded: boolean,
  limit = COLLAPSED_SESSION_LIMIT,
): { visible: T[]; hiddenCount: number } {
  if (expanded) return { visible: rows, hiddenCount: 0 };
  let idleCount = 0;
  const visible = rows.filter((row) => {
    if (row.id === currentId || row.session.running || row.session.pinned) return true;
    idleCount += 1;
    return idleCount <= limit;
  });
  return { visible, hiddenCount: rows.length - visible.length };
}
