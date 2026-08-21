import { statWorkspaceFiles } from "../api";

/** Coalesce MessageFilesCard stats: many messages mount in one tick and
 *  would each POST the same workspace. One flush per root per ~32ms. */
type Waiter = {
  paths: string[];
  resolve: (existing: string[]) => void;
};

const buckets = new Map<string, { paths: Set<string>; waiters: Waiter[] }>();
let flushTimer = 0;

export function statWorkspaceFilesBatched(root: string, paths: string[]): Promise<string[]> {
  if (!root || paths.length === 0) return Promise.resolve([]);
  return new Promise((resolve) => {
    let bucket = buckets.get(root);
    if (!bucket) {
      bucket = { paths: new Set(), waiters: [] };
      buckets.set(root, bucket);
    }
    for (const p of paths) bucket.paths.add(p);
    bucket.waiters.push({ paths, resolve });
    if (!flushTimer) flushTimer = window.setTimeout(flush, 32);
  });
}

async function flush() {
  flushTimer = 0;
  const jobs = [...buckets.entries()];
  buckets.clear();
  for (const [root, bucket] of jobs) {
    const { ok, data } = await statWorkspaceFiles(root, [...bucket.paths]);
    const existing = new Set(ok ? (data?.existing || []) : []);
    for (const w of bucket.waiters) {
      w.resolve(w.paths.filter((p) => existing.has(p)));
    }
  }
}
