import { statWorkspaceFiles } from "../api";
import { isAbsOrHomePath, splitExternalTarget } from "./filePath";

/** Coalesce MessageFilesCard stats: many messages mount in one tick and
 *  would each POST the same workspace. One flush per root per ~32ms. */
type Waiter = {
  root: string;
  paths: string[];
  resolve: (existing: string[]) => void;
};

const buckets = new Map<string, Set<string>>();
const waiters: Waiter[] = [];
let flushTimer = 0;

function addBucketPath(targetRoot: string, targetRel: string) {
  if (!targetRoot || !targetRel) return;
  let set = buckets.get(targetRoot);
  if (!set) {
    set = new Set();
    buckets.set(targetRoot, set);
  }
  set.add(targetRel);
}

export function statWorkspaceFilesBatched(root: string, paths: string[]): Promise<string[]> {
  if (paths.length === 0) return Promise.resolve([]);
  return new Promise((resolve) => {
    for (const p of paths) {
      if (isAbsOrHomePath(p)) {
        const { targetRoot, targetRel } = splitExternalTarget(p, root);
        addBucketPath(targetRoot, targetRel);
      } else if (root) {
        addBucketPath(root, p);
      }
    }
    waiters.push({ root, paths, resolve });
    if (!flushTimer) flushTimer = window.setTimeout(flush, 32);
  });
}

async function flush() {
  flushTimer = 0;
  const jobs = [...buckets.entries()];
  const pending = waiters.splice(0, waiters.length);
  buckets.clear();
  const foundByRoot = new Map<string, Set<string>>();
  await Promise.all(
    jobs.map(async ([targetRoot, relSet]) => {
      const { ok, data } = await statWorkspaceFiles(targetRoot, [...relSet]);
      foundByRoot.set(targetRoot, new Set(ok ? (data?.existing || []) : []));
    }),
  );
  for (const w of pending) {
    w.resolve(
      w.paths.filter((p) => {
        if (isAbsOrHomePath(p)) {
          const { targetRoot, targetRel } = splitExternalTarget(p, w.root);
          return Boolean(foundByRoot.get(targetRoot)?.has(targetRel));
        }
        return Boolean(w.root && foundByRoot.get(w.root)?.has(p));
      }),
    );
  }
}
