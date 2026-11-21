import { statWorkspaceFiles, type WorkspaceScope, type WorkspaceTarget } from "../api";
import { splitExternalTarget, toWorkspaceRelative } from "./filePath";

/** Coalesce MessageFilesCard stats: many messages mount in one tick and
 *  would each POST the same workspace. One flush per root per ~32ms. */
type Waiter = {
  root: string;
  scope?: WorkspaceScope;
  paths: string[];
  resolve: (existing: string[]) => void;
};

const buckets = new Map<string, { target: WorkspaceTarget; paths: Set<string> }>();
/** Last answer per `root\0path`. A chat re-opened, or a card remounted, draws
 *  from this at once instead of popping in after a round trip. */
const known = new Map<string, boolean>();
const scopeKey = (target: WorkspaceTarget) => typeof target === "string" ? target : `${target.session_id}\0${target.binding}`;
const knownKey = (root: WorkspaceTarget, path: string) => scopeKey(root) + "\0" + path;

/** The existing subset of `paths` if every one has been looked up before, else null. */
export function cachedExisting(root: WorkspaceTarget, paths: string[]): string[] | null {
  const out: string[] = [];
  for (const p of paths) {
    const hit = known.get(knownKey(root, p));
    if (hit === undefined) return null;
    if (hit) out.push(p);
  }
  return out;
}
const waiters: Waiter[] = [];
let flushTimer = 0;

function pathTarget(path: string, root: string, scope?: WorkspaceScope): [WorkspaceTarget, string] {
  const rel = scope ? toWorkspaceRelative(path, scope.root) : null;
  if (scope && rel !== null) return [scope, rel];
  const { targetRoot, targetRel } = splitExternalTarget(path, root);
  return [targetRoot, targetRel];
}

export function statWorkspaceFilesBatched(root: string, paths: string[], scope?: WorkspaceScope): Promise<string[]> {
  if (paths.length === 0) return Promise.resolve([]);
  return new Promise((resolve) => {
    for (const p of paths) {
      const [target, rel] = pathTarget(p, root, scope);
      if (!target || !rel) continue;
      const key = scopeKey(target);
      const bucket = buckets.get(key) ?? { target, paths: new Set<string>() };
      bucket.paths.add(rel);
      buckets.set(key, bucket);
    }
    waiters.push({ root, scope, paths, resolve });
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
    jobs.map(async ([key, { target, paths }]) => {
      const names = [...paths];
      const found = new Set<string>();
      for (let i = 0; i < names.length; i += 200) {
        const { ok, data } = await statWorkspaceFiles(target, names.slice(i, i + 200));
        if (ok) for (const path of data?.existing ?? []) found.add(path);
      }
      foundByRoot.set(key, found);
    }),
  );
  for (const w of pending) {
    const found = w.paths.filter((p) => {
      const [target, rel] = pathTarget(p, w.root, w.scope);
      return Boolean(foundByRoot.get(scopeKey(target))?.has(rel));
    });
    const present = new Set(found);
    for (const p of w.paths) known.set(knownKey(w.scope ?? w.root, p), present.has(p));
    w.resolve(found);
  }
}
