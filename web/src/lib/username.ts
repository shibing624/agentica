/** Username: after cleanup, 2–32 chars, a lowercase letter then letters / digits / underscore.

 *  The same rule the server enforces (`accounts.normalize_account_id` then
 *  `_ID_RE`). It is a directory name (`users/<id>/` and the default Project
 *  folder), so separators and `..` are dropped rather than kept. */
export const USERNAME_PATTERN = /^[a-z][a-z0-9_]{1,31}$/;

export function normalizeUsername(raw: string): string {
  let s = raw.normalize("NFKC").trim().toLowerCase();
  s = s.replace(/[\s-]+/g, "_").replace(/[^a-z0-9_]/g, "");
  s = s.replace(/_+/g, "_").replace(/^_|_$/g, "");
  if (/^[0-9]/.test(s)) s = "u_" + s;
  if (s.length > 32) s = s.slice(0, 32).replace(/_+$/g, "");
  return s;
}
