import type { ApiResult } from "../api";

export type PickOutcome =
  | { kind: "picked"; path: string }
  | { kind: "cancelled" }
  /** No native dialog to use (remote client, no chooser, request failed): show the in-page folder walk. */
  | { kind: "fallback" };

/** Read the answer of `POST /api/fs/pick`. Only a 200 means the OS dialog ran;
 *  anything else (409 remote, 501 no chooser, network) is "use the page's own picker". */
export function pickOutcome(res: ApiResult<{ path: string | null }>): PickOutcome {
  if (!res.ok || !res.data) return { kind: "fallback" };
  return res.data.path ? { kind: "picked", path: res.data.path } : { kind: "cancelled" };
}
