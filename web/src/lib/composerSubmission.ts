/**
 * Composer submission policy (deepseek-harness `submission-policy.ts`): one
 * preference decides what plain Enter means while the agent is busy, and the
 * accelerator chord always delivers the *other* mode.
 *
 * The preference is a setting, not a heuristic — a long turn where the model is
 * doing the right thing wants "queue next turn", while a turn that needs
 * correcting wants "steer". Neither can be inferred from the draft, so the user
 * picks the default once and flips it per message with the chord.
 *
 * `steeringAvailable` is the authority: a slash command, a document attachment,
 * or a turn that is already stopping cannot be steered, so the gesture degrades
 * to `queue` instead of being refused after the fact.
 */

export type BusyEnterBehavior = "queue" | "steer";

/** Plain Enter (or the Send button) vs the Cmd/Ctrl-accelerated chord. */
export type SubmitGesture = "enter" | "chord";

export type SubmitMode = "queue" | "steer";

export const DEFAULT_BUSY_ENTER: BusyEnterBehavior = "queue";

export function resolveSubmitMode(
  preferred: BusyEnterBehavior,
  running: boolean,
  gesture: SubmitGesture,
  steeringAvailable: boolean,
): SubmitMode {
  if (!running || !steeringAvailable) return "queue";
  if (gesture === "enter") return preferred;
  return preferred === "queue" ? "steer" : "queue";
}

/** Read the stored preference; anything else falls back to the default. */
export function readBusyEnter(raw: string | null): BusyEnterBehavior {
  return raw === "steer" ? "steer" : DEFAULT_BUSY_ENTER;
}
