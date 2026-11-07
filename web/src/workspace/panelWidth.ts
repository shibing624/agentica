/**
 * Right files panel geometry, the same contract as deepseek-harness's right
 * column (ui-layout/columns.ts): first open takes 45% of the window, a drag
 * clamps to [300px, 70% of the window], and the chat column keeps at least
 * 400px — the panel gives way first, then closes its track entirely.
 */
export const PANEL_MIN = 300;
export const PANEL_MAX_RATIO = 0.7;
export const PANEL_DEFAULT_RATIO = 0.45;
export const CENTER_MIN = 400;

export function clampWidth(px: number, min: number, max: number): number {
  return Math.min(max, Math.max(min, Math.round(px)));
}

/** Width the user asked for: their last drag, or the first-open default. */
export function preferredWidth(saved: number | null, viewport: number): number {
  const want = saved ?? viewport * PANEL_DEFAULT_RATIO;
  return clampWidth(want, PANEL_MIN, Math.max(PANEL_MIN, viewport * PANEL_MAX_RATIO));
}

/** Width actually drawn this frame; 0 when the chat would drop below CENTER_MIN. */
export function resolvedWidth(preferred: number, viewport: number, sidebar: number): number {
  const available = viewport - sidebar - CENTER_MIN;
  if (available < PANEL_MIN) return 0;
  return Math.min(available, preferred);
}
