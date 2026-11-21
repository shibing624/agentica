/**
 * Right files panel geometry, the same contract as deepseek-harness's right
 * column (ui-layout/columns.ts): first open takes 45% of the window, a drag
 * clamps to [300px, 70% of the window], and the chat column keeps at least
 * 400px when docked. Below that, float the panel rather than hiding an open one.
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

export function resolvePanelLayout(preferred: number, viewport: number, sidebar: number) {
  const available = viewport - sidebar - CENTER_MIN;
  if (available < PANEL_MIN) return { width: Math.min(viewport, preferred), floating: true };
  return { width: Math.min(available, preferred), floating: false };
}

export const TREE_DEFAULT_WIDTH = 248;

export function resolveTreeLayout(available: number, preferred: number) {
  if (available < 400) return { mode: "fill", width: available, min: available, max: available } as const;
  const mode = available >= 560 ? "beside" : "overlay";
  const min = 180;
  const max = available - (mode === "beside" ? 336 : 136);
  return { mode, width: clampWidth(preferred, min, max), min, max };
}
