/** Pure parts of the reply footer's token-usage panel (TurnUsagePanel). */

const GAP = 8;
const MARGIN = 12;

type Rect = { left: number; top: number; bottom: number };
type Size = { width: number; height: number };

/** Above the trigger; below it when there is no room; clamped inside the
 *  viewport (12px margin) either way, so a footer near an edge cannot push
 *  the panel off screen. */
export function placeAbove(anchor: Rect, panel: Size, viewport: Size): { left: number; top: number } {
  let top = anchor.top - GAP - panel.height;
  if (top < MARGIN) top = Math.min(anchor.bottom + GAP, viewport.height - MARGIN - panel.height);
  const left = Math.min(Math.max(anchor.left, MARGIN), viewport.width - MARGIN - panel.width);
  return { left: Math.max(MARGIN, left), top: Math.max(MARGIN, top) };
}

export type TurnUsageInput = {
  tokIn?: number;
  tokOut?: number;
  cacheRead?: number;
  cacheWrite?: number;
  cacheHitPercent?: number | null;
  requests?: number;
  model?: string;
  llmMs?: number;
  costUsd?: number;
};

export type TurnUsageLabels = {
  turnUsageModel: string;
  cacheHitTip: string;
  turnUsageInput: string;
  turnUsageCacheRead: string;
  turnUsageCacheWrite: string;
  ctxOutput: string;
  turnUsageRequests: string;
  turnUsageSpeed: string;
  costUsdTip: string;
};

export type TurnUsageRow = { label: string; value: string; mono?: boolean };

const exact = (n: number) => Math.round(n).toLocaleString("en-US");

function fmtCost(cost: number) {
  if (cost > 0 && cost < 0.00005) return "<$0.0001";
  if (cost < 0.00995) return `$${cost.toFixed(4)}`;
  return `$${cost.toFixed(2)}`;
}

/**
 * Rows for one reply, top to bottom. ``tokIn`` is the whole prompt the model
 * saw (fresh + cache read + cache write), as the CLI footer counts it, so
 * "uncached input" is what is left after the two cache buckets. Rows whose
 * source the provider never reported are left out rather than shown as 0.
 */
export function turnUsageRows(m: TurnUsageInput, L: TurnUsageLabels): { total: string; rows: TurnUsageRow[] } {
  const prompt = m.tokIn || 0;
  const output = m.tokOut || 0;
  const cacheRead = m.cacheRead || 0;
  const cacheWrite = m.cacheWrite || 0;
  const rows: TurnUsageRow[] = [];
  if (m.model) rows.push({ label: L.turnUsageModel, value: m.model, mono: true });
  if (m.cacheHitPercent != null) rows.push({ label: L.cacheHitTip, value: `${m.cacheHitPercent.toFixed(1)}%` });
  rows.push({ label: L.turnUsageInput, value: exact(Math.max(prompt - cacheRead - cacheWrite, 0)) });
  if (m.cacheRead != null) rows.push({ label: L.turnUsageCacheRead, value: exact(cacheRead) });
  if (cacheWrite) rows.push({ label: L.turnUsageCacheWrite, value: exact(cacheWrite) });
  rows.push({ label: L.ctxOutput, value: exact(output) });
  if (m.requests) rows.push({ label: L.turnUsageRequests, value: String(m.requests) });
  if (m.llmMs && output) rows.push({ label: L.turnUsageSpeed, value: `${(output / (m.llmMs / 1000)).toFixed(1)} tok/s` });
  if ((m.costUsd || 0) > 0) rows.push({ label: L.costUsdTip, value: fmtCost(m.costUsd || 0) });
  return { total: exact(prompt + output), rows };
}
