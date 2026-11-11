/** Lines a folded user bubble shows. */
export const USER_FOLD_LINES = 9;
/** Slack factor so a bubble only 1-2 lines over the limit stays unfolded. */
export const USER_FOLD_SLACK = 1.3;

/**
 * Compute the max-height (in px) of a folded user bubble and whether its
 * rendered `scrollHeight` overflows enough to fold.
 *
 * Handles `line-height: normal` (where `parseFloat("normal")` is `NaN`) by
 * falling back to `1.5 * fontSize`.
 */
export function measureUserBubbleFold(
  scrollHeight: number,
  lineHeightCss: string,
  fontSizeCss = "14px",
  lines = USER_FOLD_LINES,
): { limit: number; overflows: boolean } {
  const parsedLine = parseFloat(lineHeightCss);
  const parsedFont = parseFloat(fontSizeCss);
  const linePx = Number.isFinite(parsedLine) && parsedLine > 0
    ? parsedLine
    : (Number.isFinite(parsedFont) && parsedFont > 0 ? parsedFont * 1.5 : 22.4);
  const limit = linePx * lines;
  return {
    limit,
    overflows: scrollHeight > limit * USER_FOLD_SLACK,
  };
}
