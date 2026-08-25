/** Keyboard routing while an approval card is on screen.

The composer is locked for the duration, so leftover text must not
steal Enter (that used to steer). Enter is allow-once; Esc is deny.
*/
export function approvalKeyAction(
  key: string,
  opts: { shiftKey?: boolean } = {},
): "allow" | "deny" | null {
  if (key === "Escape") return "deny";
  if (key === "Enter" && !opts.shiftKey) return "allow";
  return null;
}
