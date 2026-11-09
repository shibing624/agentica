/** True while an input method is composing (pinyin, kana, …).
 *
 *  `isComposing` alone misses the Enter that commits a candidate: Chrome and
 *  Safari deliver that keydown with `isComposing === false` but
 *  `keyCode === 229`. Treating it as a real Enter sends half a sentence. */
export function isImeKey(e: { isComposing?: boolean; keyCode?: number }): boolean {
  return !!e.isComposing || e.keyCode === 229;
}