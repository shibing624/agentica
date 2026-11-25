/** Search rendered text in document order, without changing React-owned DOM. */
export function pageMatches(root: HTMLElement, query: string): Range[] {
  const words = query.trim().split(/\s+/);
  if (!words[0]) return [];
  const pattern = new RegExp(words.map((word) => word.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")).join("\\s+"), "giu");
  const matches: Range[] = [];
  let nodes: Text[] = [];

  const flush = () => {
    if (!nodes.length) return;
    let offset = 0;
    const parts = nodes.map((node) => ({ node, start: offset, end: offset += node.length }));
    const text = nodes.map((node) => node.data).join("");
    nodes = [];
    let start = 0, end = 0;
    for (const match of text.matchAll(pattern)) {
      const first = match.index, last = first + match[0].length;
      while (parts[start].end <= first) start++;
      while (parts[end].end < last) end++;
      const range = root.ownerDocument.createRange();
      range.setStart(parts[start].node, first - parts[start].start);
      range.setEnd(parts[end].node, last - parts[end].start);
      // Whitespace and browser-collapsed content must not create phantom hits.
      if (range.getClientRects().length) matches.push(range);
    }
  };

  const visit = (node: Node) => {
    if (node instanceof Text) {
      if (node.length) nodes.push(node);
      return;
    }
    if (!(node instanceof HTMLElement) || node.hasAttribute("data-page-find")) return;
    const style = getComputedStyle(node);
    if (style.display === "none" || style.visibility !== "visible" || style.contentVisibility === "hidden" || style.opacity === "0") return;
    const boundary = node.tagName === "BR" || (style.display !== "inline" && style.display !== "contents");
    if (boundary) flush();
    if (node instanceof HTMLDetailsElement && !node.open) {
      const summary = node.querySelector(":scope > summary");
      if (summary) visit(summary);
    } else {
      for (const child of node.childNodes) visit(child);
    }
    if (boundary) flush();
  };

  visit(root);
  flush();
  return matches;
}

/** A hit can be deep inside a tall reply or scrollable code block. Scroll to
 *  the range, not its parent element (which might surround the whole file). */
export function revealPageMatch(range: Range) {
  const scrollable = /^(auto|scroll|overlay)$/;
  const delta = (start: number, end: number, lo: number, hi: number) =>
    start < lo || end > hi ? (start + end - lo - hi) / 2 : 0;
  for (let el = range.startContainer.parentElement; el; el = el.parentElement) {
    const { overflowX, overflowY } = getComputedStyle(el);
    const x = scrollable.test(overflowX) && el.scrollWidth > el.clientWidth;
    const y = scrollable.test(overflowY) && el.scrollHeight > el.clientHeight;
    if (!x && !y) continue;
    const rect = range.getBoundingClientRect(), box = el.getBoundingClientRect();
    const left = box.left + el.clientLeft, top = box.top + el.clientTop;
    el.scrollBy({
      left: x ? delta(rect.left, rect.right, left, left + el.clientWidth) : 0,
      top: y ? delta(rect.top, rect.bottom, top, top + el.clientHeight) : 0,
      behavior: "instant",
    });
  }
  const rect = range.getBoundingClientRect();
  window.scrollBy({ left: delta(rect.left, rect.right, 0, innerWidth), top: delta(rect.top, rect.bottom, 0, innerHeight), behavior: "instant" });
}