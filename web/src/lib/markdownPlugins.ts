import remarkGfm from "remark-gfm";
import type { PluggableList } from "unified";

const HTML_BR_RE = /^(?:<br\s*\/?>\s*)+$/i;
const SINGLE_BR_RE = /<br\s*\/?>/gi;
const BLOCK_CONTAINERS = new Set(["root", "blockquote", "listItem"]);

type MdastNode = {
  type: string;
  value?: string;
  children?: MdastNode[];
};

/**
 * Turn literal `<br>` / `<br/>` / `<br />` HTML tokens emitted by LLMs
 * (especially inside GFM table cells and numbered points) into standard
 * Markdown `break` nodes without enabling `rehype-raw` on untrusted chat text.
 * Code spans/fences (`inlineCode` / `code`), `<` comparisons, and arbitrary
 * `<tag>` strings stay untouched and escaped.
 */
export function remarkHtmlBreak() {
  const walk = (parent: MdastNode) => {
    const children = parent.children;
    if (!Array.isArray(children)) return;
    for (let i = 0; i < children.length; i++) {
      const child = children[i];
      if (child.type === "html" && typeof child.value === "string" && HTML_BR_RE.test(child.value.trim())) {
        const count = child.value.match(SINGLE_BR_RE)?.length || 1;
        const breaks: MdastNode[] = Array.from({ length: count }, () => ({ type: "break" }));
        if (BLOCK_CONTAINERS.has(parent.type)) {
          children.splice(i, 1, { type: "paragraph", children: breaks });
        } else {
          children.splice(i, 1, ...breaks);
          i += breaks.length - 1;
        }
        continue;
      }
      walk(child);
    }
  };
  return (tree: MdastNode) => walk(tree);
}

/** Single `~` is a numeric range in Chinese prose; GFM strikethrough needs `~~`. */
export const GFM_PLUGINS: PluggableList = [
  [remarkGfm, { singleTilde: false }],
  remarkHtmlBreak,
];