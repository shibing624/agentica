import remarkGfm from "remark-gfm";
import type { PluggableList } from "unified";
import { defaultUrlTransform } from "react-markdown";
import * as api from "../api";
import { classifyPreviewTarget, parseLocalFileUrl } from "./filePath";

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

export type LocalFileLinkTarget = { path: string; previewable: boolean };

const LOCAL_FILE_MARKER = "1";

function localFileContentUrl(path: string): string {
  const slash = path.lastIndexOf("/");
  const root = slash < 0 ? "" : (slash === 0 ? "/" : path.slice(0, slash));
  const name = slash < 0 ? path : path.slice(slash + 1);
  const download = classifyPreviewTarget(path) === "unsupported";
  return `${api.workspaceContentUrl(root, name, download)}&local_file=${LOCAL_FILE_MARKER}`;
}

/** Keep `file:` / `computer:` links out of the DOM while the default sanitizer still handles every other scheme. */
export function markdownUrlTransform(url: string): string {
  const path = parseLocalFileUrl(url);
  return path === null ? defaultUrlTransform(url) : localFileContentUrl(path);
}

export function localFileLinkTarget(href: string): LocalFileLinkTarget | null {
  let url: URL;
  try { url = new URL(href, "http://agentica.local"); } catch { return null; }
  if (url.pathname !== "/api/workspace/content" || url.searchParams.get("local_file") !== LOCAL_FILE_MARKER) return null;
  const root = url.searchParams.get("root");
  const name = url.searchParams.get("path");
  if (root === null || name === null) return null;
  const prefix = root.endsWith("/") ? root : `${root}/`;
  const path = root === "" ? name : `${prefix}${name}`;
  return { path, previewable: classifyPreviewTarget(path) !== "unsupported" };
}

export function openLocalFile(
  target: LocalFileLinkTarget,
  onOpenFile?: (path: string) => void,
  openExternal: (path: string, app: string) => Promise<unknown> = api.openPathApi,
): void {
  if (target.previewable && onOpenFile) {
    onOpenFile(target.path);
    return;
  }
  void openExternal(target.path, "finder");
}
