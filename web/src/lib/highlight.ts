import hljs from "highlight.js/lib/core";
import bash from "highlight.js/lib/languages/bash";
import c from "highlight.js/lib/languages/c";
import cpp from "highlight.js/lib/languages/cpp";
import css from "highlight.js/lib/languages/css";
import diff from "highlight.js/lib/languages/diff";
import dockerfile from "highlight.js/lib/languages/dockerfile";
import go from "highlight.js/lib/languages/go";
import ini from "highlight.js/lib/languages/ini";
import java from "highlight.js/lib/languages/java";
import javascript from "highlight.js/lib/languages/javascript";
import json from "highlight.js/lib/languages/json";
import kotlin from "highlight.js/lib/languages/kotlin";
import latex from "highlight.js/lib/languages/latex";
import makefile from "highlight.js/lib/languages/makefile";
import markdown from "highlight.js/lib/languages/markdown";
import php from "highlight.js/lib/languages/php";
import python from "highlight.js/lib/languages/python";
import ruby from "highlight.js/lib/languages/ruby";
import rust from "highlight.js/lib/languages/rust";
import scss from "highlight.js/lib/languages/scss";
import sql from "highlight.js/lib/languages/sql";
import swift from "highlight.js/lib/languages/swift";
import typescript from "highlight.js/lib/languages/typescript";
import xml from "highlight.js/lib/languages/xml";
import yaml from "highlight.js/lib/languages/yaml";

hljs.registerLanguage("bash", bash);
hljs.registerLanguage("c", c);
hljs.registerLanguage("cpp", cpp);
hljs.registerLanguage("css", css);
hljs.registerLanguage("diff", diff);
hljs.registerLanguage("dockerfile", dockerfile);
hljs.registerLanguage("go", go);
hljs.registerLanguage("ini", ini);
hljs.registerLanguage("java", java);
hljs.registerLanguage("javascript", javascript);
hljs.registerLanguage("json", json);
hljs.registerLanguage("kotlin", kotlin);
hljs.registerLanguage("latex", latex);
hljs.registerLanguage("makefile", makefile);
hljs.registerLanguage("markdown", markdown);
hljs.registerLanguage("php", php);
hljs.registerLanguage("python", python);
hljs.registerLanguage("ruby", ruby);
hljs.registerLanguage("rust", rust);
hljs.registerLanguage("scss", scss);
hljs.registerLanguage("sql", sql);
hljs.registerLanguage("swift", swift);
hljs.registerLanguage("toml", ini);
hljs.registerLanguage("typescript", typescript);
hljs.registerLanguage("xml", xml);
hljs.registerLanguage("yaml", yaml);

const ALIAS: Record<string, string> = {
  md: "markdown",
  mdx: "markdown",
  py: "python",
  python3: "python",
  py3: "python",
  pyi: "python",
  pyw: "python",
  yml: "yaml",
  js: "javascript",
  mjs: "javascript",
  cjs: "javascript",
  jsx: "javascript",
  ts: "typescript",
  tsx: "typescript",
  mts: "typescript",
  cts: "typescript",
  jsonc: "json",
  jsonl: "json",
  ndjson: "json",
  ipynb: "json",
  htm: "xml",
  html: "xml",
  vue: "xml",
  svg: "xml",
  sh: "bash",
  zsh: "bash",
  fish: "bash",
  rs: "rust",
  h: "c",
  hpp: "cpp",
  cc: "cpp",
  cxx: "cpp",
  hh: "cpp",
  hxx: "cpp",
  rb: "ruby",
  kt: "kotlin",
  kts: "kotlin",
  conf: "ini",
  cfg: "ini",
  properties: "ini",
  env: "ini",
  gitignore: "ini",
  patch: "diff",
  tex: "latex",
  sty: "latex",
  cls: "latex",
  bib: "latex",
  bibtex: "latex",
  mk: "makefile",
};

const EXT_LANG: Record<string, string> = {
  md: "markdown",
  markdown: "markdown",
  mdx: "markdown",
  py: "python",
  pyi: "python",
  pyw: "python",
  yaml: "yaml",
  yml: "yaml",
  toml: "toml",
  java: "java",
  js: "javascript",
  mjs: "javascript",
  cjs: "javascript",
  jsx: "javascript",
  ts: "typescript",
  tsx: "typescript",
  mts: "typescript",
  cts: "typescript",
  json: "json",
  jsonc: "json",
  jsonl: "json",
  ndjson: "json",
  ipynb: "json",
  css: "css",
  scss: "scss",
  html: "xml",
  htm: "xml",
  xml: "xml",
  vue: "xml",
  svg: "xml",
  sh: "bash",
  bash: "bash",
  zsh: "bash",
  fish: "bash",
  rs: "rust",
  go: "go",
  c: "c",
  h: "c",
  cpp: "cpp",
  hpp: "cpp",
  cc: "cpp",
  cxx: "cpp",
  hh: "cpp",
  hxx: "cpp",
  sql: "sql",
  rb: "ruby",
  php: "php",
  ini: "ini",
  conf: "ini",
  cfg: "ini",
  properties: "ini",
  env: "ini",
  gitignore: "ini",
  kt: "kotlin",
  kts: "kotlin",
  swift: "swift",
  diff: "diff",
  patch: "diff",
  tex: "latex",
  latex: "latex",
  sty: "latex",
  cls: "latex",
  bib: "latex",
  mk: "makefile",
  makefile: "makefile",
};

// The workspace preview serves up to 256 KB; a core module (runner/loop.py is
// ~110 KB) highlights in ~30 ms, so the cap only keeps pathological blobs plain.
export const HIGHLIGHT_LIMIT = 256 * 1024;

function escapeHtml(text: string) {
  return text
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;");
}

export function resolveLanguage(name: string): string {
  const raw = (name || "").trim().toLowerCase();
  if (!raw) return "";
  if (raw === "dockerfile" || raw.startsWith("dockerfile.")) return "dockerfile";
  if (raw === "makefile" || raw === "gnumakefile") return "makefile";
  if (Object.hasOwn(ALIAS, raw)) return ALIAS[raw]!;
  if (hljs.getLanguage(raw)) return raw;
  return "";
}

export function languageFromFilename(name: string): string {
  const base = name.includes("/") ? name.slice(name.lastIndexOf("/") + 1) : name;
  const lower = base.toLowerCase();
  if (lower === "dockerfile" || lower.startsWith("dockerfile.")) return "dockerfile";
  if (lower === "makefile" || lower === "gnumakefile") return "makefile";
  const i = lower.lastIndexOf(".");
  if (i < 0) return "";
  const ext = lower.slice(i + 1);
  if (Object.hasOwn(EXT_LANG, ext)) return EXT_LANG[ext]!;
  return resolveLanguage(ext);
}

export function highlightToHtml(code: string, language?: string, highlight = true): string {
  if (!highlight) return escapeHtml(code);
  const lang = language ? resolveLanguage(language) : "";
  if (lang && hljs.getLanguage(lang)) {
    return hljs.highlight(code, { language: lang, ignoreIllegals: true }).value;
  }
  return escapeHtml(code);
}
