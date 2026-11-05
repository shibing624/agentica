import assert from "node:assert/strict";
import test, { after } from "node:test";
import { fileURLToPath } from "node:url";
import { createServer } from "vite";

const vite = await createServer({
  root: fileURLToPath(new URL("../..", import.meta.url)),
  configFile: false,
  appType: "custom",
  logLevel: "silent",
  server: { middlewareMode: true },
});
const { plusMenuGroups, slashMenuGroups, flatItems } = await vite.ssrLoadModule("/src/lib/composerCommands.ts");
after(() => vite.close());

const S = {
  cmdGroupAdd: "添加", cmdGroupCommands: "指令", cmdGroupSkills: "技能",
  cmdFile: "文件", cmdGoal: "目标", cmdQueue: "排队", cmdCompact: "压缩",
  cmdPermission: "权限", cmdModel: "模型", cmdExport: "下载日志",
  slashGoal: "g", slashQueue: "q", slashCompact: "c",
  cmdPermissionDesc: "p", cmdModelDesc: "m", cmdExportDesc: "e",
};
const skills = [
  { name: "paper-analysis", slash: "/paper-analysis", description: "论文" },
  { name: "handoff", trigger: "/handoff", description: "Compact the conversation" },
  { name: "hidden", slash: "/hidden", is_hidden: true },
  { name: "auto-only", slash: "/auto-only", user_invocable: false },
];
const ids = (groups) => flatItems(groups).map((it) => it.id || it.cmd);

test("+ menu: add (file, goal), then commands (compact, permission, model, export); no plan or feedback", () => {
  const g = plusMenuGroups(S);
  assert.deepEqual(g.map((x) => x.title), ["添加", "指令"]);
  assert.deepEqual(ids(g), ["file", "goal", "compact", "permission", "model", "export"]);
});

test("bare / lists every command, then only invocable skills", () => {
  const g = slashMenuGroups("/", skills, S);
  assert.deepEqual(g.map((x) => x.title), ["指令", "技能"]);
  assert.deepEqual(ids(g), ["goal", "queue", "compact", "permission", "model", "export", "/paper-analysis", "/handoff"]);
});

test("/ filters commands and skills by prefix; empty groups disappear", () => {
  assert.deepEqual(ids(slashMenuGroups("/p", skills, S)), ["permission", "/paper-analysis"]);
  assert.deepEqual(slashMenuGroups("/h", skills, S).map((x) => x.title), ["技能"]);
  assert.deepEqual(slashMenuGroups("/zzz", skills, S), []);
});

test("a command also matches on its label, so typing in the UI language works", () => {
  assert.deepEqual(ids(slashMenuGroups("/压", skills, S)), ["compact"]);
});

test("the file picker is a + entry only, never a slash command", () => {
  assert.ok(!ids(slashMenuGroups("/f", skills, S)).includes("file"));
});
