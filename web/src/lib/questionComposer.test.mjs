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
after(() => vite.close());

test("question requests preserve batch identity, order, headings and selection mode", async () => {
  const { parseQuestionRequest } = await vite.ssrLoadModule("/src/lib/questionComposer.ts");
  const q = parseQuestionRequest({
    question_id: "q-123", can_engage: true, engaged: true,
    questions: [
      { id: "scope", header: "范围", question: "选哪些？", options: ["Web", "Desktop"], multi_select: true },
      { id: "notes", question: "写一点备注。\n可以分行。" },
    ],
  });
  assert.deepEqual(q, {
    questionId: "q-123", batch: true, canEngage: true, engaged: true,
    items: [
      { id: "scope", header: "范围", prompt: "选哪些？", options: ["Web", "Desktop"], multiSelect: true },
      { id: "notes", header: "", prompt: "写一点备注。\n可以分行。", options: [], multiSelect: false },
    ],
  });
});

test("submission distinguishes a missing draft from a deliberate skip", async () => {
  const { parseQuestionRequest, questionSubmission } = await vite.ssrLoadModule("/src/lib/questionComposer.ts");
  const q = parseQuestionRequest({ question_id: "q-123", questions: [
    { id: "scope", question: "范围？" }, { id: "notes", question: "备注？" }, { id: "extra", question: "其他？" },
  ] });
  const drafts = [
    { picked: ["Desktop", "Web"], text: "", skipped: false },
    { picked: [], text: "第一行\n第二行", skipped: false },
    { picked: [], text: "  ", skipped: false },
  ];
  assert.deepEqual(questionSubmission(q, drafts), { missingIndex: 2 });
  drafts[2].skipped = true;
  assert.deepEqual(questionSubmission(q, drafts), {
    answer: { scope: "Desktop, Web", notes: "第一行\n第二行", extra: "" },
  });
});

test("legacy question requests retain their string reply protocol", async () => {
  const { parseQuestionRequest, questionSubmission } = await vite.ssrLoadModule("/src/lib/questionComposer.ts");
  const q = parseQuestionRequest({ question_id: "q-123", prompt: "选哪个？", options: ["A", "B"] });
  assert.equal(q.batch, false);
  assert.equal(q.canEngage, false);
  assert.deepEqual(questionSubmission(q, [{ picked: ["A"], text: "", skipped: false }]), { answer: "A" });
});