export type QuestionItem = {
  id: string;
  header: string;
  prompt: string;
  options: string[];
  multiSelect: boolean;
};

export type QuestionRequest = {
  questionId: string;
  batch: boolean;
  canEngage: boolean;
  engaged: boolean;
  items: QuestionItem[];
};

export type QuestionDraft = { picked: string[]; text: string; skipped: boolean };
export type QuestionAnswer = string | Record<string, string>;

export function parseQuestionRequest(data?: Record<string, unknown>): QuestionRequest | undefined {
  if (typeof data?.question_id !== "string" || !data.question_id) return;
  const parseItem = (item: Record<string, unknown>, id: string, prompt: unknown): QuestionItem => ({
    id,
    header: String(item.header ?? ""),
    prompt: String(prompt ?? ""),
    options: Array.isArray(item.options) ? item.options.map(String) : [],
    multiSelect: item.multi_select === true,
  });
  const batch = Array.isArray(data.questions);
  const items = Array.isArray(data.questions)
    ? data.questions.map((item) => parseItem(item, String(item.id), item.question))
    : [parseItem(data, data.question_id, data.prompt)];
  if (!items.length) return;
  return { questionId: data.question_id, batch, items, canEngage: data.can_engage === true, engaged: data.engaged === true };
}

export function draftAnswer(draft: QuestionDraft): string {
  if (draft.skipped) return "";
  return draft.text.trim() ? draft.text : draft.picked.join(", ");
}

export function questionSubmission(q: QuestionRequest, drafts: QuestionDraft[]): { answer: QuestionAnswer } | { missingIndex: number } {
  const answers = drafts.map(draftAnswer);
  const missingIndex = drafts.findIndex((draft, i) => !draft.skipped && !answers[i]);
  if (missingIndex !== -1) return { missingIndex };
  return { answer: q.batch ? Object.fromEntries(q.items.map((item, i) => [item.id, answers[i]])) : answers[0] };
}