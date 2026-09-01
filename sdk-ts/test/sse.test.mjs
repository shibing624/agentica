import assert from "node:assert/strict";
import { test } from "node:test";
import { parseSSE } from "../dist/index.js";

test("parseSSE skips keepalives and stops at [DONE]", async () => {
  const body =
    ": keepalive\n\n" +
    "data: {\"seq\":1,\"event\":\"thinking\",\"data\":\"hmm\"}\n\n" +
    "data: [DONE]\n\n" +
    "data: {\"seq\":2,\"event\":\"content\",\"data\":\"late\"}\n\n";
  const events = [];
  for await (const event of parseSSE(new Response(body))) {
    events.push(event);
  }
  assert.deepEqual(events, [{ seq: 1, event: "thinking", data: "hmm" }]);
});
