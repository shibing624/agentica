import assert from "node:assert/strict";
import { test } from "node:test";
import { Agentica, AgenticaError } from "../dist/index.js";

test("sends Bearer token and client header", async () => {
  const calls = [];
  const fetchMock = async (url, init = {}) => {
    calls.push({ url: String(url), init });
    return new Response(JSON.stringify({ status: "ok", version: "1.4.14" }), {
      status: 200,
      headers: { "Content-Type": "application/json" },
    });
  };
  const client = new Agentica({
    baseURL: "http://127.0.0.1:8881",
    apiKey: "tok",
    fetch: fetchMock,
  });
  const health = await client.health();
  assert.equal(health.status, "ok");
  assert.equal(calls[0].url, "http://127.0.0.1:8881/api/health");
  assert.equal(calls[0].init.headers.Authorization, "Bearer tok");
  assert.equal(calls[0].init.headers["X-Agentica-Client"], "sdk");
});

test("401 becomes AgenticaError", async () => {
  const fetchMock = async () =>
    new Response(JSON.stringify({ detail: "Sign in at /login" }), { status: 401 });
  const client = new Agentica({ baseURL: "http://example.test", fetch: fetchMock });
  await assert.rejects(
    () => client.status(),
    (err) => {
      assert.ok(err instanceof AgenticaError);
      assert.equal(err.status, 401);
      assert.match(err.message, /Sign in/);
      return true;
    },
  );
});

test("chat.stream POSTs /api/chat/stream and yields SSE events", async () => {
  const frames =
    "data: {\"seq\":1,\"event\":\"content\",\"data\":\"Hi\"}\n\n" +
    "data: [DONE]\n\n";
  const fetchMock = async (url, init = {}) => {
    assert.equal(String(url), "http://127.0.0.1:8881/api/chat/stream");
    assert.equal(init.method, "POST");
    const body = JSON.parse(init.body);
    assert.equal(body.message, "hello");
    assert.equal(body.session_id, "s1");
    return new Response(frames, {
      status: 200,
      headers: { "Content-Type": "text/event-stream" },
    });
  };
  const client = new Agentica({ baseURL: "http://127.0.0.1:8881", fetch: fetchMock });
  const events = [];
  for await (const event of client.chat.stream({ message: "hello", session_id: "s1" })) {
    events.push(event);
  }
  assert.equal(events.length, 1);
  assert.equal(events[0].event, "content");
  assert.equal(events[0].data, "Hi");
});

test("sessions.list unwraps the envelope", async () => {
  const fetchMock = async () =>
    new Response(JSON.stringify({ sessions: [{ session_id: "a", name: "A" }] }), {
      status: 200,
    });
  const client = new Agentica({ baseURL: "http://127.0.0.1:8881", fetch: fetchMock });
  const sessions = await client.sessions.list();
  assert.equal(sessions[0].session_id, "a");
});
