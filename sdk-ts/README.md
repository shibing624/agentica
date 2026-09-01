# @agentica-ai/sdk

TypeScript HTTP client for a running [`agentica-gateway`](https://github.com/shibing624/agentica). It is **not** a port of the Python `Agent` runtime: chat, tools and memory stay in that process. This package sends the same REST/SSE calls the Web UI uses.

Starting the Web UI is unchanged: `pip install "agentica[gateway]"` then `agentica-gateway`. You only need this package if you are writing a Node/TypeScript program that talks to that gateway.

## Install

```bash
npm install @agentica-ai/sdk
```

Always the scoped name **`@agentica-ai/sdk`**. The CLI talks to `https://registry.npmjs.org/` (not `www.npmjs.com`). Requires Node 18+ (native `fetch`).

## Usage

```ts
import { Agentica } from "@agentica-ai/sdk";

const agentica = new Agentica({
  baseURL: "http://127.0.0.1:8881",
  // Machine token from ~/.agentica/cache/gateway/runtime.json
  // Omit when GATEWAY_AUTH=false.
  apiKey: process.env.AGENTICA_GATEWAY_TOKEN,
});

await agentica.health();

const sessions = await agentica.sessions.list();

for await (const event of agentica.chat.stream({
  session_id: "demo",
  message: "What files are in this repo?",
})) {
  if (event.event === "content") process.stdout.write(String(event.data));
}
```

`baseURL` falls back to `AGENTICA_URL` then `http://127.0.0.1:8881`. `apiKey` falls back to `AGENTICA_GATEWAY_TOKEN`.

Background runs (refresh-safe, same as the Web UI):

```ts
const run = await agentica.chat.createRun({ session_id: "demo", message: "…" });
for await (const event of agentica.chat.events(run.run_id, run.seq)) {
  // reconnect with the last seq after a drop
}
await agentica.chat.approve("demo", toolCallId, "allow");
```

## Auth

Scripts present the **machine token** (`Authorization: Bearer`), not the browser session cookie. Read it from the running gateway's `runtime.json`, or pin it with `AGENTICA_GATEWAY_TOKEN` on both the gateway and the client.

## Develop

```bash
cd sdk-ts
npm install
npm test
```

## Publish

This repo publishes **one** npm package: `@agentica-ai/sdk` under the [Agentica AI](https://www.npmjs.com/~agentica-ai) organization.
