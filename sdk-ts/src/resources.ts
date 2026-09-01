import { HttpClient } from "./http.js";
import { parseSSE } from "./sse.js";
import type { StreamEvent } from "./sse.js";
import type {
  ApprovalDecision,
  ChatRequest,
  ChatResponse,
  Run,
  Session,
} from "./types.js";

function chatBody(req: ChatRequest): Record<string, unknown> {
  return {
    message: req.message,
    session_id: req.session_id ?? "default",
    ...(req.work_dir ? { work_dir: req.work_dir } : {}),
    ...(req.approval_mode ? { approval_mode: req.approval_mode } : {}),
    ...(req.images?.length ? { images: req.images } : {}),
  };
}

export class Sessions {
  constructor(private readonly http: HttpClient) {}

  async list(): Promise<Session[]> {
    const body = await this.http.request<{ sessions: Session[] }>(
      "GET",
      "/api/sessions",
    );
    return body.sessions;
  }

  async delete(sessionId: string): Promise<void> {
    await this.http.request("DELETE", `/api/sessions/${encodeURIComponent(sessionId)}`);
  }

  async rename(sessionId: string, name: string): Promise<void> {
    await this.http.request("POST", `/api/sessions/${encodeURIComponent(sessionId)}/rename`, {
      body: { name },
    });
  }

  async archive(sessionId: string): Promise<void> {
    await this.http.request("POST", `/api/sessions/${encodeURIComponent(sessionId)}/archive`);
  }

  async unarchive(sessionId: string): Promise<void> {
    await this.http.request("POST", `/api/sessions/${encodeURIComponent(sessionId)}/unarchive`);
  }

  async compact(sessionId: string, instructions = ""): Promise<unknown> {
    return this.http.request(
      "POST",
      `/api/sessions/${encodeURIComponent(sessionId)}/compact`,
      { body: { instructions } },
    );
  }

  async usage(sessionId: string): Promise<unknown> {
    return this.http.request(
      "GET",
      `/api/sessions/${encodeURIComponent(sessionId)}/usage`,
    );
  }
}

export class Chat {
  constructor(private readonly http: HttpClient) {}

  /** One-shot, waits for the full reply. Prefer ``stream`` for a UI. */
  complete(req: ChatRequest): Promise<ChatResponse> {
    return this.http.request<ChatResponse>("POST", "/api/chat", {
      body: chatBody(req),
    });
  }

  /** Start a background run and subscribe in one request. */
  async *stream(
    req: ChatRequest,
    signal?: AbortSignal,
  ): AsyncGenerator<StreamEvent, void, undefined> {
    const response = await this.http.send("POST", "/api/chat/stream", {
      body: chatBody(req),
      signal,
    });
    yield* parseSSE(response);
  }

  /** Start a background run; disconnect does not cancel it. */
  createRun(req: ChatRequest): Promise<Run> {
    return this.http.request<Run>("POST", "/api/chat/runs", {
      body: chatBody(req),
    });
  }

  async active(sessionId: string): Promise<Run | null> {
    const body = await this.http.request<{ run: Run | null }>(
      "GET",
      "/api/chat/runs/active",
      { query: { session_id: sessionId } },
    );
    return body.run;
  }

  async *events(
    runId: string,
    after = 0,
    signal?: AbortSignal,
  ): AsyncGenerator<StreamEvent, void, undefined> {
    const response = await this.http.send(
      "GET",
      `/api/chat/runs/${encodeURIComponent(runId)}/events`,
      { query: { after }, signal },
    );
    yield* parseSSE(response);
  }

  cancel(runId: string): Promise<{ status: string; cancelled: boolean }> {
    return this.http.request(
      "POST",
      `/api/chat/runs/${encodeURIComponent(runId)}/cancel`,
      { body: {} },
    );
  }

  cancelSession(sessionId: string): Promise<{ status?: string; cancelled: boolean }> {
    return this.http.request("POST", "/api/chat/cancel", {
      body: { session_id: sessionId, message: "" },
    });
  }

  steer(
    sessionId: string,
    message: string,
  ): Promise<{ accepted: boolean }> {
    return this.http.request("POST", "/api/chat/steer", {
      body: { session_id: sessionId, message },
    });
  }

  approve(
    sessionId: string,
    toolCallId: string,
    decision: ApprovalDecision,
  ): Promise<unknown> {
    return this.http.request(
      "POST",
      `/api/sessions/${encodeURIComponent(sessionId)}/approvals/${encodeURIComponent(toolCallId)}`,
      { body: { decision } },
    );
  }
}
