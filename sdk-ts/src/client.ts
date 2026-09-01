import { Chat, Sessions } from "./resources.js";
import { HttpClient } from "./http.js";
import type { HealthResponse } from "./types.js";

export interface AgenticaOptions {
  /** Gateway origin, e.g. ``http://127.0.0.1:8881``. */
  baseURL?: string;
  /**
   * Machine token from the running gateway's ``runtime.json``.
   * Sent as ``Authorization: Bearer``. Leave unset when ``GATEWAY_AUTH=false``.
   */
  apiKey?: string;
  timeout?: number;
  fetch?: typeof fetch;
}

function env(name: string): string | undefined {
  if (typeof process === "undefined" || process.env == null) return undefined;
  const value = process.env[name];
  return value && value.length > 0 ? value : undefined;
}

/**
 * HTTP client for a running ``agentica-gateway``.
 *
 * This is not a TypeScript port of ``Agent``. Chat, tools and memory stay
 * in the Python process; this package only speaks the REST/SSE API.
 */
export class Agentica {
  readonly sessions: Sessions;
  readonly chat: Chat;
  private readonly http: HttpClient;

  constructor(options: AgenticaOptions = {}) {
    const baseURL =
      options.baseURL ?? env("AGENTICA_URL") ?? "http://127.0.0.1:8881";
    const apiKey = options.apiKey ?? env("AGENTICA_GATEWAY_TOKEN");
    this.http = new HttpClient({
      baseURL,
      apiKey,
      timeout: options.timeout,
      fetch: options.fetch,
    });
    this.sessions = new Sessions(this.http);
    this.chat = new Chat(this.http);
  }

  health(): Promise<HealthResponse> {
    return this.http.request<HealthResponse>("GET", "/api/health");
  }

  status(): Promise<unknown> {
    return this.http.request("GET", "/api/status");
  }
}
