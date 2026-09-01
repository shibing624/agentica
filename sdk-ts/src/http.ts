import { AgenticaError } from "./errors.js";

export const CLIENT_HEADER = "X-Agentica-Client";

export type QueryParams = Record<
  string,
  string | number | boolean | undefined
>;

export interface HttpClientOptions {
  baseURL: string;
  apiKey?: string;
  timeout?: number;
  fetch?: typeof fetch;
}

export interface RequestOptions {
  query?: QueryParams;
  body?: unknown;
  headers?: Record<string, string>;
  timeout?: number;
  signal?: AbortSignal;
}

const DEFAULT_TIMEOUT_MS = 60_000;

export class HttpClient {
  readonly baseURL: string;
  readonly apiKey?: string;
  readonly timeout: number;
  readonly fetchImpl: typeof fetch;

  constructor(options: HttpClientOptions) {
    this.baseURL = options.baseURL.replace(/\/$/, "");
    this.apiKey = options.apiKey;
    this.timeout = options.timeout ?? DEFAULT_TIMEOUT_MS;
    this.fetchImpl = options.fetch ?? globalThis.fetch.bind(globalThis);
  }

  async request<T>(
    method: string,
    path: string,
    options: RequestOptions = {},
  ): Promise<T> {
    const response = await this.send(method, path, options);
    if (response.status === 204) {
      return undefined as T;
    }
    const text = await response.text();
    if (!text) {
      return undefined as T;
    }
    return JSON.parse(text) as T;
  }

  async send(
    method: string,
    path: string,
    options: RequestOptions = {},
  ): Promise<Response> {
    const url = this.buildURL(path, options.query);
    const headers: Record<string, string> = {
      [CLIENT_HEADER]: "sdk",
      ...options.headers,
    };
    if (this.apiKey) {
      headers.Authorization = `Bearer ${this.apiKey}`;
    }
    let body: string | undefined;
    if (options.body !== undefined) {
      headers["Content-Type"] = "application/json";
      body = JSON.stringify(options.body);
    }

    const timeout = options.timeout ?? this.timeout;
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), timeout);
    const onOuterAbort = () => controller.abort();
    options.signal?.addEventListener("abort", onOuterAbort);

    try {
      const response = await this.fetchImpl(url, {
        method,
        headers,
        body,
        signal: controller.signal,
      });
      if (!response.ok) {
        throw await AgenticaError.fromResponse(response);
      }
      return response;
    } catch (err) {
      if (err instanceof AgenticaError) {
        throw err;
      }
      if (controller.signal.aborted) {
        throw new AgenticaError(0, `Request timed out after ${timeout}ms`);
      }
      throw err;
    } finally {
      clearTimeout(timer);
      options.signal?.removeEventListener("abort", onOuterAbort);
    }
  }

  buildURL(path: string, query?: QueryParams): string {
    const url = new URL(
      path.startsWith("/") ? path : `/${path}`,
      `${this.baseURL}/`,
    );
    if (query) {
      for (const [key, value] of Object.entries(query)) {
        if (value === undefined) continue;
        url.searchParams.set(key, String(value));
      }
    }
    return url.toString();
  }
}
