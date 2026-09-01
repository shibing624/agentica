/** HTTP failure from agentica-gateway. */
export class AgenticaError extends Error {
  readonly status: number;
  readonly body: unknown;

  constructor(status: number, message: string, body: unknown = null) {
    super(message);
    this.name = "AgenticaError";
    this.status = status;
    this.body = body;
  }

  static async fromResponse(response: Response): Promise<AgenticaError> {
    let body: unknown = null;
    const text = await response.text();
    if (text) {
      try {
        body = JSON.parse(text);
      } catch {
        body = text;
      }
    }
    const detail =
      body && typeof body === "object" && "detail" in body
        ? String((body as { detail: unknown }).detail)
        : text || response.statusText;
    return new AgenticaError(response.status, detail, body);
  }
}
