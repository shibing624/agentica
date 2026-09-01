/** One gateway SSE frame. ``seq`` is the replay cursor for ``?after=``. */
export interface StreamEvent {
  seq?: number;
  event: string;
  data: unknown;
}

/**
 * Parse ``text/event-stream`` from a fetch Response.
 * Stops on ``data: [DONE]``. Comment lines (keepalive) are skipped.
 */
export async function* parseSSE(
  response: Response,
): AsyncGenerator<StreamEvent, void, undefined> {
  if (!response.body) {
    throw new Error("Response body is null");
  }
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  try {
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split("\n");
      buffer = lines.pop() ?? "";
      for (const line of lines) {
        const event = parseDataLine(line);
        if (event === "done") return;
        if (event) yield event;
      }
    }
    const tail = parseDataLine(buffer);
    if (tail && tail !== "done") yield tail;
  } finally {
    reader.releaseLock();
  }
}

function parseDataLine(line: string): StreamEvent | "done" | null {
  if (!line.startsWith("data: ")) return null;
  const raw = line.slice(6);
  if (raw.trim() === "[DONE]") return "done";
  try {
    return JSON.parse(raw) as StreamEvent;
  } catch {
    return null;
  }
}
