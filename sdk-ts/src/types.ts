export interface ChatImage {
  mime?: string;
  data: string;
}

export interface ChatRequest {
  message: string;
  session_id?: string;
  work_dir?: string;
  approval_mode?: string;
  images?: ChatImage[];
}

export interface ChatResponse {
  content: string;
  session_id: string;
  user_id: string;
  tool_calls: number;
}

export interface Run {
  run_id: string;
  session_id: string;
  status: string;
  kind: string;
  seq: number;
}

export type ApprovalDecision =
  | "allow"
  | "allow_prefix"
  | "deny"
  | "deny_prefix";

export interface Session {
  session_id: string;
  name: string;
  preview?: string;
  archived?: boolean;
  work_dir?: string;
  running?: boolean;
  unread?: boolean;
  first_timestamp?: string | null;
  last_timestamp?: string | null;
}

export interface HealthResponse {
  status: string;
  version?: string;
}
