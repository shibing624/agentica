# -*- coding: utf-8 -*-
"""Build normalized ReconcileMem replay cases from an Agentica workspace.

The script is intentionally extraction-only: it does not call an LLM and does
not fabricate gold labels. It prepares annotation/replay rows that contain
confirmed memories, quarantined candidates, and raw conversation blocks.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List

from agentica.workspace import Workspace
from evaluation.reconcile_mem.schemas import EvidenceItem, ReconcileCase


def parse_frontmatter(content: str) -> Dict[str, Any]:
    match = re.match(r"^---\s*\n(.*?)\n---\s*\n?", content, flags=re.DOTALL)
    if not match:
        return {}
    metadata: Dict[str, Any] = {}
    for line in match.group(1).splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        key = key.strip()
        value = value.strip()
        if key == "evidence_refs":
            try:
                parsed = json.loads(value)
            except json.JSONDecodeError:
                parsed = []
            metadata[key] = [str(item) for item in parsed] if isinstance(parsed, list) else []
        else:
            metadata[key] = value
    return metadata


def strip_frontmatter(content: str) -> str:
    return re.sub(r"^---\s*\n[\s\S]*?\n---\s*\n?", "", content, count=1).strip()


def read_memory_file(workspace: Workspace, path: Path, source: str) -> EvidenceItem:
    raw = path.read_text(encoding="utf-8").strip()
    metadata = parse_frontmatter(raw)
    return EvidenceItem(
        content=strip_frontmatter(raw),
        source=source,
        evidence_id=str(path.relative_to(workspace.path)),
        file_path=str(path.relative_to(workspace.path)),
        score=0.0,
    )


def collect_conversation_blocks(workspace: Workspace, query: str, limit: int) -> List[EvidenceItem]:
    results = workspace.search_conversations(query=query, limit=limit)
    items: List[EvidenceItem] = []
    for result in results:
        file_path = str(result.get("file_path", ""))
        items.append(EvidenceItem(
            content=str(result.get("content", "")),
            source="conversation",
            evidence_id=file_path,
            file_path=file_path,
            score=float(result.get("score", 0.0) or 0.0),
        ))
    return items


def build_cases(workspace_root: Path, user_id: str, conversation_limit: int) -> List[ReconcileCase]:
    workspace = Workspace(str(workspace_root), user_id=user_id)
    workspace.initialize()

    confirmed: List[EvidenceItem] = []
    memory_dir = workspace._get_user_memory_dir()
    if memory_dir.exists():
        for path in sorted(memory_dir.glob("*.md")):
            confirmed.append(read_memory_file(workspace, path, "memory"))

    cases: List[ReconcileCase] = []
    for candidate in workspace.list_memory_candidates():
        candidate_path = Path(candidate["path"])
        candidate_item = read_memory_file(workspace, candidate_path, "memory_candidate")
        query = str(candidate.get("name") or candidate_path.stem).replace("_", " ")
        conversations = collect_conversation_blocks(workspace, query=query, limit=conversation_limit)
        case = ReconcileCase(
            case_id=f"candidate::{candidate_path.stem}",
            query=query,
            confirmed_memories=confirmed,
            memory_candidates=[candidate_item],
            conversations=conversations,
            metadata={
                "workspace_root": str(workspace_root),
                "user_id": user_id,
                "needs_annotation": True,
                "suggested_task": "label promote/reject/unresolved and gold evidence refs",
            },
        )
        cases.append(case)

    return cases


def write_jsonl(path: Path, cases: List[ReconcileCase]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for case in cases:
            handle.write(json.dumps(case.to_dict(), ensure_ascii=False) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build ReconcileMem cases from Agentica workspace logs")
    parser.add_argument("--workspace-root", required=True, help="Agentica workspace root")
    parser.add_argument("--user-id", default="default")
    parser.add_argument("--out-jsonl", required=True)
    parser.add_argument("--conversation-limit", type=int, default=5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cases = build_cases(
        workspace_root=Path(args.workspace_root).expanduser().resolve(),
        user_id=args.user_id,
        conversation_limit=args.conversation_limit,
    )
    out_path = Path(args.out_jsonl).resolve()
    write_jsonl(out_path, cases)
    print(f"wrote {len(cases)} cases to {out_path}")


if __name__ == "__main__":
    main()
