# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: Durable file checkpoint / rollback primitive for coding agents.

Unlike the in-memory per-file ``undo_edit`` on ``BuiltinFileTool`` (single
process, lost on restart), ``CheckpointManager`` persists pre-edit file content
to disk under ``~/.agentica/cache/checkpoints/<session_id>/<checkpoint_id>/`` so a run
can be rolled back across multiple files and even across process restarts.

Design (kept deliberately small):
    cm = CheckpointManager(session_id="abc")
    ckpt = cm.create("before refactor", ["a.py", "b.py"])  # snapshots CURRENT content
    ... agent edits a.py / b.py ...
    cm.diff(ckpt.id)        # unified diff snapshot -> current
    cm.restore(ckpt.id)     # write snapshot content back (deletes files that
                            # did not exist when the checkpoint was taken)
    cm.list()               # newest-first list of checkpoints

This is an SDK primitive: it never auto-runs. Callers (CLI commands, user code,
a tool that wraps it) decide when to create/restore. It is intentionally NOT
wired into ``BuiltinFileTool``'s edit path: per-edit auto-snapshots produced one
single-file checkpoint dir per edit (no retention cap, and defeating the whole
multi-file point). In-process undo there is handled by ``undo_edit``; reach for
``CheckpointManager`` when you want explicit, durable, multi-file rollback.
"""
import difflib
import json
import os
import shutil
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from agentica.config import AGENTICA_CACHE_DIR
from agentica.utils.log import logger

DEFAULT_CHECKPOINT_ROOT = os.path.join(AGENTICA_CACHE_DIR, "checkpoints")


@dataclass
class CheckpointFile:
    """One file captured in a checkpoint."""
    path: str           # absolute path on disk (the restore target)
    existed: bool       # whether the file existed when the checkpoint was taken
    stored_name: Optional[str] = None  # filename holding the old content (None if !existed)


@dataclass
class Checkpoint:
    """A point-in-time snapshot of one or more files."""
    id: str
    label: str
    created_at: str
    files: List[CheckpointFile] = field(default_factory=list)
    dir: Optional[Path] = None

    @property
    def paths(self) -> List[str]:
        return [f.path for f in self.files]


class CheckpointManager:
    """Disk-backed, multi-file, cross-process checkpoint store."""

    def __init__(self, session_id: Optional[str] = None, root_dir: Optional[str] = None):
        self.session_id = session_id or "default"
        self.root = Path(root_dir or DEFAULT_CHECKPOINT_ROOT) / self.session_id

    # ── helpers ──────────────────────────────────────────────────────────
    def _checkpoint_dir(self, checkpoint_id: str) -> Path:
        return self.root / checkpoint_id

    @staticmethod
    def _new_id() -> str:
        # Sortable: microsecond timestamp prefix + short random suffix, so
        # checkpoints created within the same second still order by creation.
        return datetime.now().strftime("%Y%m%d%H%M%S%f") + "_" + uuid.uuid4().hex[:8]

    def _manifest_path(self, checkpoint_id: str) -> Path:
        return self._checkpoint_dir(checkpoint_id) / "manifest.json"

    # ── create ───────────────────────────────────────────────────────────
    def create(self, label: str, paths: List) -> Checkpoint:
        """Snapshot the CURRENT content of ``paths`` into a new checkpoint.

        Non-existent paths are recorded as ``existed=False`` so a later
        restore can delete them (undoing a file creation).
        """
        checkpoint_id = self._new_id()
        ckpt_dir = self._checkpoint_dir(checkpoint_id)
        files_dir = ckpt_dir / "files"
        files_dir.mkdir(parents=True, exist_ok=True)

        entries: List[CheckpointFile] = []
        for idx, raw in enumerate(paths):
            abs_path = str(Path(raw).expanduser().resolve())
            src = Path(abs_path)
            if src.exists() and src.is_file():
                stored_name = f"{idx}.snap"
                shutil.copyfile(src, files_dir / stored_name)
                entries.append(CheckpointFile(path=abs_path, existed=True, stored_name=stored_name))
            else:
                entries.append(CheckpointFile(path=abs_path, existed=False, stored_name=None))

        ckpt = Checkpoint(
            id=checkpoint_id,
            label=label,
            created_at=datetime.now().isoformat(timespec="seconds"),
            files=entries,
            dir=ckpt_dir,
        )
        self._write_manifest(ckpt)
        logger.debug(f"Created checkpoint {checkpoint_id} ({label}) with {len(entries)} file(s)")
        return ckpt

    def _write_manifest(self, ckpt: Checkpoint) -> None:
        manifest = {
            "id": ckpt.id,
            "label": ckpt.label,
            "created_at": ckpt.created_at,
            "files": [
                {"path": f.path, "existed": f.existed, "stored_name": f.stored_name}
                for f in ckpt.files
            ],
        }
        with open(self._manifest_path(ckpt.id), "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2, ensure_ascii=False)

    # ── read ─────────────────────────────────────────────────────────────
    def get(self, checkpoint_id: str) -> Optional[Checkpoint]:
        manifest_path = self._manifest_path(checkpoint_id)
        if not manifest_path.exists():
            return None
        try:
            with open(manifest_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError):
            return None
        return Checkpoint(
            id=data["id"],
            label=data.get("label", ""),
            created_at=data.get("created_at", ""),
            files=[
                CheckpointFile(path=f["path"], existed=f["existed"], stored_name=f.get("stored_name"))
                for f in data.get("files", [])
            ],
            dir=self._checkpoint_dir(checkpoint_id),
        )

    def list(self) -> List[Checkpoint]:
        """Return checkpoints, newest first."""
        if not self.root.exists():
            return []
        ids = sorted(
            (p.name for p in self.root.iterdir() if p.is_dir() and self._manifest_path(p.name).exists()),
            reverse=True,
        )
        result = []
        for cid in ids:
            ckpt = self.get(cid)
            if ckpt is not None:
                result.append(ckpt)
        return result

    def latest(self) -> Optional[Checkpoint]:
        items = self.list()
        return items[0] if items else None

    # ── diff ─────────────────────────────────────────────────────────────
    def diff(self, checkpoint_id: str) -> str:
        """Unified diff from the checkpointed content to the CURRENT on-disk content."""
        ckpt = self.get(checkpoint_id)
        if ckpt is None:
            return f"Checkpoint not found: {checkpoint_id}"

        chunks: List[str] = []
        for f in ckpt.files:
            old_text = ""
            if f.existed and f.stored_name:
                stored = ckpt.dir / "files" / f.stored_name
                old_text = stored.read_text(encoding="utf-8", errors="ignore") if stored.exists() else ""
            cur_path = Path(f.path)
            new_text = cur_path.read_text(encoding="utf-8", errors="ignore") if cur_path.exists() else ""
            if old_text == new_text:
                continue
            old_label = f"a/{f.path}" if f.existed else "/dev/null"
            new_label = f"b/{f.path}" if cur_path.exists() else "/dev/null"
            diff = difflib.unified_diff(
                old_text.splitlines(keepends=True),
                new_text.splitlines(keepends=True),
                fromfile=old_label,
                tofile=new_label,
            )
            chunks.append("".join(diff))
        return "\n".join(c for c in chunks if c.strip()) or "(no changes since checkpoint)"

    # ── restore ──────────────────────────────────────────────────────────
    def restore(self, checkpoint_id: str) -> List[str]:
        """Restore files to their checkpointed state. Returns affected paths.

        - Files that existed at snapshot time are rewritten to old content.
        - Files that did NOT exist at snapshot time are deleted (undo creation).
        """
        ckpt = self.get(checkpoint_id)
        if ckpt is None:
            raise ValueError(f"Checkpoint not found: {checkpoint_id}")

        restored: List[str] = []
        for f in ckpt.files:
            target = Path(f.path)
            if f.existed and f.stored_name:
                stored = ckpt.dir / "files" / f.stored_name
                if stored.exists():
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(stored, target)
                    restored.append(f.path)
            else:
                # Did not exist at snapshot time -> remove if it exists now.
                if target.exists():
                    target.unlink()
                    restored.append(f.path)
        logger.debug(f"Restored checkpoint {checkpoint_id}: {len(restored)} file(s)")
        return restored

    # ── cleanup ──────────────────────────────────────────────────────────
    def delete(self, checkpoint_id: str) -> bool:
        ckpt_dir = self._checkpoint_dir(checkpoint_id)
        if ckpt_dir.exists():
            shutil.rmtree(ckpt_dir, ignore_errors=True)
            return True
        return False

    def clear(self) -> int:
        """Delete all checkpoints for this session. Returns count removed."""
        items = self.list()
        for ckpt in items:
            self.delete(ckpt.id)
        return len(items)
