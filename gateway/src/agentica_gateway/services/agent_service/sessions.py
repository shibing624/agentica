# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: Session listing, SessionLog locating, fork/feedback/archive/pin, and background jobs.
"""
from __future__ import annotations

import asyncio
import os
import queue
import shutil
import sys
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

from agentica.memory.session_log import SessionLog, is_session_unread, iso_timestamp
from agentica.run.context import RunSource
from agentica.tools.background_processes import (
    BackgroundProcess,
    BackgroundProcessRegistry,
    background_job_view_from_process,
    background_report_from_record,
    claim_job_delivery,
    count_running_session_jobs,
    delivery_owner,
    load_session_job_views,
    mark_job_delivered,
    note_job_delivery_failure,
    read_log_tail,
    release_job_claim,
    stop_all_session_jobs,
    stop_session_job,
    undelivered_job_records,
)
from agentica.utils.log import logger

from ...config import settings
from .. import chat_images
from .types import (
    CRON_SESSION_PREFIX,
    IM_SESSION_SOURCES,
    _TITLE_PROMPT,
    _infer_im_source_from_session_id,
    _log_report_failure,
)


def _pkg():
    return sys.modules["agentica_gateway.services.agent_service"]


class AgentSessionsMixin:
    """Session discovery, fork/feedback/rename/archive/delete, and background processes."""

    def _session_preview(self, entry: Dict[str, Any]) -> Dict[str, Any]:
        path = str(entry["path"])
        version = (
            int(entry.get("size_bytes") or 0),
            int(entry.get("mtime_ns") or 0),
        )
        with self._session_preview_lock:
            cached = self._session_preview_cache.get(path)
            if cached is not None and cached[0] == version:
                self._session_preview_cache.move_to_end(path)
                return cached[1]
        preview = SessionLog.session_preview(path)
        with self._session_preview_lock:
            self._session_preview_cache[path] = (version, preview)
            self._session_preview_cache.move_to_end(path)
            while len(self._session_preview_cache) > 10000:
                self._session_preview_cache.popitem(last=False)
        return preview

    def list_sessions(
        self,
        owner: Optional[str] = None,
        scope: str = "recent",
        recent_days: int = 30,
        include_cli: bool = False,
        hide_temp_dirs: bool = False,
    ) -> List[Dict[str, Any]]:
        """List lightweight session summaries across the owner's projects."""
        if scope not in ("all", "recent"):
            raise ValueError(f"Unknown session scope: {scope}")
        if recent_days < 1:
            raise ValueError("recent_days must be positive")
        pkg = _pkg()
        uid = self._owner(owner)
        projects = SessionLog.list_projects(user_id=uid)
        if not projects:
            projects = [{
                "base_dir": self._session_base_dir(str(settings.base_dir), owner),
                "work_dir": str(settings.base_dir),
            }]
        candidates: Dict[str, tuple[Dict[str, Any], str]] = {}
        for project in projects:
            base_dir = project["base_dir"]
            work_dir = project.get("work_dir") or ""
            for entry in SessionLog.list_sessions(base_dir=base_dir, work_dir=work_dir or None):
                sid = entry["session_id"]
                if sid.startswith(CRON_SESSION_PREFIX):
                    continue
                previous = candidates.get(sid)
                if previous is None or int(entry.get("size_bytes") or 0) > int(previous[0].get("size_bytes") or 0):
                    candidates[sid] = (entry, work_dir)

        cutoff = time.time() - recent_days * 86400
        gateway_dir = os.path.realpath(str(settings.base_dir))

        def _hidden_temp(work_dir: str) -> bool:
            if not pkg.is_system_temp_path(work_dir):
                return False
            real = os.path.realpath(work_dir)
            return not (real == gateway_dir or real.startswith(gateway_dir + os.sep))

        out: List[Dict[str, Any]] = []
        for entry, project_work_dir in candidates.values():
            sid = entry["session_id"]
            running = (
                self.is_session_active(sid, owner=uid)
                or pkg.live_turn.active(sid, owner=uid) is not None
            )
            source = entry.get("source")
            im_from_sid = _infer_im_source_from_session_id(sid)
            if not source:
                if im_from_sid:
                    source = im_from_sid
                elif entry.get("cli"):
                    source = "cli"
                elif entry.get("last_read_at"):
                    source = "gateway"
                else:
                    source = "legacy"
            bg_dir = Path(entry["path"]).parent / sid / "background"
            bg_running = count_running_session_jobs(
                bg_dir, self._session_background.get(self._sk(sid, uid)),
            )
            has_live_work = running or bg_running > 0
            origin = im_from_sid or entry.get("origin") or source
            is_gateway_or_im = any(
                s in ("gateway", "cron") or s in IM_SESSION_SOURCES for s in (source, origin)
            )
            archived = bool(entry.get("archived"))
            pinned = bool(entry.get("pinned"))
            if not has_live_work and not (is_gateway_or_im or (include_cli and source == "cli")):
                continue
            if (
                not has_live_work
                and hide_temp_dirs
                and _hidden_temp(entry.get("work_dir") or project_work_dir)
                and not self._session_preview(entry).get("user_count")
                and not self._session_preview(entry).get("cron_job_name")
            ):
                continue
            if (
                scope == "recent"
                and not is_gateway_or_im
                and not has_live_work
                and not archived
                and not pinned
                and float(entry.get("mtime") or 0) < cutoff
            ):
                continue
            preview = self._session_preview(entry)
            first_user = chat_images.strip_markers(preview.get("first_user", "") or "")
            cron_title = str(preview.get("cron_job_name") or "").strip()
            first_ts = entry.get("first_timestamp")
            if not first_ts and entry.get("mtime"):
                first_ts = iso_timestamp(entry["mtime"])
            last_ts = entry.get("last_timestamp")
            last_read_at = entry.get("last_read_at")
            out.append({
                "session_id": sid,
                "name": (
                    entry.get("name")
                    or (first_user[:40] if first_user else "")
                    or (cron_title[:60] if cron_title else "Chat")
                ),
                "preview": first_user or cron_title,
                "user_count": preview.get("user_count", 0),
                "first_timestamp": first_ts,
                "last_timestamp": last_ts,
                "last_read_at": last_read_at,
                "unread": (not running) and is_session_unread(last_ts, last_read_at),
                "size_bytes": entry.get("size_bytes", 0),
                "archived": archived,
                "pinned": pinned,
                "work_dir": entry.get("work_dir") or project_work_dir,
                "running": running,
                "background_jobs": bg_running,
                "source": source,
                "origin": origin,
            })
        seen = {row["session_id"] for row in out}
        for turn in pkg.live_turn.iter_owner(uid):
            if turn.session_id.startswith(CRON_SESSION_PREFIX) or turn.session_id in seen:
                continue
            live_log = self.session_log_for(turn.session_id, owner=uid)
            live_source = (
                live_log.get_source()
                or _infer_im_source_from_session_id(turn.session_id)
                or "gateway"
            )
            live_title = (
                live_log.get_name()
                or ((turn.cron_job or {}).get("name") or "").strip()[:60]
                or (chat_images.strip_markers(turn.user_message).strip()[:40] if turn.user_message else "")
                or "Chat"
            )
            out.append({
                "session_id": turn.session_id,
                "name": live_title,
                "preview": "",
                "user_count": 0,
                "first_timestamp": iso_timestamp(),
                "last_timestamp": None,
                "last_read_at": None,
                "unread": False,
                "size_bytes": 0,
                "archived": False,
                "work_dir": self.get_session_work_dir(turn.session_id, owner=uid),
                "running": not turn.done,
                "source": live_source,
            })
        out.sort(key=lambda row: str(row.get("first_timestamp") or ""), reverse=True)
        return out

    def session_log_for(self, session_id: str, owner: Optional[str] = None) -> SessionLog:
        """Open the on-disk SessionLog for a session (may not exist yet)."""
        base_dir, work_dir = self._locate_session(session_id, owner)
        return SessionLog(session_id=session_id, base_dir=base_dir, work_dir=work_dir or None)

    @staticmethod
    def _jsonl_has_content(path: Path) -> bool:
        """True when ``path`` is a real transcript, not a 0-byte touch stub."""
        try:
            return path.is_file() and path.stat().st_size > 0
        except OSError:
            return False

    def _locate_session(self, session_id: str, owner: Optional[str] = None) -> tuple[str, str]:
        """Find the project directory that actually holds this session's jsonl."""
        remembered = self._session_work_dirs.get(self._sk(session_id, owner))
        if remembered:
            return self._session_base_dir(remembered, owner), remembered

        fallback_work = str(settings.base_dir)
        fallback = self._session_base_dir(fallback_work, owner)
        hits = [
            h for h in SessionLog.find_sessions(session_id, user_id=self._owner(owner))
            if h.get("session_id") == session_id
        ]
        contentful = [h for h in hits if (h.get("size_bytes") or 0) > 0]
        chosen = contentful or hits
        if chosen:
            hit = max(chosen, key=lambda h: int(h.get("size_bytes") or 0))
            work_dir = hit.get("work_dir") or fallback_work
            if work_dir:
                self._session_work_dirs[self._sk(session_id, owner)] = work_dir
            keep = Path(hit["path"])
            for shadow in hits:
                path = Path(shadow["path"])
                if path != keep and not self._jsonl_has_content(path):
                    try:
                        path.unlink()
                    except OSError:
                        pass
            return str(hit["base_dir"]), work_dir
        return fallback, fallback_work

    def _drop_cached_agents(self) -> None:
        """Make every session rebuild its agent with the current settings."""
        running = {key for key, lock in self._session_locks.items() if lock.locked()}
        for key in self._cache.keys():
            if key in running:
                self._stale_agents.add(key)
            else:
                self._cache.delete(key)

    def delete_session(self, session_id: str, owner: Optional[str] = None) -> bool:
        """Delete a session: cached Agent + persistent SessionLog JSONL + meta."""
        base_dir, _work_dir = self._locate_session(session_id, owner)
        key = self._sk(session_id, owner)
        removed = self._cache.delete(key)
        _pkg().live_turn.drop(session_id, owner=self._owner(owner))
        self._session_work_dirs.pop(key, None)
        self._session_approval_modes.pop(key, None)
        self._session_grants.pop(key, None)
        background = self._session_background.pop(key, None)
        bg_records_dir = Path(base_dir) / session_id / "background"
        if background is not None or bg_records_dir.is_dir():
            threading.Thread(
                target=stop_all_session_jobs,
                args=(bg_records_dir, background, self._owner(owner)),
                daemon=True,
            ).start()
        self._session_locks.pop(key, None)
        self._stale_agents.discard(key)
        log_existed = False
        try:
            log = SessionLog(session_id=session_id, base_dir=base_dir)
            with self._session_preview_lock:
                self._session_preview_cache.pop(str(log.path), None)
            if log.path.exists():
                log.path.unlink()
                log_existed = True
            meta = log.base_dir / f"{session_id}.meta.json"
            if meta.exists():
                meta.unlink()
            session_artifacts = log.base_dir / session_id
            if session_artifacts.is_dir():
                shutil.rmtree(session_artifacts, ignore_errors=True)
        except Exception as e:
            logger.warning(f"Failed to remove SessionLog for {session_id}: {e}")
        logger.debug(f"Session deleted: {session_id}")
        return removed or log_existed

    # ============== Background commands (the Web's /ps and /stop) ==============

    def _session_bg_records_dir(self, session_id: str, owner: Optional[str]) -> Path:
        base_dir, _ = self._locate_session(session_id, owner)
        return Path(base_dir) / session_id / "background"

    def _background_registry(self, session_id: str, owner: Optional[str]) -> BackgroundProcessRegistry:
        key = self._sk(session_id, owner)
        uid = self._owner(owner)
        records_dir = self._session_bg_records_dir(session_id, owner)
        registry = self._session_background.get(key)
        if registry is None:
            registry = BackgroundProcessRegistry(
                user_id=uid,
                delivers_results=self.background_report_handler is not None,
                session_id=session_id,
                records_dir=records_dir,
                on_change=lambda: _pkg().live_turn.bump_pulse(uid),
            )
            self._session_background[key] = registry
            threading.Thread(
                target=self._deliver_background_reports,
                args=(registry, session_id, owner),
                daemon=True,
                name=f"bg-report-{session_id}",
            ).start()
        else:
            registry.bind_session(session_id=session_id, records_dir=records_dir)
        return registry

    def _deliver_background_reports(
        self, registry: BackgroundProcessRegistry, session_id: str, owner: Optional[str],
    ) -> None:
        """Hand each finished command back to the agent, exactly once.

        The durable record is the source, not this process's completion queue: a
        job started by the CLI has a record here and never had an event in this
        registry, and a job whose owner process died has a record and no event
        anywhere. The queue is only a wakeup, so the sweep also runs on a timer
        to adopt reports that finished while nothing was watching.
        """
        idle = 0
        while self._background_registry_is_current(session_id, owner, registry):
            try:
                registry.wait_completed(timeout=1.0)
                idle = 0
            except queue.Empty:
                idle += 1
            if idle and idle % 5:
                continue
            loop = self._loop
            if loop is None or loop.is_closed():
                return
            future = asyncio.run_coroutine_threadsafe(
                self._sweep_undelivered_background(session_id, owner), loop,
            )
            future.add_done_callback(_log_report_failure)

    def _background_registry_is_current(
        self, session_id: str, owner: Optional[str], registry: BackgroundProcessRegistry,
    ) -> bool:
        """False once the session is deleted: its pending reports are void."""
        return self._session_background.get(self._sk(session_id, owner)) is registry

    async def _sweep_undelivered_background(self, session_id: str, owner: Optional[str]) -> None:
        """Deliver every finished-but-unreported job for this session, once.

        Waits for the session to be idle *before* claiming, so a long busy
        period cannot expire the claim and let a second process take the same
        record. The claim itself then covers only the hand-off, which is short.
        """
        records_dir = self._session_bg_records_dir(session_id, owner)
        if not undelivered_job_records(records_dir):
            return
        registry = self._session_background.get(self._sk(session_id, owner))
        if registry is None:
            return
        while self._background_registry_is_current(session_id, owner, registry) and (
            self.is_session_active(session_id, owner) or _pkg().live_turn.active(session_id, self._owner(owner))
        ):
            await asyncio.sleep(1.0)
        handler = self.background_report_handler
        if handler is None or not self._background_registry_is_current(session_id, owner, registry):
            return
        who = delivery_owner("gateway")
        for path, rec in undelivered_job_records(records_dir):
            if not claim_job_delivery(path, who):
                continue
            report = background_report_from_record(rec)
            if report is None:
                # Settled with nothing to say (stopped, or no exit code): mark
                # it rather than leaving it permanently owed.
                mark_job_delivered(path, who)
                continue
            try:
                await handler(session_id, self._owner(owner), report)
            except Exception as exc:  # noqa: BLE001 - one bad record must not stop the sweep
                logger.warning(f"background report delivery failed for {path.name}: {exc}")
                note_job_delivery_failure(path)
                release_job_claim(path)
                continue
            mark_job_delivered(path, who)

    @staticmethod
    def _background_job_view(item: BackgroundProcess) -> Dict[str, Any]:
        return background_job_view_from_process(item)

    def list_background_jobs(self, session_id: str, owner: Optional[str] = None) -> List[Dict[str, Any]]:
        registry = self._session_background.get(self._sk(session_id, owner))
        records_dir = self._session_bg_records_dir(session_id, owner)
        return load_session_job_views(records_dir, registry)

    def background_job_output(
        self, session_id: str, job_id: str, owner: Optional[str] = None,
        max_lines: int = 400, max_chars: int = 20000,
    ) -> Optional[Dict[str, Any]]:
        target = job_id.strip()
        if not target or target.lower() in {"all", "*"}:
            return None
        views = self.list_background_jobs(session_id, owner=owner)
        matched = next(
            (
                v for v in views
                if target in {v["id"], str(v.get("num")), f"#{v.get('num')}", str(v.get("pid"))}
            ),
            None,
        )
        if matched is None:
            return None
        log_path = str(matched.get("log_path") or "")
        return {
            **matched,
            "output": read_log_tail(log_path, max_lines=max_lines, max_chars=max_chars) if log_path else "",
            "log_path": log_path,
        }

    async def stop_background_job(
        self, session_id: str, job_id: str, owner: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        registry = self._session_background.get(self._sk(session_id, owner))
        records_dir = self._session_bg_records_dir(session_id, owner)
        uid = self._owner(owner)
        stopped = await asyncio.to_thread(stop_session_job, records_dir, job_id, registry, uid)
        if stopped is not None:
            _pkg().live_turn.bump_pulse(uid)
        return stopped

    async def stop_all_session_background_jobs(
        self, session_id: str, owner: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        registry = self._session_background.get(self._sk(session_id, owner))
        records_dir = self._session_bg_records_dir(session_id, owner)
        uid = self._owner(owner)
        stopped = await asyncio.to_thread(stop_all_session_jobs, records_dir, registry, uid)
        if stopped:
            _pkg().live_turn.bump_pulse(uid)
        return stopped

    def stop_all_background_jobs(self) -> None:
        for registry in list(self._session_background.values()):
            registry.stop()

    def fork_session(self, session_id: str, at_uuid: Optional[str] = None, owner: Optional[str] = None) -> str:
        """Copy a session up to ``at_uuid`` (inclusive) into a new one; return its id."""
        log = self.session_log_for(session_id, owner=owner)
        if not log.path.exists() or log.path.stat().st_size == 0:
            raise FileNotFoundError(f"Session not found: {session_id}")
        if at_uuid and not log.has_entry(at_uuid):
            raise ValueError(f"No entry {at_uuid} in session {session_id}")
        new_id = str(uuid.uuid4())
        branch = log.fork(new_id, at_uuid=at_uuid)
        branch.set_source(RunSource.gateway.value)
        meta = SessionLog._read_meta(log.meta_path)
        title = SessionLog._meta_name(meta) or (chat_images.strip_markers(log.first_user_message() or "") or "Chat")[:40]
        SessionLog.rename_session(new_id, f"{title} (fork)", base_dir=str(log.base_dir))
        if log._cwd:
            self._session_work_dirs[self._sk(new_id, owner)] = log._cwd
        return new_id

    def session_feedback(self, session_id: str, owner: Optional[str] = None) -> Dict[str, Dict[str, Any]]:
        return self.session_log_for(session_id, owner=owner).read_feedback()

    def set_session_feedback(
        self, session_id: str, entry_uuid: str, rating: Optional[str], comment: str = "",
        tags: Optional[List[str]] = None, owner: Optional[str] = None,
    ) -> Dict[str, Dict[str, Any]]:
        log = self.session_log_for(session_id, owner=owner)
        if not log.path.exists():
            raise FileNotFoundError(f"Session not found: {session_id}")
        if not log.has_entry(entry_uuid):
            raise ValueError(f"No entry {entry_uuid} in session {session_id}")
        return log.set_feedback(entry_uuid, rating, comment, tags)

    def rename_session(self, session_id: str, name: str, owner: Optional[str] = None) -> None:
        """Rename a session by writing the sidecar .meta.json (SessionLog)."""
        base_dir, _work_dir = self._locate_session(session_id, owner)
        SessionLog.rename_session(session_id, name, base_dir=base_dir)

    def _schedule_title(self, session_id: str, owner: Optional[str], question: str, answer: str) -> None:
        """Name an unnamed session from its first exchange, in the background."""
        if not answer.strip():
            return
        key = (owner or "", session_id)
        if key in self._titled:
            return
        self._titled.add(key)
        log = self.session_log_for(session_id, owner=owner)
        if log.get_name():
            return
        task = asyncio.create_task(self._write_title(log, question, answer))
        self._title_tasks.add(task)
        task.add_done_callback(self._title_tasks.discard)

    async def _write_title(self, log: SessionLog, question: str, answer: str) -> None:
        prompt = (
            f"User:\n{chat_images.strip_markers(question)[:2000]}\n\n"
            f"Assistant:\n{answer[:2000]}"
        )
        try:
            title = await self.side_task(_TITLE_PROMPT, prompt)
        except Exception as e:
            logger.warning(f"Session title generation failed ({log.session_id}): {e}")
            return
        title = title.splitlines()[0].strip().strip("\"'“”「」").strip() if title else ""
        if title and not log.get_name():
            log.set_name(title[:60])

    def archive_session(
        self, session_id: str, archived: bool = True, owner: Optional[str] = None,
    ) -> None:
        """Archive/unarchive a session by writing SessionLog sidecar metadata."""
        base_dir, _work_dir = self._locate_session(session_id, owner)
        SessionLog.archive_session(session_id, archived=archived, base_dir=base_dir)

    def pin_session(
        self, session_id: str, pinned: bool = True, owner: Optional[str] = None,
    ) -> None:
        """Pin/unpin a session (sidebar order) by writing sidecar metadata."""
        base_dir, _work_dir = self._locate_session(session_id, owner)
        SessionLog.pin_session(session_id, pinned=pinned, base_dir=base_dir)

    def mark_session_read(self, session_id: str, owner: Optional[str] = None) -> str:
        """Stamp ``last_read_at`` so the sidebar drops the unread dot."""
        base_dir, _work_dir = self._locate_session(session_id, owner)
        return SessionLog.mark_session_read(session_id, base_dir=base_dir)

    def clear_session(self, session_id: str, owner: Optional[str] = None) -> bool:
        """Alias for delete_session (for compatibility)."""
        return self.delete_session(session_id, owner)

    def set_session_work_dir(
        self, session_id: str, work_dir: str, owner: Optional[str] = None,
    ) -> None:
        """Set the working directory for a specific session."""
        self._session_work_dirs[self._sk(session_id, owner)] = work_dir

    def get_session_work_dir(self, session_id: str, owner: Optional[str] = None) -> str:
        """Get the working directory for a session (falls back to global base_dir)."""
        key = self._sk(session_id, owner)
        if key not in self._session_work_dirs:
            self._locate_session(session_id, owner)
        return self._session_work_dirs.get(
            key, str(settings.base_dir),
        )

    def _session_base_dir(self, work_dir: str, owner: Optional[str] = None) -> str:
        """Resolve the SessionLog storage dir for a given project work_dir."""
        from agentica.config.project import project_base_dir

        return project_base_dir(work_dir, user_id=self._owner(owner))
