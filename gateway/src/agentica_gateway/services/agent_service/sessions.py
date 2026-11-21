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

from agentica.session.catalog import (
    DEFAULT_RECENT_DAYS,
    Catalog,
    LiveState,
    forget_preview,
    im_source_of_session_id,
    list_catalog,
)
from agentica.session.log import SessionLog, iso_timestamp
from agentica.session.rows import strip_image_markers
from agentica.run.context import RunSource
from agentica.tools.background_processes import (
    BackgroundProcess,
    BackgroundProcessCompleted,
    BackgroundProcessRegistry,
    background_job_view_from_process,
    background_report_from_record,
    background_result_for_agent,
    claim_job_delivery,
    completed_from_record,
    delivery_owner,
    load_session_job_views,
    mark_job_delivered,
    note_job_delivery_failure,
    read_log_tail,
    record_file_is_settled,
    release_job_claim,
    render_batch_report,
    stop_all_session_jobs,
    stop_session_job,
    undelivered_job_records,
)
from agentica.utils.log import logger

from ...config import settings
from .types import (
    _TITLE_PROMPT,
    _log_report_failure,
)


def _pkg():
    return sys.modules["agentica_gateway.services.agent_service"]


#: How often a deferred background report re-checks whether its session went
#: idle. Named so tests can shorten it instead of sleeping through real seconds.
BACKGROUND_IDLE_POLL_SECONDS = 1.0


class AgentSessionsMixin:
    """Session discovery, fork/feedback/rename/archive/delete, and background processes."""

    def live_state(self, owner: Optional[str] = None) -> LiveState:
        """What this process knows about the owner's sessions and the disk does not."""
        uid = self._owner(owner)
        prefix = self._sk("", uid)
        running = {turn.session_id for turn in _pkg().live_turn.iter_owner(uid) if not turn.done}
        running |= {
            key[len(prefix):] for key, lock in self._session_locks.items()
            if key.startswith(prefix) and lock.locked()
        }
        return LiveState(
            running=running,
            registries={k[len(prefix):]: v for k, v in self._session_background.items() if k.startswith(prefix)},
            approval_modes={k[len(prefix):]: v for k, v in self._session_approval_modes.items() if k.startswith(prefix)},
        )

    def session_catalog(
        self,
        owner: Optional[str] = None,
        *,
        work_dir: Optional[str] = None,
        scope: str = "recent",
        recent_days: int = DEFAULT_RECENT_DAYS,
        include_cli: bool = False,
        hide_temp_dirs: bool = False,
    ) -> Catalog:
        """The owner's session list (``agentica.session.catalog``) with this process's live state."""
        catalog = list_catalog(
            user_id=self._owner(owner),
            work_dir=work_dir,
            scope=scope,
            recent_days=recent_days,
            include_cli=include_cli,
            hide_temp_dirs=hide_temp_dirs,
            workspace_dir=str(settings.base_dir),
            live=self.live_state(owner),
            fallback_base_dir=self._session_base_dir(str(settings.base_dir), owner),
        )
        self._add_unlogged_live_rows(catalog, owner, work_dir)
        return catalog

    def list_sessions(self, owner: Optional[str] = None, **kwargs: Any) -> List[Dict[str, Any]]:
        return self.session_catalog(owner, **kwargs).rows

    def _add_unlogged_live_rows(self, catalog: Catalog, owner: Optional[str], work_dir: Optional[str]) -> None:
        """A run started a moment ago has a LiveTurn and no log line yet: list it now."""
        uid = self._owner(owner)
        seen = {row["session_id"] for row in catalog.rows}
        real_dir = os.path.realpath(work_dir) if work_dir else ""
        for turn in _pkg().live_turn.iter_owner(uid):
            if turn.done or turn.session_id in seen:
                continue
            turn_dir = self.get_session_work_dir(turn.session_id, owner=uid)
            if real_dir and os.path.realpath(turn_dir or "") != real_dir:
                continue
            live_log = self.session_log_for(turn.session_id, owner=uid)
            source = live_log.get_source() or im_source_of_session_id(turn.session_id) or "gateway"
            catalog.rows.insert(0, {
                "session_id": turn.session_id,
                "name": SessionLog.display_title(
                    name=live_log.get_name(),
                    first_user=strip_image_markers(turn.user_message or ""),
                    cron_job_name=((turn.cron_job or {}).get("name") or ""),
                ),
                "preview": "",
                "user_count": 0,
                "first_timestamp": iso_timestamp(),
                "last_timestamp": None,
                "last_read_at": None,
                "mtime": time.time(),
                "unread": False,
                "size_bytes": 0,
                "archived": False,
                "pinned": False,
                "work_dir": turn_dir,
                "running": True,
                "background_jobs": 0,
                "source": source,
                "origin": source,
                "approval_mode": self._saved_approval_mode(turn.session_id, owner=uid),
            })

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
        key = self._sk(session_id, owner)
        remembered = self._session_work_dirs.get(key)
        if remembered and key in self._session_log_dirs:
            return self._session_log_dirs[key], remembered

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
            work_dir = hit.get("workspace_dir") or hit.get("work_dir") or fallback_work
            self._session_work_dirs[key] = work_dir
            self._session_log_dirs[key] = str(hit["base_dir"])
            keep = Path(hit["path"])
            for shadow in hits:
                path = Path(shadow["path"])
                if path != keep and not self._jsonl_has_content(path):
                    try:
                        path.unlink()
                    except OSError:
                        pass
            return self._session_log_dirs[key], work_dir
        return fallback, fallback_work

    def _drop_cached_agents(self) -> None:
        """Make every session rebuild its agent with the current settings."""
        running = {key for key, lock in self._session_locks.items() if lock.locked()}
        for key in self._cache.keys():
            if key in running:
                self._stale_agents.add(key)
            else:
                self._cache.delete(key)

    def invalidate_session_agent(
        self, session_id: str, owner: Optional[str] = None,
    ) -> None:
        """Make one session rebuild its agent after its cwd/settings changed."""
        key = self._sk(session_id, owner)
        lock = self._session_locks.get(key)
        if lock is not None and lock.locked():
            self._stale_agents.add(key)
            return
        self._cache.delete(key)

    def delete_session(self, session_id: str, owner: Optional[str] = None) -> bool:
        """Delete a session: cached Agent + persistent SessionLog JSONL + meta."""
        base_dir, _work_dir = self._locate_session(session_id, owner)
        key = self._sk(session_id, owner)
        removed = self._cache.delete(key)
        _pkg().live_turn.drop(session_id, owner=self._owner(owner))
        self._session_work_dirs.pop(key, None)
        self._session_log_dirs.pop(key, None)
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
            forget_preview(str(log.path))
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
        key = self._sk(session_id, owner)
        # Events whose durable record is unusable. Held, not dropped: while a
        # sweep is in flight the queue still has to be drained, and discarding
        # these here would lose the only copy of the report.
        unusable: List[BackgroundProcessCompleted] = []
        while self._background_registry_is_current(session_id, owner, registry):
            event = None
            try:
                event = registry.wait_completed(timeout=1.0)
                idle = 0
            except queue.Empty:
                idle += 1
            if (
                event is not None
                and not event.stop_requested
                and not record_file_is_settled(event.record_path)
            ):
                unusable.append(event)
            loop = self._loop
            if loop is None or loop.is_closed():
                return
            # One delivery job per session at a time. A sweep waits for the
            # conversation to be idle, so on a long turn every tick would
            # otherwise stack another waiting coroutine.
            if key in self._bg_delivery_inflight:
                continue
            if unusable:
                # The record cannot be adopted by the sweep (no transcript, or
                # the best-effort completion write failed), so the event is the
                # only copy left.
                pending_event = unusable.pop(0)
                self._bg_delivery_inflight.add(key)
                future = asyncio.run_coroutine_threadsafe(
                    self._report_background_event(session_id, owner, registry, pending_event), loop,
                )
                future.add_done_callback(self._background_delivery_done(key))
                continue
            if idle and idle % 5:
                continue
            self._bg_delivery_inflight.add(key)
            future = asyncio.run_coroutine_threadsafe(
                self._sweep_undelivered_background(session_id, owner), loop,
            )
            future.add_done_callback(self._background_delivery_done(key))

    def _background_delivery_done(self, key: str):
        """Release the per-session delivery slot and log a failed coroutine."""

        def _cb(future) -> None:
            self._bg_delivery_inflight.discard(key)
            _log_report_failure(future)

        return _cb

    def _background_registry_is_current(
        self, session_id: str, owner: Optional[str], registry: BackgroundProcessRegistry,
    ) -> bool:
        """False once the session is deleted: its pending reports are void."""
        return self._session_background.get(self._sk(session_id, owner)) is registry

    async def _report_background_event(
        self,
        session_id: str,
        owner: Optional[str],
        registry: BackgroundProcessRegistry,
        event: BackgroundProcessCompleted,
    ) -> None:
        """Deliver a report whose durable record cannot be adopted.

        Kept because the record is written best-effort: when the completion
        write fails there is no record to claim, and dropping the event would
        bring back exactly the loss this change is about.
        """
        report = background_result_for_agent(event)
        while self._background_registry_is_current(session_id, owner, registry) and (
            self.is_session_active(session_id, owner) or _pkg().live_turn.active(session_id, self._owner(owner))
        ):
            await asyncio.sleep(BACKGROUND_IDLE_POLL_SECONDS)
        handler = self.background_report_handler
        if handler is None or not self._background_registry_is_current(session_id, owner, registry):
            return
        await handler(session_id, self._owner(owner), report)

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
            await asyncio.sleep(BACKGROUND_IDLE_POLL_SECONDS)
        handler = self.background_report_handler
        if handler is None or not self._background_registry_is_current(session_id, owner, registry):
            return
        who = delivery_owner("gateway")
        # Same pacing as the CLI sweep: a session that comes back after downtime
        # can owe several reports, and they are handed over as ONE turn so the
        # phone and the page get one summary instead of one per job.
        claimed: List[tuple[Path, Dict[str, Any], BackgroundProcessCompleted]] = []
        for path, rec in undelivered_job_records(records_dir)[:5]:
            if not claim_job_delivery(path, who):
                continue
            report = background_report_from_record(rec)
            if report is None:
                # Settled with nothing to say (stopped, or no exit code): mark
                # it rather than leaving it permanently owed.
                mark_job_delivered(path, who)
                continue
            event = completed_from_record(rec)
            if event is None or event.stop_requested:
                mark_job_delivered(path, who)
                continue
            claimed.append((path, rec, event))
        if not claimed:
            return
        try:
            await handler(
                session_id, self._owner(owner),
                render_batch_report([event for _, _, event in claimed]),
            )
        except Exception as exc:  # noqa: BLE001 - the batch stays owed, not marked
            logger.warning(f"batched background report delivery failed: {exc}")
            for path, _, _ in claimed:
                note_job_delivery_failure(path)
                release_job_claim(path)
            return
        for path, _, _ in claimed:
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
        title = SessionLog.display_title(
            name=log.get_name(),
            first_user=strip_image_markers(log.first_user_message() or ""),
        )
        SessionLog.rename_session(new_id, f"{title} (fork)", base_dir=str(log.base_dir))
        branch.set_workspace_dir(self.get_session_work_dir(session_id, owner))
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
        task = asyncio.create_task(
            self._write_title(log, question, answer, self._owner(owner))
        )
        self._title_tasks.add(task)
        task.add_done_callback(self._title_tasks.discard)

    async def _write_title(self, log: SessionLog, question: str, answer: str, owner: str) -> None:
        prompt = (
            f"User:\n{strip_image_markers(question)[:2000]}\n\n"
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
            _pkg().live_turn.bump_pulse(owner)

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
        """Set the effective directory, keeping existing history in place."""
        key = self._sk(session_id, owner)
        base_dir, current = self._locate_session(session_id, owner)
        if key not in self._session_log_dirs:
            base_dir = self._session_base_dir(work_dir, owner)
        elif current != work_dir and (Path(base_dir) / f"{session_id}.jsonl").is_file():
            SessionLog(session_id, base_dir=base_dir).set_workspace_dir(work_dir)
        self._session_log_dirs[key] = base_dir
        self._session_work_dirs[key] = work_dir

    def get_session_work_dir(self, session_id: str, owner: Optional[str] = None) -> str:
        """Get the working directory for a session (falls back to global base_dir)."""
        key = self._sk(session_id, owner)
        if key not in self._session_work_dirs:
            self._locate_session(session_id, owner)
        return self._session_work_dirs.get(
            key, str(settings.base_dir),
        )

    def session_workspace_dir(self, session_id: str, owner: str) -> Optional[str]:
        """Read an existing binding without treating the global default as one."""
        self._locate_session(session_id, owner)
        return self._session_work_dirs.get(self._sk(session_id, owner))

    def _session_base_dir(self, work_dir: str, owner: Optional[str] = None) -> str:
        """Resolve the SessionLog storage dir for a given project work_dir."""
        from agentica.config.project import project_base_dir

        return project_base_dir(work_dir, user_id=self._owner(owner))
