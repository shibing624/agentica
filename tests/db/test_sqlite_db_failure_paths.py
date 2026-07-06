# -*- coding: utf-8 -*-
"""
Tests for SqliteDb failure-path semantics:

- "no such table" errors trigger a single create + retry, not unbounded recursion.
- Other database errors propagate immediately (no silent swallow, no recursion).
- SQLite engine has WAL + busy_timeout pragmas applied.
"""
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pytest

pytest.importorskip("sqlalchemy", reason="SqliteDb tests require agentica[sql]")

from sqlalchemy.exc import IntegrityError, OperationalError

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agentica.db.base import MemoryRow
from agentica.db.sqlite import SqliteDb


class _FakeSession:
    """Minimal stand-in for a SQLAlchemy Session that always raises on execute."""

    def __init__(self, exc):
        self.exc = exc

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False

    def execute(self, *_args, **_kwargs):
        raise self.exc

    def commit(self):  # pragma: no cover - never reached
        raise AssertionError("commit should not be called when execute raises")


def _make_integrity_error() -> IntegrityError:
    return IntegrityError("INSERT ...", None, Exception("UNIQUE constraint failed"))


def _make_missing_table_error() -> OperationalError:
    return OperationalError(
        "SELECT ...",
        None,
        Exception("no such table: agentica_memories"),
    )


class TestSqliteDbFailurePaths(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.db_path = Path(self._tmp.name) / "test.db"
        self.db = SqliteDb(db_file=str(self.db_path))

    def tearDown(self):
        self._tmp.cleanup()

    def test_missing_table_triggers_single_create_and_retry(self):
        """If the table genuinely vanishes, we recreate it once and the op succeeds."""
        memory = MemoryRow(id="m1", user_id="u1", memory={"k": "v"})

        self.db._get_memory_table().drop(self.db.db_engine)

        with patch.object(
            self.db,
            "create_memory_table",
            wraps=self.db.create_memory_table,
        ) as create_spy:
            result = self.db.upsert_memory(memory)

        create_spy.assert_called_once()
        self.assertIsNotNone(result)
        self.assertEqual(result.id, "m1")

        roundtrip = self.db.read_memories(user_id="u1")
        self.assertEqual(len(roundtrip), 1)
        self.assertEqual(roundtrip[0].id, "m1")

    def test_persistent_missing_table_does_not_recurse(self):
        """If create_table is a no-op (simulating persistent failure) we retry exactly once and propagate."""
        memory = MemoryRow(id="m1", user_id="u1", memory={"k": "v"})
        self.db._get_memory_table().drop(self.db.db_engine)

        with patch.object(self.db, "create_memory_table") as create_spy:
            with self.assertRaises(OperationalError):
                self.db.upsert_memory(memory)

        create_spy.assert_called_once()

    def test_non_missing_table_error_propagates_without_create(self):
        """IntegrityError (or any non missing-table error) must propagate, no retry, no create."""
        memory = MemoryRow(id="m1", user_id="u1", memory={"k": "v"})

        def fake_session_factory():
            return _FakeSession(_make_integrity_error())

        with patch.object(self.db, "create_memory_table") as create_spy, patch.object(
            self.db, "Session", side_effect=fake_session_factory
        ):
            with self.assertRaises(IntegrityError):
                self.db.upsert_memory(memory)

        create_spy.assert_not_called()

    def test_insert_metrics_does_not_recurse_on_persistent_failure(self):
        """insert_metrics had the same recursive bug; verify the fix."""
        from agentica.db.base import MetricsRow

        metrics = MetricsRow(
            id="x1",
            agent_id="a",
            session_id="s",
            run_id="r",
            metrics={"foo": "bar"},
        )

        def always_raise():
            return _FakeSession(_make_missing_table_error())

        with patch.object(self.db, "create_metrics_table") as create_spy, patch.object(
            self.db, "Session", side_effect=always_raise
        ):
            with self.assertRaises(OperationalError):
                self.db.insert_metrics(metrics)

        create_spy.assert_called_once()

    def test_upsert_session_does_not_recurse_on_persistent_failure(self):
        from agentica.db.base import SessionRow

        row = SessionRow(session_id="s1", agent_id="a1")

        def always_raise():
            return _FakeSession(_make_missing_table_error())

        with patch.object(self.db, "create_session_table") as create_spy, patch.object(
            self.db, "Session", side_effect=always_raise
        ):
            with self.assertRaises(OperationalError):
                self.db.upsert_session(row)

        create_spy.assert_called_once()

    def test_wal_and_busy_timeout_pragmas_applied(self):
        """File-backed SQLite engines must run with WAL + a sane busy_timeout."""
        with self.db.db_engine.connect() as conn:
            journal_mode = conn.exec_driver_sql("PRAGMA journal_mode").fetchone()[0]
            busy_timeout = conn.exec_driver_sql("PRAGMA busy_timeout").fetchone()[0]

        self.assertEqual(str(journal_mode).lower(), "wal")
        self.assertGreaterEqual(int(busy_timeout), 1000)


if __name__ == "__main__":
    unittest.main()
