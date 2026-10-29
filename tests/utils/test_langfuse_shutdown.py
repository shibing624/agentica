"""Bounded Langfuse exit shutdown (start_shutdown_thread / shutdown_langfuse_bounded).

The CLI must not be held hostage by langfuse's ~2s atexit cleanup (span
force_flush + consumer-thread polls); the bounded path overlaps teardown and
caps the wait. Tests pin the behavioral contract: no client spawned just to
be shut down, daemon thread semantics, and the join timeout being honored.
"""
import base64
import threading
import time
import unittest
from unittest.mock import MagicMock, patch

import pytest

import agentica.utils.langfuse_integration as li


def _require_langfuse():
    pytest.importorskip("langfuse", reason="Langfuse tests require the langfuse extra")


class TestLangfuseMediaGuard(unittest.TestCase):
    def test_sse_frame_is_not_a_media_uri(self):
        self.assertFalse(
            li._looks_like_base64_data_uri(
                'data: {"choices":[{"delta":{"role":"assistant"}}]}'
            )
        )

    def test_real_image_data_uri_is_recognized(self):
        self.assertTrue(
            li._looks_like_base64_data_uri("data:image/png;base64,iVBORw0KGgo=")
        )

    def test_guard_swallows_sse_frames(self):
        _require_langfuse()
        from langfuse.media import LangfuseMedia

        li._install_langfuse_media_guard()
        media = LangfuseMedia(
            base64_data_uri='data: {"choices":[{"delta":{"role":"assistant"}}]}'
        )
        self.assertIsNone(media._content_bytes)

    def test_guard_still_parses_real_base64_data_uri(self):
        _require_langfuse()
        from langfuse.media import LangfuseMedia

        li._install_langfuse_media_guard()
        raw = b"\x01\x02\x03\x04"
        import base64

        uri = "data:audio/wav;base64," + base64.b64encode(raw).decode()
        media = LangfuseMedia(base64_data_uri=uri)
        self.assertEqual(media._content_bytes, raw)

    def test_guard_does_not_log_incorrect_padding(self):
        """A data URI chopped at 4000 chars used to log Incorrect padding.

        That is what ``analyze_image`` hit: the tool result carries the whole
        screenshot, ``bound_langfuse_value`` used to slice it, and Langfuse
        still tried to decode the slice.
        """
        _require_langfuse()
        from langfuse.media import LangfuseMedia

        li._install_langfuse_media_guard()
        raw = b"\xff\xd8\xff" + b"\x00" * 5000
        uri = "data:image/jpeg;base64," + base64.b64encode(raw).decode()
        chopped = uri[:4000] + "...[truncated]"
        self.assertGreater(len(uri), 4000)
        with self.assertNoLogs("langfuse", level="ERROR"):
            media = LangfuseMedia(base64_data_uri=chopped)
        self.assertIsNone(media._content_bytes)


class TestBoundLangfuseMedia(unittest.TestCase):
    def _screenshot_uri(self) -> str:
        raw = b"\xff\xd8\xff" + b"\x00" * 5000
        return "data:image/jpeg;base64," + base64.b64encode(raw).decode()

    def test_decodable_data_uri_is_not_truncated(self):
        uri = self._screenshot_uri()
        self.assertGreater(len(uri), 4000)
        self.assertEqual(li.bound_langfuse_value(uri), uri)
        envelope = {
            "_multimodal": True,
            "text": "look",
            "images": [{"url": uri}],
        }
        bounded = li.bound_langfuse_value(envelope)
        self.assertEqual(bounded["images"][0]["url"], uri)
        self.assertEqual(bounded["text"], "look")

    def test_chopped_data_uri_is_not_sent_as_media(self):
        uri = self._screenshot_uri()
        chopped = uri[:4000] + "...[truncated]"
        omitted = li.bound_langfuse_value(chopped)
        self.assertFalse(omitted.startswith("data:"))
        self.assertIn("image/jpeg", omitted)

    def test_ordinary_long_text_is_still_capped(self):
        text = "hello " * 2000
        bounded = li.bound_langfuse_value(text)
        self.assertTrue(bounded.endswith("...[truncated]"))
        self.assertLessEqual(len(bounded), 4000 + len("...[truncated]"))


class TestStartShutdownThread(unittest.TestCase):
    def test_returns_none_when_not_configured(self):
        with patch.object(li, "is_langfuse_configured", return_value=False):
            self.assertIsNone(li.start_shutdown_thread())
            self.assertFalse(
                any(t.name == "langfuse-shutdown" for t in threading.enumerate())
            )

    def test_does_not_spawn_client_when_nothing_traced(self):
        """get_client() lazily CREATES a client; the peek at the singleton
        registry must short-circuit first so a never-traced session doesn't
        pay shutdown cost it never incurred."""
        _require_langfuse()
        from langfuse._client.resource_manager import LangfuseResourceManager

        with (
            patch.object(li, "is_langfuse_configured", return_value=True),
            patch.object(LangfuseResourceManager, "_instances", {}),
            patch("langfuse.get_client") as mock_get_client,
        ):
            self.assertIsNone(li.start_shutdown_thread())
            mock_get_client.assert_not_called()

    def test_peek_failure_degrades_to_none(self):
        """Private-API drift must never break the caller's exit path."""
        _require_langfuse()

        # The registry peek reads ``_instances`` off the CLASS, so the raising
        # descriptor has to live on the metaclass: a plain ``@property`` here
        # would just hand back the property object (truthy, no raise), letting
        # the call fall through to get_client() and — whenever langfuse happens
        # to be importable/initialised in this process — spawn a real
        # ``langfuse-shutdown`` thread that outlives the test and breaks its
        # siblings.
        class _BoomMeta(type):
            @property
            def _instances(cls):
                raise RuntimeError("api drift")

        class _BoomRM(metaclass=_BoomMeta):
            _lock = threading.Lock()

        with (
            patch.object(li, "is_langfuse_configured", return_value=True),
            patch(
                "langfuse._client.resource_manager.LangfuseResourceManager", _BoomRM
            ),
        ):
            self.assertIsNone(li.start_shutdown_thread())
        self.assertFalse(
            any(t.name == "langfuse-shutdown" for t in threading.enumerate()),
            "degrading to None must not leave a shutdown thread behind",
        )

    def test_shutdown_runs_on_daemon_thread(self):
        client = MagicMock()
        done = threading.Event()
        client.shutdown.side_effect = lambda: done.set()

        _require_langfuse()
        from langfuse._client.resource_manager import LangfuseResourceManager

        with (
            patch.object(li, "is_langfuse_configured", return_value=True),
            patch.object(LangfuseResourceManager, "_instances", {"pk": object()}),
            patch("langfuse.get_client", return_value=client),
        ):
            thread = li.start_shutdown_thread()

        self.assertIsNotNone(thread)
        assert thread is not None
        self.assertTrue(thread.daemon)
        self.assertEqual(thread.name, "langfuse-shutdown")
        self.assertTrue(done.wait(timeout=5), "shutdown() never ran")
        thread.join(timeout=5)
        client.shutdown.assert_called_once()

    def test_shutdown_exception_stays_in_thread(self):
        client = MagicMock()
        client.shutdown.side_effect = RuntimeError("boom")

        _require_langfuse()
        from langfuse._client.resource_manager import LangfuseResourceManager

        with (
            patch.object(li, "is_langfuse_configured", return_value=True),
            patch.object(LangfuseResourceManager, "_instances", {"pk": object()}),
            patch("langfuse.get_client", return_value=client),
        ):
            thread = li.start_shutdown_thread()

        assert thread is not None
        # join must not re-raise the client's failure
        thread.join(timeout=5)
        self.assertFalse(thread.is_alive())


class TestShutdownLangfuseBounded(unittest.TestCase):
    def test_join_timeout_is_honored(self):
        blocked = threading.Event()

        def _slow_shutdown():
            blocked.wait(timeout=30)

        client = MagicMock()
        client.shutdown.side_effect = _slow_shutdown

        _require_langfuse()
        from langfuse._client.resource_manager import LangfuseResourceManager

        with (
            patch.object(li, "is_langfuse_configured", return_value=True),
            patch.object(LangfuseResourceManager, "_instances", {"pk": object()}),
            patch("langfuse.get_client", return_value=client),
        ):
            t0 = time.monotonic()
            li.shutdown_langfuse_bounded(timeout=0.3)
            elapsed = time.monotonic() - t0

        blocked.set()  # release the daemon thread
        self.assertLess(elapsed, 5.0, f"bounded shutdown held for {elapsed:.2f}s")
        self.assertGreaterEqual(elapsed, 0.25)

    def test_noop_when_unconfigured(self):
        with patch.object(li, "is_langfuse_configured", return_value=False):
            t0 = time.monotonic()
            li.shutdown_langfuse_bounded(timeout=30)
            self.assertLess(time.monotonic() - t0, 1.0)


if __name__ == "__main__":
    unittest.main()
