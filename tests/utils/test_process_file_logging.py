"""CLI/gateway opt-in file logging (SDK stays silent by default)."""
import logging
import os
from datetime import datetime
from pathlib import Path

import agentica.config
from agentica.utils.log import enable_process_file_logging, logger


def _detach_file_handler(path: str) -> None:
    abs_path = os.path.abspath(path)
    for handler in list(logger.handlers):
        if isinstance(handler, logging.FileHandler) and handler.baseFilename == abs_path:
            logger.removeHandler(handler)
            handler.close()


def test_enable_process_file_logging_opts_out_on_empty_env(monkeypatch):
    monkeypatch.setenv("AGENTICA_LOG_FILE", "")
    assert enable_process_file_logging() == ""


def test_enable_process_file_logging_defaults_to_dated_pid_path(tmp_path, monkeypatch):
    monkeypatch.delenv("AGENTICA_LOG_FILE", raising=False)
    monkeypatch.setattr("agentica.config.AGENTICA_HOME", str(tmp_path))
    monkeypatch.setattr("agentica.config.AGENTICA_LOG_FILE", "")

    path = enable_process_file_logging()
    try:
        expected = tmp_path / "logs" / f"{datetime.now().strftime('%Y%m%d')}-{os.getpid()}.log"
        assert Path(path) == expected
        assert agentica.config.AGENTICA_LOG_FILE == path
    finally:
        _detach_file_handler(path)


def test_enable_process_file_logging_reuses_explicit_path(tmp_path, monkeypatch):
    explicit = str(tmp_path / "custom.log")
    monkeypatch.delenv("AGENTICA_LOG_FILE", raising=False)
    monkeypatch.setattr("agentica.config.AGENTICA_LOG_FILE", explicit)

    path = enable_process_file_logging()
    try:
        assert path == explicit
    finally:
        _detach_file_handler(path)
