# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: `agentica watch`: tracking a job agentica did not start, by exit-code checks.
"""
import os
import signal
import subprocess
import sys
import time

import pytest

if os.name == "nt":
    pytest.skip("checks run through /bin/sh", allow_module_level=True)

from agentica.tools.background_processes import (
    BackgroundProcessCompleted,
    BackgroundProcessRegistry,
    background_result_for_agent,
    read_log_tail,
)
from agentica.tools.job_watch import LOST_CONTACT_EXIT, RESULT_UNKNOWN_EXIT, new_lines, watch

# What `agentica watch` runs, without depending on the console script being installed.
WATCH = [sys.executable, "-c", "import sys; sys.argv[0] = 'agentica'; from agentica.cli.main import main; main()", "watch"]


def _job(tmp_path, body: str) -> str:
    """Launch a detached job that leaves job.exit with its exit code, like a user would."""
    script = tmp_path / "job.sh"
    script.write_text(body)
    subprocess.Popen(
        f"sh {script} > {tmp_path}/job.log 2>&1; echo $? > {tmp_path}/job.exit",
        shell=True, start_new_session=True,
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    return str(tmp_path)


def _spec(d: str, **extra) -> dict:
    spec = {
        "done": f"test -f {d}/job.exit",
        "progress": f"tail -n 2 {d}/job.log",
        "result": f"tail -n 5 {d}/job.log; exit $(cat {d}/job.exit)",
        "interval": 0.05,
    }
    spec.update(extra)
    return spec


class TestWatchLoop:
    def test_ends_with_the_jobs_exit_code_and_streams_its_output(self, tmp_path, capsys):
        d = _job(tmp_path, "echo step 1; sleep 0.3; echo step 2; sleep 0.3; echo boom; exit 3\n")
        assert watch(_spec(d)) == 3
        out = capsys.readouterr().out
        streamed = out.split("[result]")[0]
        # Each line once, in order, although consecutive tails overlap.
        assert streamed.count("step 1") == 1 and streamed.count("step 2") == 1
        assert streamed.index("step 1") < streamed.index("step 2")
        assert "[job finished: exit 3]" in out

    def test_failed_check_is_retried_not_read_as_the_job_ending(self, tmp_path, capsys):
        """A check exit that is neither 0 nor 1 (ssh's 255) is a lost check, not 'done'."""
        flaky = tmp_path / "flaky"
        d = _job(tmp_path, "sleep 0.6; exit 0\n")
        done = (
            f"n=$(cat {flaky} 2>/dev/null || echo 0); echo $((n+1)) > {flaky}; "
            f"[ $n -lt 2 ] && exit 255; test -f {d}/job.exit"
        )
        assert watch(_spec(d, done=done, max_failures=5)) == 0
        out = capsys.readouterr().out
        assert "probe failed (exit 255), job state unknown, retrying (1/5)" in out
        assert "[job finished: exit 0]" in out

    def test_lost_contact_exits_75_and_says_the_job_may_still_run(self, capsys):
        rc = watch({"done": "exit 255", "interval": 0.01, "max_failures": 3})
        assert rc == LOST_CONTACT_EXIT
        out = capsys.readouterr().out
        assert "retrying (3/3)" in out and "may still be running" in out

    def test_hung_check_is_killed_and_counted_as_a_failure(self, capsys):
        began = time.monotonic()
        rc = watch({"done": "sleep 30", "interval": 0.01, "max_failures": 1, "probe_timeout": 0.2})
        assert rc == LOST_CONTACT_EXIT
        assert time.monotonic() - began < 10
        assert "probe failed (timed out)" in capsys.readouterr().out

    def test_result_timeout_is_not_lost_contact(self, capsys):
        """The job already finished; a hung --result must not look like exit 75."""
        rc = watch({
            "done": "true",
            "result": "sleep 30",
            "interval": 0.01,
            "probe_timeout": 0.2,
        })
        assert rc == RESULT_UNKNOWN_EXIT
        out = capsys.readouterr().out
        assert "result probe timed out" in out
        assert "lost contact" not in out

    def test_without_result_a_finished_job_reports_exit_0(self, capsys):
        assert watch({"done": "true", "interval": 0.01}) == 0
        assert "[job finished]" in capsys.readouterr().out

    def test_new_lines_prints_only_what_follows_the_overlap(self):
        assert new_lines(["a", "b", "c"], ["b", "c", "d", "e"]) == ["d", "e"]
        assert new_lines([], ["a"]) == ["a"]
        assert new_lines(["a", "b"], ["a", "b"]) == []
        assert new_lines(["x"], ["y", "z"]) == ["y", "z"]


class TestAgenticaWatchCommand:
    def test_requires_done(self):
        proc = subprocess.run(WATCH, capture_output=True, text=True, timeout=60)
        assert proc.returncode == 2
        assert "--done" in proc.stderr

    def test_as_a_background_command_it_reports_the_jobs_exit(self, tmp_path):
        """The way the skill uses it: one execute(background=True) whose exit is the job's."""
        d = _job(tmp_path, "echo loss=0.5; sleep 0.5; echo loss=0.1; exit 4\n")
        spec = _spec(d)
        registry = BackgroundProcessRegistry()
        try:
            item = registry.start(
                "agentica watch ...", cwd=str(tmp_path), label="训练任务",
                argv=[*WATCH, "--done", spec["done"], "--progress", spec["progress"],
                      "--result", spec["result"], "--interval", "0.1"],
            )
            assert item.finished.wait(60)
        finally:
            registry.stop()
        assert item.returncode == 4
        tail = read_log_tail(item.log_path)
        assert "loss=0.1" in tail and "[job finished: exit 4]" in tail

    def test_stopping_the_watch_leaves_the_job_running(self, tmp_path):
        d = _job(tmp_path, "sleep 30\n")
        time.sleep(0.2)
        job_pids = subprocess.run(
            ["pgrep", "-f", f"sh {tmp_path}/job.sh"], capture_output=True, text=True,
        ).stdout.split()
        assert job_pids
        spec = _spec(d)
        watcher = subprocess.Popen(
            [*WATCH, "--done", spec["done"], "--progress", spec["progress"], "--interval", "0.1"],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, start_new_session=True,
        )
        try:
            time.sleep(1.5)
            watcher.send_signal(signal.SIGTERM)
            out, _ = watcher.communicate(timeout=10)
            # Died by the signal: the registry records "stopped", not a job failure.
            assert watcher.returncode == -signal.SIGTERM
            assert b"the job itself was not touched" in out
            for pid in job_pids:
                os.kill(int(pid), 0)  # raises if the job died with the watch
        finally:
            for pid in job_pids:
                try:
                    os.killpg(int(pid), signal.SIGKILL)
                except ProcessLookupError:
                    pass


def _completed(log_path: str, returncode: int) -> BackgroundProcessCompleted:
    now = time.time()
    return BackgroundProcessCompleted(
        id="term_1", num=1, pid=1, command="agentica watch --done 'exit 255'",
        cwd="", log_path=log_path, started_at=now - 5, completed_at=now,
        returncode=returncode,
    )


def test_lost_contact_report_does_not_call_the_job_failed(tmp_path):
    log = tmp_path / "w.log"
    log.write_text(
        "[lost contact: the done probe failed 3 times in a row. "
        "The job may still be running; fix the connection and watch it again.]\n"
    )
    report = background_result_for_agent(_completed(str(log), LOST_CONTACT_EXIT))
    assert report.startswith("[Background terminal #1 (term_1) lost contact:")
    assert "do not report the job as failed" in report


def test_a_job_that_exits_75_is_still_reported_as_failed(tmp_path):
    log = tmp_path / "w.log"
    log.write_text("[job finished: exit 75]\n")
    report = background_result_for_agent(_completed(str(log), 75))
    assert report.startswith("[Background terminal #1 (term_1) failed:")
    assert "lost contact" not in report


def test_result_timeout_report_says_the_job_ended(tmp_path):
    log = tmp_path / "w.log"
    log.write_text("[job finished, but the result probe timed out; its exit code is unknown]\n")
    report = background_result_for_agent(_completed(str(log), RESULT_UNKNOWN_EXIT))
    assert "result unknown" in report.splitlines()[0]
    assert "may still be running" not in report
    assert "exit code is unknown" in report
