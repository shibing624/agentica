# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description:
"""
from typing import Optional
from time import perf_counter


class Timer:
    """Timer class for timing code execution"""

    def __init__(self):
        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None
        self.elapsed_time: Optional[float] = None

    @property
    def elapsed(self) -> float:
        return self.elapsed_time or (perf_counter() - self.start_time) if self.start_time else 0.0

    def start(self) -> float:
        self.start_time = perf_counter()
        return self.start_time

    def stop(self) -> float:
        self.end_time = perf_counter()
        if self.start_time is not None:
            self.elapsed_time = self.end_time - self.start_time
        return self.end_time

    def __enter__(self):
        self.start_time = perf_counter()
        return self

    def __exit__(self, *args):
        self.end_time = perf_counter()
        if self.start_time is not None:
            self.elapsed_time = self.end_time - self.start_time


def format_turn_clock(seconds: float) -> str:
    """Last-turn clock on the status bar.

    Under a minute keep a tenth (``12.3s``). Past a minute split units
    (``30m3s``, ``5m13s``, ``16m``) so it is not a four-digit second count.
    Past an hour keep every non-zero unit (``1h2m3s``, ``1h3s``). A zero
    unit is omitted (``1h``, not ``1h0m0s``): this is the live turn, and
    the clock has to tick while the CLI is still running.
    """
    if seconds < 60:
        return f"{seconds:.1f}s"
    total = max(0, int(seconds + 0.5))
    hours, rem = divmod(total, 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        text = f"{hours}h"
        if minutes:
            text += f"{minutes}m"
        if secs:
            text += f"{secs}s"
        return text
    return f"{minutes}m{secs}s" if secs else f"{minutes}m"
