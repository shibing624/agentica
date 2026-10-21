# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: Scale the local token estimate by what the provider actually counted.

The compression layers decide on ``count_tokens``, a tiktoken estimate of the
request. It cannot see what the provider adds on its side: replayed encrypted
reasoning items (OpenAI Responses), image tokens it prices differently, its own
tokenizer. Measured on a 9-day production session (gpt-6.1-sol, 1.05M window):
the estimate said ~640k while the gateway billed 1,047,795 for the same request.
No layer ever fired — not the 75% reminder, not evict at 0.8, not the 95% cut —
and the session hit ``input exceeds the context window`` for hours.

The provider reports the real prompt size of every request it answers
(``Message.metrics["context_tokens"]``, set by the model adapters). Comparing it
with the estimate made for that same request gives a ratio, and every later
estimate is multiplied by it. That is protocol data, not a guess.

The ratio only ever scales *up*. An estimate that runs high (CJK char fallback
when tiktoken is offline) compacts a little early, which is today's behaviour;
one that runs low is the failure above. A provider whose figure is lower than
the estimate therefore leaves the estimate alone.
"""

from dataclasses import dataclass
from typing import List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from agentica.model.message import Message


@dataclass
class ContextCalibration:
    """Per-model ratio between provider-counted and locally-estimated prompt size."""

    ratio: float = 1.0
    # The estimate made for the last request sent, and the message it ended
    # with. Identity, not an index: the next run rebuilds the message list, and
    # an index into the old one would pair a reply with the wrong estimate.
    _sent_estimate: int = 0
    _sent_tail: Optional["Message"] = None

    def observe(self, messages: List["Message"]) -> None:
        """Update the ratio from the reply to the last recorded request, if present."""
        tail = self._sent_tail
        estimate = self._sent_estimate
        if tail is None or estimate <= 0:
            return
        start = next((i for i, m in enumerate(messages) if m is tail), -1)
        if start < 0:
            # The list was rebuilt or the window cut: that pairing is gone.
            self._sent_tail = None
            return
        for message in messages[start + 1:]:
            if message.role != "assistant":
                continue
            reported = message.metrics.get("context_tokens") or 0
            if reported > 0:
                self.ratio = max(1.0, reported / estimate)
            self._sent_tail = None
            return

    def scale(self, local_tokens: int) -> int:
        return round(local_tokens * self.ratio)

    def record_send(self, messages: List["Message"], local_tokens: int) -> None:
        """Remember the estimate for the request about to go out."""
        if not messages or local_tokens <= 0:
            return
        self._sent_tail = messages[-1]
        self._sent_estimate = local_tokens
