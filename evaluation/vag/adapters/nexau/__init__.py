# -*- coding: utf-8 -*-
"""
NexAU adapter (paper appendix).

NexAU is the internal "Next-Action Utility" benchmark referenced in the VaG
appendix. The full data pipeline is not open-source; this adapter exposes a
schema-compatible loader and a tiny synthetic split so reviewers can
reproduce the harness on a placeholder dataset.
"""
from evaluation.vag.adapters.nexau.adapter import NexAUTask, load_split, run_split

__all__ = ["NexAUTask", "load_split", "run_split"]
