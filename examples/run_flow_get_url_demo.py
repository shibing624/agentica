# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description:
"""

import sys

sys.path.append('..')
from actionflow import ActionFlow

if __name__ == "__main__":
    flow = ActionFlow(flow_path="flows/example_summarize_url.json")
    flow.run()
