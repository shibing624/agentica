# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description:
"""

import sys

sys.path.append('..')
from actionflow import Assistant

if __name__ == "__main__":
    flow = Assistant(flow_path="flows/example_summarize_url.json")
    print(flow)
    flow.run()
