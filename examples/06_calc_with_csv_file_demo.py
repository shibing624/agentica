# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description:
"""
import sys

sys.path.append('..')
from agentica.file.csv import CsvFile
from agentica import PythonAgent

m = PythonAgent(
    files=[
        CsvFile(
            data_path="https://phidata-public.s3.amazonaws.com/demo_data/IMDB-Movie-Data.csv",
            description="Contains information about movies from IMDB.",
        )
    ]
)

r = m.run("What is the min rating of movies?")
print(r)
