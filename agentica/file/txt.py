# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description:
"""
from __future__ import annotations
from pathlib import Path
from typing import Any, Dict

from agentica.file.base import File


class TextFile(File):
    type: str = "TEXT"

    def get_metadata(self) -> Dict[str, Any]:
        if self.name is None:
            self.name = Path(self.data_path).name
        return self.model_dump(exclude_none=True)
