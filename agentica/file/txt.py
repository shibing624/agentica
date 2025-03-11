# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description:
"""
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

from agentica.utils.misc import dataclass_to_dict
from agentica.file.base import File


@dataclass
class TextFile(File):
    type: str = "TEXT"

    def get_metadata(self) -> Dict[str, Any]:
        if self.name is None:
            self.name = Path(self.data_path).name
        return dataclass_to_dict(self, exclude_none=True)
