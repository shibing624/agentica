# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description:
"""
from dataclasses import dataclass
from typing import List, Optional, Any, Dict
from agentica.utils.misc import dataclass_to_dict


@dataclass
class File:
    name: Optional[str] = None
    description: Optional[str] = None
    columns: Optional[List[str]] = None
    data_path: Optional[str] = None
    type: str = "FILE"

    def get_metadata(self) -> Dict[str, Any]:
        return dataclass_to_dict(self, exclude_none=True)