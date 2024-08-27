# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description:
"""
from typing import List, Optional, Any, Dict

from pydantic import BaseModel


class File(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    columns: Optional[List[str]] = None
    data_path: Optional[str] = None
    type: str = "FILE"

    def get_metadata(self) -> Dict[str, Any]:
        return self.model_dump(exclude_none=True)
