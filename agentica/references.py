# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description:
part of the code from https://github.com/phidatahq/phidata
"""
from typing import Optional
from pydantic import BaseModel


class References(BaseModel):
    """Model for LLM references"""

    # The question asked by the user.
    query: str
    # The references from the vector database.
    references: Optional[str] = None
    # Performance in seconds.
    time: Optional[float] = None
