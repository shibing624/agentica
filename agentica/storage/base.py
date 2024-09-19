# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: 
"""
from abc import ABC, abstractmethod
from typing import Optional, List
from agentica.run_record import RunRecord


class AssistantStorage(ABC):
    @abstractmethod
    def create(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def read(self, run_id: str) -> Optional[RunRecord]:
        raise NotImplementedError

    @abstractmethod
    def get_all_run_ids(self, user_id: Optional[str] = None) -> List[str]:
        raise NotImplementedError

    @abstractmethod
    def get_all_runs(self, user_id: Optional[str] = None) -> List[RunRecord]:
        raise NotImplementedError

    @abstractmethod
    def upsert(self, row: RunRecord) -> Optional[RunRecord]:
        raise NotImplementedError

    @abstractmethod
    def delete(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def export(self, csv_file_path: str, user_id: Optional[str] = None) -> None:
        raise NotImplementedError
