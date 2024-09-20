# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: 
"""
from datetime import datetime

import pytest
from sqlalchemy import create_engine

from agentica.run_record import RunRecord
from agentica.storage.sqlite_storage import SqlAssistantStorage


# Mock current_datetime function
def mock_current_datetime():
    return datetime(2024, 1, 1, 0, 0, 0)


@pytest.fixture(scope="module")
def sqlite_storage():
    # 创建内存数据库
    engine = create_engine("sqlite:///:memory:")
    storage = SqlAssistantStorage(table_name="runs", db_engine=engine)
    yield storage
    # 测试结束后不需要清理，因为是内存数据库


def test_create_table(sqlite_storage):
    assert sqlite_storage.table_exists() is True


def test_upsert(sqlite_storage):
    run_record = RunRecord(
        run_id="1",
        name="Test Assistant",
        run_name="Test Run",
        user_id="user_1",
        llm={"model": "gpt-3"},
        memory={"key": "value"},
        assistant_data={"info": "test"},
        run_data={"data": "test"},
        user_data={"user_info": "test"},
        task_data={"task_info": "test"},
    )

    # 执行 upsert 操作
    result = sqlite_storage.upsert(run_record)
    assert result is not None
    assert result.run_id == "1"
    assert result.name == "Test Assistant"

    # 更新记录
    run_record.name = "Updated Assistant"
    result = sqlite_storage.upsert(run_record)
    assert result.name == "Updated Assistant"


def test_read(sqlite_storage):
    run_record = RunRecord(
        run_id="2",
        name="Another Assistant",
        run_name="Another Run",
        user_id="user_2",
        llm={"model": "gpt-3"},
        memory={"key": "value"},
        assistant_data={"info": "test"},
        run_data={"data": "test"},
        user_data={"user_info": "test"},
        task_data={"task_info": "test"},
    )

    sqlite_storage.upsert(run_record)
    read_record = sqlite_storage.read(run_id="2")
    assert read_record is not None
    assert read_record.run_id == "2"
    assert read_record.name == "Another Assistant"


def test_get_all_run_ids(sqlite_storage):
    run_record1 = RunRecord(run_id="3", name="Assistant 1", run_name="Run 1", user_id="user_3", llm={}, memory={})
    run_record2 = RunRecord(run_id="4", name="Assistant 2", run_name="Run 2", user_id="user_3", llm={}, memory={})

    sqlite_storage.upsert(run_record1)
    sqlite_storage.upsert(run_record2)

    run_ids = sqlite_storage.get_all_run_ids(user_id="user_3")
    assert len(run_ids) == 2
    assert "3" in run_ids
    assert "4" in run_ids


def test_delete(sqlite_storage):
    initial_run_count = len(sqlite_storage.get_all_run_ids())
    print("Initial run count:", initial_run_count)
    sqlite_storage.delete()
    assert sqlite_storage.table_exists() is False
