# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: 
"""
import os
import tempfile
import pytest
from datetime import datetime
from actionflow.run_record import RunRecord
from actionflow.sqlite_storage import SqliteStorage

# Mock current_datetime function
def mock_current_datetime():
    return datetime(2023, 1, 1, 0, 0, 0)

@pytest.fixture
def sqlite_storage():
    # Create a temporary file for the SQLite database
    db_fd, db_path = tempfile.mkstemp()
    os.close(db_fd)

    # Initialize the SqliteStorage with the temporary database file
    storage = SqliteStorage(
        table_name="assistant_runs",
        db_file=db_path
    )

    yield storage

    # Cleanup: remove the temporary database file
    os.remove(db_path)

def test_write_and_read(sqlite_storage):
    # Create a RunRecord instance
    run_record = RunRecord(
        run_id="test_run_id",
        name="test_name",
        run_name="test_run_name",
        user_id="test_user_id",
        llm={"model": "test_model"},
        memory={"key": "value"},
        assistant_data={"assistant_key": "assistant_value"},
        run_data={"run_key": "run_value"},
        user_data={"user_key": "user_value"},
        task_data={"task_key": "task_value"},
        created_at=mock_current_datetime(),
        updated_at=mock_current_datetime()
    )

    # Write the RunRecord to the database
    sqlite_storage.create()
    sqlite_storage.add

    # Read the RunRecord from the database
    retrieved_record = sqlite_storage.read(run_id="test_run_id")

    # Assert that the retrieved record matches the original record
    assert retrieved_record is not None
    assert retrieved_record.run_id == run_record.run_id
    assert retrieved_record.name == run_record.name
    assert retrieved_record.run_name == run_record.run_name
    assert retrieved_record.user_id == run_record.user_id
    assert retrieved_record.llm == run_record.llm
    assert retrieved_record.memory == run_record.memory
    assert retrieved_record.assistant_data == run_record.assistant_data
    assert retrieved_record.run_data == run_record.run_data
    assert retrieved_record.user_data == run_record.user_data
    assert retrieved_record.task_data == run_record.task_data
    assert retrieved_record.created_at == run_record.created_at
    assert retrieved_record.updated_at == run_record.updated_at