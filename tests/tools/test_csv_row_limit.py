import csv
import json

import pytest

from agentica.utils.file_parser import read_csv_file


@pytest.mark.parametrize("limit, expected", [(0, "[]"), (1, '[{"value": "first"}]')])
def test_csv_row_limit_does_not_parse_later_records(tmp_path, limit, expected):
    path = tmp_path / "records.csv"
    path.write_text("value\nfirst\n" + "x" * (csv.field_size_limit() + 1) + "\n", encoding="utf-8")

    assert read_csv_file(path, row_limit=limit) == expected


@pytest.mark.parametrize(
    "limit, expected",
    [
        (-1, '[{"value": "first"}, {"value": "second"}]'),
        (-2, '[{"value": "first"}]'),
    ],
)
def test_csv_negative_row_limit_preserves_exact_slice_output(tmp_path, limit, expected):
    path = tmp_path / "records.csv"
    path.write_text("value\nfirst\nsecond\nthird\n", encoding="utf-8")

    assert read_csv_file(path, row_limit=limit) == expected


@pytest.mark.parametrize("limit", [None, -1, 0, 1, 2, 10, 10**30])
def test_csv_row_limit_preserves_existing_slice_results(tmp_path, limit):
    path = tmp_path / "records.csv"
    path.write_text("value\nfirst\nsecond\n", encoding="utf-8")
    rows = [{"value": "first"}, {"value": "second"}]

    assert json.loads(read_csv_file(path, row_limit=limit)) == rows[:limit]


def test_csv_error_within_requested_records_is_still_reported(tmp_path):
    path = tmp_path / "records.csv"
    path.write_text("value\n" + "x" * (csv.field_size_limit() + 1) + "\n", encoding="utf-8")

    assert read_csv_file(path, row_limit=1) == ""
