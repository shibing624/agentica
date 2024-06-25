import csv
from pathlib import Path
from typing import Any

from actionflow.file.base import File
from actionflow.utils.log import logger


class CsvFile(File):
    type: str = "CSV"

    def get_metadata(self) -> dict[str, Any]:
        if self.name is None:
            self.name = Path(self.data_path).name

        if self.columns is None:
            try:
                # Get the columns from the file
                with open(self.data_path) as f:
                    dict_reader = csv.DictReader(f)
                    if dict_reader.fieldnames is not None:
                        self.columns = list(dict_reader.fieldnames)
            except Exception as e:
                logger.warning(f"Error getting columns from file: {e}")

        return self.model_dump(exclude_none=True)
