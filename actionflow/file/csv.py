from pathlib import Path
from typing import Any

from loguru import logger

from actionflow.file.base import File


class CsvFile(File):
    path: str
    type: str = "CSV"

    def get_metadata(self) -> dict[str, Any]:
        if self.name is None:
            self.name = Path(self.path).name

        if self.columns is None:
            try:
                # Get the columns from the file
                import csv

                with open(self.path) as csvfile:
                    dict_reader = csv.DictReader(csvfile)
                    if dict_reader.fieldnames is not None:
                        self.columns = list(dict_reader.fieldnames)
            except Exception as e:
                logger.warning(f"Error getting columns from file: {e}")

        return self.model_dump(exclude_none=True)
