from pathlib import Path
from typing import Any

from actionflow.file.base import File


class TextFile(File):
    type: str = "TEXT"

    def get_metadata(self) -> dict[str, Any]:
        if self.name is None:
            self.name = Path(self.data_path).name
        return self.model_dump(exclude_none=True)
