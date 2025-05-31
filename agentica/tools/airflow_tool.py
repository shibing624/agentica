# -*- encoding: utf-8 -*-
"""
@author: orange-crow, XuMing(xuming624@qq.com)
@description:
part of the code is from phidata
"""
from pathlib import Path
from typing import Optional, Union

from agentica.utils.log import logger
from agentica.tools.base import Tool


class AirflowTool(Tool):
    def __init__(self, dags_dir: Optional[Union[Path, str]] = None, save_dag: bool = True, read_dag: bool = True):
        super().__init__(name="AirflowTool")

        _dags_dir: Optional[Path] = None
        if dags_dir is not None:
            if isinstance(dags_dir, str):
                _dags_dir = Path.cwd().joinpath(dags_dir)
            else:
                _dags_dir = dags_dir
        self.dags_dir: Path = _dags_dir or Path.cwd()
        if save_dag:
            self.register(self.save_dag_file, sanitize_arguments=False)
        if read_dag:
            self.register(self.read_dag_file)

    def save_dag_file(self, contents: str, dag_file: str) -> str:
        """Saves Python code for an Airflow DAG to a file called `dag_file` and returns the file path if successful.

        Args:
            `contents` (str): The contents of the DAG.
            `dag_file` (str): The name of the file to save to.

        Example:
            ```python
            from agentica.tools.airflow_tool import AirflowTool
            m = AirflowTool()
            result = m.save_dag_file(contents="DAG code here", dag_file="example_dag.py")
            print(result)
            ```

        Returns:
            str: The file path if successful, otherwise returns an error message.
        """

        try:
            file_path = self.dags_dir.joinpath(dag_file)
            logger.debug(f"Saving contents to {file_path}")
            if not file_path.parent.exists():
                file_path.parent.mkdir(parents=True, exist_ok=True)
            file_path.write_text(contents)
            logger.info(f"Saved: {file_path}")
            return str(str(file_path))
        except Exception as e:
            logger.error(f"Error saving to file: {e}")
            return f"Error saving to file: {e}"

    def read_dag_file(self, dag_file: str) -> str:
        """Reads an Airflow DAG file `dag_file` and returns the contents if successful.

        Args:
            `dag_file` (str): The name of the file to read.

        Example:
            ```python
            from agentica.tools.airflow_tool import AirflowTool
            m = AirflowTool()
            contents = m.read_dag_file(dag_file="example_dag.py")
            print(contents)
            ```

        Returns:
            str: The contents of the file if successful, otherwise returns an error message.
        """

        try:
            logger.info(f"Reading file: {dag_file}")
            file_path = self.dags_dir.joinpath(dag_file)
            contents = file_path.read_text()
            return str(contents)
        except Exception as e:
            logger.error(f"Error reading file: {e}")
            return f"Error reading file: {e}"
