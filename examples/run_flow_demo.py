"""
This module is used to run Actionflow flows. To run one, use the following command:

.. code-block:: bash
    cd examples
    python run_flow_demo.py --flow_path=flows/example.json --variables '<variable>=<value>' '<variable>=<value>'

Optionally, use -v for verbose output.

"""

import argparse
import sys

from loguru import logger

sys.path.append('..')
from actionflow import Flow


def parse_variables(variables: list[str]) -> dict[str, str]:
    """
    Parses the variables provided as command line arguments.

    :param variables: A list of strings where each string is a key-value pair in the format 'key=value'.
    :type variables: list[str]
    :return: A dictionary where the keys are the variable names and the values are the corresponding values.
    :rtype: dict[str, str]
    """
    if not variables:
        return {}

    variable_dict = {}
    for variable in variables:
        key, value = variable.split("=")
        variable_dict[key] = value
    return variable_dict


def main():
    """
    The main function that parses command line arguments and runs the specified flow.
    """
    parser = argparse.ArgumentParser(description="Actionflow")
    parser.add_argument(
        "--flow_path",
        type=str,
        default="flows/example.json",
        help="The file path of the flow to run. the json file",
    )
    parser.add_argument(
        "--variables",
        nargs="*",
        help="Variables to be used in the flow. Should be in the format key1=value1 key2=value2. Put key=value pairs in quotes if they contain space.",
        dest="variables",
    )
    parser.add_argument(
        "--verbose", action="store_true", help="Show detailed output."
    )

    args = parser.parse_args()
    variables = parse_variables(args.variables)
    if args.verbose:
        logger.info("Verbose mode enabled.")
        logger.info(f"Flow name: {args.flow}, variables: {variables}")

    flow = Flow(flow_path=args.flow_path, variables=variables)
    flow.run()


if __name__ == "__main__":
    main()
