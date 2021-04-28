import os
from typing import List


def parse_root_results_directory_argument(args: List[str]):
    ARG_NAME = "results_directory="
    for arg in args:
        if arg.startswith(ARG_NAME):
            results_dir = arg.split(ARG_NAME)[1]
            root_results_dir = os.path.dirname(results_dir)
            return root_results_dir
    raise ValueError(f"No argument {ARG_NAME} provided.")
