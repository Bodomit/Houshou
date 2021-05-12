#!/usr/bin/env python3
import os
from datetime import datetime, timezone

import yaml
import namegenerator

APP_CONFIG_PATH = "appconfig.yaml"


def get_results_directory() -> str:
    try:
        with open(APP_CONFIG_PATH, "r") as appconfig_file:
            app_config = yaml.safe_load(appconfig_file)
    except FileNotFoundError:
        app_config = {}

    root_results_directory = app_config.get("root_results_directory", "results")

    experiment_name = namegenerator.gen()
    datetime_str = datetime.now(timezone.utc).strftime("%y%m%d_%H%M%S")
    results_directory_name = f"{datetime_str}_{experiment_name}"

    results_directory = os.path.join(root_results_directory, results_directory_name)
    return results_directory


if __name__ == "__main__":
    results_directory = get_results_directory()
    print(results_directory)
