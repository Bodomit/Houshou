import glob
import os
import pickle
import re
from functools import partial
from typing import Dict, List, Tuple, Type

import pandas as pd
import pytorch_lightning as pl
from ruyaml import YAML

from houshou.common import ROCCurve
from houshou.systems import (Alvi2019, MultitaskTrainer,
                             TwoStageMultitaskTrainer)


def parse_root_results_directory_argument(args: List[str]):
    ARG_NAME = "results_directory="
    for arg in args:
        if arg.startswith(ARG_NAME):
            results_dir = arg.split(ARG_NAME)[1]
            root_results_dir = os.path.dirname(results_dir)
            return root_results_dir
    raise ValueError(f"No argument {ARG_NAME} provided.")


def find_last_epoch_path(path: str) -> str:
    pattern = os.path.join(path, "**", "epoch*.ckpt")
    paths = glob.glob(pattern, recursive=True)

    if not paths:
        raise ValueError

    if len(paths) == 1:
        return paths[0]

    paths_with_epochs_steps: List[Tuple[str, int, int]] = []
    for path in paths:
        match = re.match(r"epoch=(\d+)-step=(\d+)\.ckpt$", os.path.basename(path))
        try:
            assert match
            epoch, step = match.groups()
            paths_with_epochs_steps.append((path, int(epoch), int(step)))
        except AssertionError:
            continue

    sorted_paths_with_epochs_steps = sorted(
        paths_with_epochs_steps,
        key=lambda x: (x[1], x[2]),
        reverse=True,
    )

    return sorted_paths_with_epochs_steps[0][0]


def load_experiment_config(path: str) -> Dict:
    pattern = os.path.join(path, "**", "config.yaml")
    paths = glob.glob(pattern, recursive=True)

    if not paths:
        raise ValueError

    if len(paths) == 1:
        config_path = paths[0]

    paths = list(sorted(paths, reverse=True))
    config_path = paths[0]

    yaml = YAML(typ="safe")

    with open(config_path, "r") as infile:
        config = yaml.load(infile)

    return config


def get_model_class_from_config(path: str) -> Type[MultitaskTrainer]:
    try:
        config = load_experiment_config(path)
        trainer_class_path = config["model"]["class_path"]
    except KeyError:
        trainer_class_path = "houshou.systems.TwoStageMultitaskTrainer"

    if trainer_class_path == "houshou.systems.Alvi2019":
        trainer_class = Alvi2019
    elif trainer_class_path == "houshou.systems.MultitaskTrainer":
        trainer_class = MultitaskTrainer
    elif trainer_class_path == "houshou.systems.TwoStageMultitaskTrainer":
        trainer_class = TwoStageMultitaskTrainer
    else:
        raise ValueError

    return trainer_class


def save_cv_verification_results(
    metrics_rocs: Tuple[pd.DataFrame, List[ROCCurve]],
    val_results_path: str,
    suffix="",
):
    fn = partial(os.path.join, val_results_path)

    metrics, rocs = metrics_rocs
    metrics.to_csv(fn(f"verification{suffix}.csv"))

    with open(fn(f"roc_curves{suffix}.pickle"), "wb") as outfile:
        pickle.dump(rocs, outfile)
