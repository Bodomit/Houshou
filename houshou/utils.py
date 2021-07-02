import glob
import os
import pickle
import re
from collections import OrderedDict
from functools import partial
from typing import Dict, List, Tuple, Type

import pandas as pd
import pytorch_lightning as pl
import torch
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


def get_lambdas(input_directory: str) -> List[str]:
    pattern = os.path.join(input_directory, "*")
    lambda_paths = glob.glob(pattern)
    lambda_values: List[str] = []
    for bn in (os.path.basename(p) for p in lambda_paths):
        try:
            float(bn)
            lambda_values.append(bn)
        except ValueError:
            continue
    return lambda_values


def sort_lambdas(lambda_values: List[str]):
    sorted_lambdas = list(sorted(lambda_values, key=lambda x: float(x)))
    return sorted_lambdas


def backwards_compatible_load(
        feature_model_checkpoint_path: str,
        trainer_class: Type[MultitaskTrainer]) -> torch.nn.Module:

    old_checkpoint = torch.load(feature_model_checkpoint_path)
    old_state_dict = old_checkpoint['state_dict']

    hyper_parameters = old_checkpoint["hyper_parameters"]
    hyper_parameters["verifier_args"] = None

    new_state_dict = OrderedDict({k.replace(".resnet.", ".feature_model."): v
                                  for k, v in old_state_dict.items()})

    # Due to me being dumb, the weights are referenced twice in the same model...
    # Both the old and new mappings must be present.
    combined = OrderedDict(new_state_dict | old_state_dict)

    try:
        new_model = trainer_class(**hyper_parameters)
    except TypeError:
        # Hack
        if "shm_uniformkldivergence" in feature_model_checkpoint_path:
            hyper_parameters["loss_a"] = "UNIFORM_KLDIVERGENCE"
            hyper_parameters["loss_f"] = "SEMIHARD_MINED_TRIPLETS"
            hyper_parameters["weight_attributes"] = False
            hyper_parameters["classification_training_scenario"] = False
            hyper_parameters["use_short_attribute_branch"] = True
            new_model = trainer_class(**hyper_parameters)
        else:
            raise

    new_model.load_state_dict(combined)

    return new_model
