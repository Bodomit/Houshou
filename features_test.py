import os
from argparse import ArgumentParser
from collections import OrderedDict
from typing import Generator, Type, Union, get_args

import pytorch_lightning as pl
import torch
from torch.utils.data.dataloader import DataLoader

from houshou.data import CelebA, Market1501, VGGFace2
from houshou.data.celeba import CelebADataset
from houshou.data.market_1501 import Market1501Dataset
from houshou.data.vggface2 import VGGFace2Dataset
from houshou.metrics import CVThresholdingVerifier, ReidentificationTester
from houshou.models import FeatureModel
from houshou.systems import (Alvi2019, MultitaskTrainer,
                             TwoStageMultitaskTrainer)
from houshou.utils import find_last_epoch_path, save_cv_verification_results

pl.seed_everything(42, workers=True)

HOUSHOU_DATASET = Union[VGGFace2Dataset, CelebADataset, Market1501Dataset]


def main(experiment_path: str, trainer_type: str, batch_size: int, is_debug: bool, is_fullbody: bool):

    if trainer_type == "Alvi2019":
        trainer_class = Alvi2019
    elif trainer_type == "Multitask":
        trainer_class = MultitaskTrainer
    elif trainer_type == "TwoStage":
        trainer_class = TwoStageMultitaskTrainer
    else:
        raise ValueError

    feature_model_checkpoint_path = find_last_epoch_path(experiment_path)

    print("Experiment Directory: ", experiment_path)
    print("Checkpoint Path; ", feature_model_checkpoint_path)

    # Load the multitask model and get the featrue model.
    try:
        multitask_trainer = trainer_class.load_from_checkpoint(
            feature_model_checkpoint_path, verifier_args=None
        )
    except RuntimeError:
        multitask_trainer = backwards_compatible_load(
            feature_model_checkpoint_path, trainer_class)

    assert isinstance(multitask_trainer, trainer_class)
    feature_model = multitask_trainer.model.feature_model
    del multitask_trainer
    assert isinstance(feature_model, FeatureModel)
    feature_model.freeze()

    # Construct the datamodules.
    if is_fullbody:
        datamodules = [Market1501(batch_size, ["gender"], buffer_size=None)]
    else:
        datamodules = [
            VGGFace2(batch_size, ["Male"], buffer_size=None),
            CelebA(batch_size, ["Male"], buffer_size=None),
        ]
    # Get the device.
    device = torch.device("cuda:0")
    feature_model.to(device)

    for test_module in datamodules:

        test_module.setup("test")
        test_dataloader = test_module.test_dataloader()
        assert isinstance(test_dataloader, DataLoader)
        test_dataset = test_dataloader.dataset
        assert any(isinstance(test_dataset, t) for t in get_args(HOUSHOU_DATASET))
        dataset_name = os.path.basename(test_module.data_dir)

        print(f"Dataset Module: {dataset_name}")

        for n_classes in n_classes_scheduler(len(test_dataset.classes)):  # type: ignore

            print(f"N Classes: {n_classes}")

            reid_scenario(
                experiment_path,
                dataset_name,
                n_classes,
                batch_size,
                is_debug,
                test_dataloader,
                feature_model,
                device)

            verification_scenario(
                experiment_path,
                dataset_name,
                n_classes,
                batch_size,
                is_debug,
                test_dataloader,
                feature_model,
                device)


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

    new_model = trainer_class(**hyper_parameters)
    new_model.load_state_dict(combined)

    return new_model


def reid_scenario(
        experiment_path: str,
        dataset_name: str,
        n_classes: int,
        batch_size: int,
        is_debug: bool,
        test_dataloader: DataLoader,
        feature_model: torch.nn.Module,
        device: torch.device):
    results_dir = os.path.join(
        experiment_path, "feature_tests", "reid", dataset_name, str(n_classes)
    )
    os.makedirs(results_dir, exist_ok=True)

    tester = ReidentificationTester(
        batch_size, max_n_classes=n_classes, debug=is_debug, seed=42)

    tester.setup(test_dataloader)


def verification_scenario(
        experiment_path: str,
        dataset_name: str,
        n_classes: int,
        batch_size: int,
        is_debug: bool,
        test_dataloader: DataLoader,
        feature_model: torch.nn.Module,
        device: torch.device):

    results_dir = os.path.join(
        experiment_path, "feature_tests", "verification", dataset_name, str(n_classes)
    )
    os.makedirs(results_dir, exist_ok=True)

    verifier = CVThresholdingVerifier(
        batch_size, max_n_classes=n_classes, debug=is_debug
    )

    verifier.setup(test_dataloader)

    (
        metrics_rocs,
        per_attribute_pair_metrics_rocs,
    ) = verifier.cv_thresholding_verification(feature_model, device)

    # Save the combined cv verification results.
    save_cv_verification_results(metrics_rocs, results_dir, "_all")

    # Save the attribuet pair results.
    for ap in per_attribute_pair_metrics_rocs:
        ap_suffix = f"_{ap[0]}_{ap[1]}"
        save_cv_verification_results(
            per_attribute_pair_metrics_rocs[ap], results_dir, ap_suffix
        )


def n_classes_scheduler(
    total_n_classes: int, start=25, step_f=lambda x: 2 * x
) -> Generator[int, None, None]:
    x = start
    yield x

    while x < total_n_classes:
        x = step_f(x)

        if x < total_n_classes:
            yield x
        else:
            yield total_n_classes
            return


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("experiment_path")
    parser.add_argument("trainer_type")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--fullbody", action="store_true")
    args = parser.parse_args()
    main(args.experiment_path, args.trainer_type, args.batch_size, args.debug, args.fullbody)
