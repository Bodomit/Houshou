import os
from argparse import ArgumentParser
from typing import Dict, Tuple, Type

import pytorch_lightning as pl
from pytorch_lightning.loggers import CSVLogger, TensorBoardLogger
from ruyaml import YAML

from houshou.data import CelebA, Market1501, VGGFace2
from houshou.data.rap2 import RAP2
from houshou.models import FeatureModel
from houshou.systems import AttributeExtractionTask
from houshou.utils import (backwards_compatible_load, find_last_epoch_path,
                           get_lambdas, get_model_class_from_config,
                           sort_lambdas)

pl.seed_everything(42, workers=True)


def main(
    root_experiment_path: str, batch_size: int, is_fast_dev_run: bool, is_fullbody: bool, n_epochs: int
):
    experiments_per_lambda = find_lambda_experiments(root_experiment_path)

    if not experiments_per_lambda:
        raise ValueError

    # Construct the datamodules.
    if is_fullbody:
        datamodules = [
            Market1501(batch_size, ["gender"], buffer_size=None),
            RAP2(batch_size, ["Male"], buffer_size=None)
        ]
    else:
        datamodules = [
            VGGFace2(batch_size, ["Male"], buffer_size=None),
            CelebA(batch_size, ["Male"], buffer_size=None),
        ]

    yaml = YAML(typ="safe")

    root_dir = os.path.join(root_experiment_path, "attacker_aem_results", f"epochs_{n_epochs}")

    for train_module in datamodules:

        # Get directory for this test.
        root_dir_train = os.path.join(
            root_dir, os.path.basename(train_module.data_dir)
        )

        # Constuct loggers.
        loggers = [
            CSVLogger(save_dir=root_dir_train, name="csv_logs"),
            TensorBoardLogger(save_dir=root_dir_train, name="tb_logs"),
        ]

        # Train the attribute extraction model.
        aem_task = AttributeExtractionTask(None, None, 0.01, freeze_feature_model=False)
        trainer = pl.Trainer(
            logger=loggers,
            max_epochs=n_epochs,
            default_root_dir=root_dir_train,
            gpus=1,
            auto_select_gpus=True,
            benchmark=True,
            fast_dev_run=is_fast_dev_run,
        )
        trainer.fit(aem_task, train_module)

        aem_model = aem_task.attribute_model
        aem_model.freeze()

        del aem_task
        del trainer

        # Test each data module.
        for test_module in datamodules:
            
            print(train_module, test_module)

            root_dir_test = os.path.join(
                root_dir_train, f"{os.path.basename(test_module.data_dir)}"
            )
            os.makedirs(os.path.dirname(root_dir_test), exist_ok=True)

            for lambda_value, (tester_class, feature_model_checkpoint_path) in experiments_per_lambda.items():

                results_dir = os.path.join(root_dir_test)
                os.makedirs(results_dir, exist_ok=True)

                print("Lambda: ", lambda_value)
                print("Checkpoint Path: ", feature_model_checkpoint_path)
                print("Results Path: ", results_dir)

                # Load the multitask model and get the featrue model.
                try:
                    multitask_trainer = tester_class.load_from_checkpoint(
                        feature_model_checkpoint_path, verifier_args=None
                    )
                except (RuntimeError, TypeError):
                    multitask_trainer = backwards_compatible_load(
                        feature_model_checkpoint_path, tester_class)

                assert isinstance(multitask_trainer, tester_class)
                feature_model = multitask_trainer.model.feature_model  # type: ignore
                del multitask_trainer
                assert isinstance(feature_model, FeatureModel)

                test_model = AttributeExtractionTask(feature_model, aem_model, 0.01)
                tester = pl.Trainer(
                    max_epochs=n_epochs,
                    default_root_dir=results_dir,
                    gpus=1,
                    auto_select_gpus=True,
                    benchmark=True,
                    fast_dev_run=is_fast_dev_run)

                results = tester.test(test_model, datamodule=test_module)
                with open(os.path.join(results_dir, f"results_{lambda_value}.yaml"), "w") as outfile:
                    yaml.dump(results[0], outfile)

                del tester
                del test_model


def find_lambda_experiments(root_experiment_path: str) -> Dict[str, Tuple[Type, str]]:

    lambdas = sort_lambdas(get_lambdas(root_experiment_path))

    # Get model class form config and path to the last checkpoint for each lambda.
    experiments_for_lambdas: Dict[str, Tuple[Type, str]] = {}
    for lambda_value in lambdas:
        experiment_path = os.path.join(root_experiment_path, lambda_value)
        trainer_class = get_model_class_from_config(experiment_path)
        feature_model_checkpoint_path = find_last_epoch_path(experiment_path)
        experiments_for_lambdas[lambda_value] = (trainer_class, feature_model_checkpoint_path)

    return experiments_for_lambdas


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("experiment_path")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--fullbody", action="store_true")
    parser.add_argument("--n-epochs", type=int, default=30)
    args = parser.parse_args()
    main(args.experiment_path, args.batch_size, args.debug, args.fullbody, args.n_epochs)
