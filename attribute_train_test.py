import os
from argparse import ArgumentParser

import pytorch_lightning as pl

from houshou.models import FeatureModel
from houshou.systems import TwoStageMultitaskTrainer, AttributeExtractionTask
from houshou.data import CelebA, VGGFace2
from houshou.utils import find_last_epoch_path

pl.seed_everything(42, workers=True)


def main(experiment_path: str, batch_size: int, is_fast_dev_run: bool):
    # Get the path to the last checkpoint.
    feature_model_checkpoint_path = find_last_epoch_path(experiment_path)

    print("Experiment Directory: ", experiment_path)
    print("Checkpoint Path; ", feature_model_checkpoint_path)

    # Load the multitask model and get the featrue model.
    multitask_trainer = TwoStageMultitaskTrainer.load_from_checkpoint(
        feature_model_checkpoint_path, verifier_args=None
    )
    assert isinstance(multitask_trainer, TwoStageMultitaskTrainer)
    feature_model = multitask_trainer.model.feature_model
    del multitask_trainer
    assert isinstance(feature_model, FeatureModel)
    feature_model.freeze()

    # Construct the datamodules. This will take some time.
    datamodules = [
        CelebA(batch_size, None, ["Male"]),
        VGGFace2(batch_size, None, ["Male"]),
    ]

    for train_module in datamodules:

        # Get directory for this test.
        root_dir = os.path.join(
            experiment_path, "aems", os.path.basename(train_module.data_dir)
        )

        # Train the attribute extraction model.
        aem_task = AttributeExtractionTask(feature_model, 0.001)
        trainer = pl.Trainer(
            max_epochs=50,
            default_root_dir=root_dir,
            gpus=1,
            auto_select_gpus=True,
            benchmark=True,
            fast_dev_run=is_fast_dev_run,
        )
        trainer.fit(aem_task, train_module)
        aem_task.freeze()
        del trainer

        # Run the tests. A new Trainer object is created each time
        # in order to have a different root_log_dir. Also, test()
        # doesn't support multiple datamodules yet.
        for test_module in datamodules:
            root_dir_test = os.path.join(
                root_dir, os.path.basename(test_module.data_dir)
            )
            tester = pl.Trainer(
                default_root_dir=root_dir_test,
                gpus=1,
                auto_select_gpus=True,
                benchmark=True,
                fast_dev_run=is_fast_dev_run,
            )
            tester.test(aem_task, datamodule=test_module)
            del tester

        del aem_task


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("experiment_path")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    main(args.experiment_path, args.batch_size, args.debug)
