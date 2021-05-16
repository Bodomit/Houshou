import os
from logging import warning

# from jsonargparse import ArgumentParser, ActionConfigFile
from argparse import ArgumentParser

import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.trainer import Trainer

from houshou.trainers import MultitaskTrainer
from houshou.data import DATASET, get_dataset_module
from houshou.data import TripletsAttributeDataModule, CelebA, VGGFace2
from houshou.metrics import CVThresholdingVerifier

# Seed everything for reproducability.
pl.seed_everything(42, workers=True)


def create_results_dir(results_directory: str) -> None:
    # Create results dir and symlink to sacred.
    try:
        os.makedirs(results_directory)
    except FileExistsError:
        if "debug" in results_directory:
            warning("Debug directory detected: overwriting.")
        else:
            raise


def main(args):
    dict_args = vars(args)

    # Create the results directory: debug dir overwrites.
    # create_results_dir(args.results_directory)

    # Logger
    # loggers = [
    #    pl_loggers.TestTubeLogger(os.path.join(args.results_directory, "tt_logs")),
    #    pl_loggers.CSVLogger(os.path.join(args.results_directory, "csv_logs")),
    # ]

    # Get the datamodule.
    datamodule = get_dataset_module(**dict_args)

    # Verifier.
    verifier = None  # CVThresholdingVerifier(datamodule.valid, args.batch_size)

    # Model
    system = MultitaskTrainer(verifier=verifier, **dict_args)

    # Callbacks

    # Training
    trainer = pl.Trainer.from_argparse_args(args)

    datamodule.num_workers = 0
    trainer.fit(system, datamodule=datamodule)


if __name__ == "__main__":
    parentparser = ArgumentParser()
    # parentparser.add_argument("--cfg", action=ActionConfigFile)

    # PROGRAM level args.
    programparser = parentparser.add_argument_group("Program")

    # MODEL specific args.
    parentparser = MultitaskTrainer.add_model_specific_args(parentparser)

    # DATASET specific args.
    datasetparser = parentparser.add_argument_group("Dataset")
    datasetparser.add_argument("--dataset", type=DATASET, required=True)
    datasetparser = TripletsAttributeDataModule.add_data_specific_args(datasetparser)
    parentparser = VGGFace2.add_data_specific_args(parentparser)
    parentparser = CelebA.add_data_specific_args(parentparser)

    # TRAINER args.
    parentparser = Trainer.add_argparse_args(parentparser)

    # Parse the args.
    args = parentparser.parse_args()

    main(args)
