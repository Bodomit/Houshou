from torch.utils.data.dataloader import DataLoader
from houshou.data.base import TripletsAttributeDataModule
from houshou.data.celeba import CelebA
from houshou.data.vggface2 import VGGFace2
import os
from logging import warning

from argparse import ArgumentParser

import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.trainer import Trainer

from houshou.losses import LOSS, get_loss
from houshou.models import MultiTaskTrainingModel
from houshou.trainers import MultitaskTrainer
from houshou.data import DATASET, get_dataset_module
from houshou.metrics import CVThresholdingVerifier

# Seed everything for reproducability.
pl.seed_everything(42, workers=True)


def sh_multitask():
    loss = LOSS.SEMIHARD_CROSSENTROPY


def default_config():
    max_epochs = 50
    batch_size = 256
    drop_last_batch = True
    shuffle_buffer_size = 1000
    dataset = DATASET.CELEBA
    dataset_attribute = "Male"


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

    # Training
    trainer = pl.Trainer.from_argparse_args(args)
    trainer.tune(system, datamodule=datamodule)
    trainer.fit(system, datamodule=datamodule)


if __name__ == "__main__":
    parser = ArgumentParser()

    # PROGRAM level args.

    # MODEL specific args.
    parser = MultitaskTrainer.add_model_specific_args(parser)

    # DATASET specific args.
    datasetparser = parser.add_argument_group("Dataset")
    datasetparser.add_argument(
        "--dataset", type=lambda d: DATASET[d], choices=list(DATASET), required=True
    )
    datasetparser = TripletsAttributeDataModule.add_data_specific_args(datasetparser)
    parser = VGGFace2.add_data_specific_args(parser)
    parser = CelebA.add_data_specific_args(parser)

    # TRAINER args.
    parser = Trainer.add_argparse_args(parser)

    # Parse the args.
    args = parser.parse_args()

    main(args)
