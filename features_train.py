import os
import sys
from logging import warning

from functools import partial

from sacred import Experiment
from sacred.observers import FileStorageObserver
from sacred import SETTINGS
from sacred.utils import apply_backspaces_and_linefeeds

from torch.utils.data import DataLoader
from torchvision import transforms
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers

import houshou.utils as utils
from houshou.losses import LOSS, get_loss
from houshou.models import MultiTaskTrainingModel
from houshou.trainers import MultitaskTrainer
from houshou.data import DATASET, get_dataset, TripletBatchRandomSampler

ROOT_RESULTS_DIR = utils.parse_root_results_directory_argument(sys.argv[1::])
SACRED_DIR = os.path.join(ROOT_RESULTS_DIR, "sacred")

# Create experiment.
ex = Experiment("houshou")
ex.captured_out_filter = apply_backspaces_and_linefeeds  # type: ignore
filestorage_observer = FileStorageObserver(SACRED_DIR)
ex.observers.append(filestorage_observer)


@ex.named_config
def sh_multitask():
    loss = LOSS.SEMIHARD_CROSSENTROPY


@ex.config
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


def construct_train_dataloader(
    dataset,
    batch_size: int,
    num_workers: int,
    drop_last: bool,
    shuffle_buffer_size: int,
):
    return DataLoader(
        dataset,
        num_workers=num_workers,
        batch_sampler=TripletBatchRandomSampler(
            dataset, batch_size, drop_last, shuffle_buffer_size
        ),
        pin_memory=True,
    )


@ex.automain
def run(
    results_directory: str,
    dataset_root_directory: str,
    lambda_value: float,
    max_epochs: int,
    batch_size: int,
    dataset: DATASET,
    dataset_attribute: str,
    loss: LOSS,
    shuffle_buffer_size: int,
    drop_last_batch: bool,
):

    # Create the results directory: debug dir overwrites.
    create_results_dir(results_directory)

    # Data
    transform = transforms.Compose(
        [
            transforms.ToTensor(),  # Reads images scales to [0, 1]
            transforms.Lambda(lambda x: x * 2 - 1),  # Change range to [-1, 1]
            transforms.Resize((160, 160)),
            transforms.RandomHorizontalFlip(p=0.5),
        ]
    )

    get_dataset_ = partial(
        get_dataset,
        dataset,
        root=dataset_root_directory,
        selected_attributes=[dataset_attribute],
    )
    train_dataset = get_dataset_(split="train", transform=transform)
    valid_dataset = get_dataset_(split="valid", transform=transform)

    train_loader = construct_train_dataloader(
        train_dataset, batch_size, 4, drop_last_batch, shuffle_buffer_size
    )
    valid_loader = DataLoader(
        valid_dataset, batch_size=batch_size, num_workers=4, pin_memory=True
    )

    # Logger
    loggers = [
        pl_loggers.TestTubeLogger(os.path.join(results_directory, "tt_logs")),
        pl_loggers.CSVLogger(os.path.join(results_directory, "csv_logs")),
    ]

    # Loss
    loss_ = get_loss(loss)

    # Model
    model = MultiTaskTrainingModel()
    system = MultitaskTrainer(model, loss_, lambda_value)

    # Training
    trainer = pl.Trainer(
        gpus=1,
        max_epochs=max_epochs,
        logger=loggers,
        default_root_dir=results_directory,
    )
    trainer.fit(system, train_loader, valid_loader)

    raise NotImplementedError
