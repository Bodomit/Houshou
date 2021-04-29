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
from torchvision.transforms.transforms import Resize

import houshou.utils as utils
from houshou.losses import LOSS, get_loss
from houshou.models import MultiTaskTrainingModel
from houshou.trainers import MultitaskTrainer
from houshou.data import DATASET, get_dataset

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
    epochs = 50
    batch_size = 8
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


@ex.automain
def run(
    results_directory: str,
    dataset_root_directory: str,
    lambda_value: float,
    epochs: int,
    batch_size: int,
    dataset: DATASET,
    dataset_attribute: str,
    loss: LOSS,
    _run,
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

    train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=4)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, num_workers=4)

    # Loss
    loss_ = get_loss(loss)

    # Model
    model = MultiTaskTrainingModel()
    system = MultitaskTrainer(model, loss_, lambda_value)

    # Training
    trainer = pl.Trainer(gpus=1)
    trainer.fit(system, train_loader, valid_loader)

    raise NotImplementedError
