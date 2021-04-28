import os
from re import L
import sys

from sacred import Experiment
from sacred.observers import FileStorageObserver
from sacred import SETTINGS
from sacred.utils import apply_backspaces_and_linefeeds

import houshou.utils as utils
from houshou.losses import LOSS

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
    batch_size = 512
    dataset_attribute = "Male"


@ex.automain
def run(
    results_directory: str,
    lambda_value: float,
    epochs: int,
    batch_size: int,
    dataset_attribute: str,
    loss: LOSS,
    _run,
):
    # Create results dir and symlink to sacred.
    os.makedirs(results_directory)
    raise NotImplementedError
