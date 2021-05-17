from pytorch_lightning.utilities.cli import LightningCLI

from pytorch_lightning.loggers import TestTubeLogger, CSVLogger

from houshou.trainers import MultitaskTrainer
from houshou.data import TripletsAttributeDataModule


cli = LightningCLI(
    MultitaskTrainer,
    TripletsAttributeDataModule,
    subclass_mode_model=True,
    subclass_mode_data=True,
    parser_kwargs={"default_config_files": ["configs/default_config.yaml"]},
    seed_everything_default=42,
)
