from pytorch_lightning.utilities.cli import LightningArgumentParser, LightningCLI

from houshou.trainers import MultitaskTrainer
from houshou.data import TripletsAttributeDataModule, DATASET, CelebA

cli = LightningCLI(
    MultitaskTrainer,
    TripletsAttributeDataModule,
    subclass_mode_data=True,
    parser_kwargs={"default_config_files": ["configs/default_config.yaml"]},
)
