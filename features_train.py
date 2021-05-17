from pytorch_lightning.utilities.cli import LightningCLI

from pytorch_lightning.loggers import TestTubeLogger, CSVLogger

from houshou.trainers import MultitaskTrainer
from houshou.data import TripletsAttributeDataModule


cli = LightningCLI(
    MultitaskTrainer,
    TripletsAttributeDataModule,
    subclass_mode_data=True,
    parser_kwargs={"default_config_files": ["configs/default_config.yaml"]},
    # trainer_defaults={
    #     "logger": [
    #         TestTubeLogger(
    #             "/home/gbrown/results/houshou/tt",
    #             name="semihard_crossentropy_0.5",
    #             create_git_tag=True,
    #             log_graph=True,
    #         ),
    #         CSVLogger(
    #             "/home/gbrown/results/houshou/csv", name="semihard_crossentropy_0.5"
    #         ),
    #     ]
    # },
    seed_everything_default=42,
)
