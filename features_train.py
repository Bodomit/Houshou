from houshou.data import TripletsAttributeDataModule
from houshou.systems import TwoStageMultitaskTrainer
from pytorch_lightning.utilities.cli import LightningCLI

cli = LightningCLI(
    TwoStageMultitaskTrainer,
    TripletsAttributeDataModule,
    subclass_mode_model=False,
    subclass_mode_data=True,
    parser_kwargs={"default_config_files": ["configs/default_config.yaml"]},
    seed_everything_default=42,
)
