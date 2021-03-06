from pytorch_lightning.utilities.cli import LightningCLI

from houshou.data import TripletsAttributeDataModule
from houshou.systems import MultitaskTrainer

import torch
torch.multiprocessing.set_sharing_strategy('file_system')

class CustomLightningCLI(LightningCLI):
    def add_arguments_to_parser(self, parser) -> None:
        parser.add_argument("--model.init_args.lambda_value", default=-1, type=float, required=True)


cli = CustomLightningCLI(
    MultitaskTrainer,
    TripletsAttributeDataModule,
    subclass_mode_model=True,
    subclass_mode_data=True,
    parser_kwargs={"default_config_files": ["configs/default_config.yaml"]},
    seed_everything_default=42,
    save_config_overwrite=True
)
