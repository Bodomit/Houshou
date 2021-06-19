from pytorch_lightning.utilities.cli import LightningCLI

from houshou.data import TripletsAttributeDataModule
from houshou.systems import MultitaskTrainer


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
)
