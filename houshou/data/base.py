from torch.utils.data import DataLoader
import pytorch_lightning as pl

from .samplers import TripletBatchRandomSampler

from typing import List, Optional


class TripletsAttributeDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_dir: str,
        batch_size: int,
        buffer_size: int,
        attribute: List[str],
        target_type: List[str] = ["identity", "attr"],
        **kwargs,
    ):
        super().__init__()

        self.train = None
        self.valid = None
        self.test = None

        self.train_sampler = None
        self.valid_sampler = None
        self.test_sampler = None

        self.data_dir = data_dir
        self.batch_size = batch_size
        self.buffer_size = buffer_size
        self.attribute = attribute
        self.target_type = target_type

    @property
    def batch_size(self):
        return self._batch_size

    @batch_size.setter
    def batch_size(self, batch_size: int):
        self._batch_size = batch_size

        if self.train_sampler:
            self.train_sampler.batch_size = batch_size
        if self.valid_sampler:
            self.valid_sampler.batch_size = batch_size
        if self.test_sampler:
            self.test_sampler.batch_size = batch_size

    @staticmethod
    def add_data_specific_args(parent_parser):
        parent_parser.add_argument(
            "--data-dir", type=str, metavar="PATH", required=True
        )
        parent_parser.add_argument("--batch-size", type=int, default=256, metavar="N")
        parent_parser.add_argument("--buffer-size", type=int, default=1000, metavar="N")
        parent_parser.add_argument(
            "--attribute", "-a", action="append", required=True, metavar="STR"
        )
        return parent_parser

    def setup(self, stage: Optional[str]) -> None:
        super().setup(stage=stage)

        if stage is None or stage == "fit":
            assert self.train
            assert self.valid

            self.train_sampler = TripletBatchRandomSampler(
                self.train.identities, self.batch_size, True, self.buffer_size
            )

            self.valid_sampler = TripletBatchRandomSampler(
                self.valid.identities, self.batch_size, False, self.buffer_size
            )

        if stage is None or stage == "test":
            assert self.test

            self.test_sampler = TripletBatchRandomSampler(
                self.test.identities, self.batch_size, False, self.buffer_size
            )

    def train_dataloader(self) -> DataLoader:
        assert self.train
        train = DataLoader(self.train, batch_sampler=self.train_sampler, num_workers=4)
        return train

    def val_dataloader(self) -> DataLoader:
        assert self.valid
        valid = DataLoader(self.valid, batch_sampler=self.valid_sampler, num_workers=4)
        return valid

    def test_dataloader(self) -> DataLoader:
        assert self.test
        test = DataLoader(self.test, batch_sampler=self.test_sampler, num_workers=4)
        return test

    @staticmethod
    def get_indexes(attr_names: List[str], selected_attrs: List[str]):
        indexs: List[int] = []
        for selected_attr in selected_attrs:
            try:
                indexs.append(attr_names.index(selected_attr))
            except ValueError:
                raise ValueError(
                    f"Selected Attribute {selected_attr}"
                    + " not in attributes for dataset."
                )
        return indexs
