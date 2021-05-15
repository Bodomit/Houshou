from torch.utils.data import DataLoader
import pytorch_lightning as pl

from .samplers import TripletBatchRandomSampler

from typing import List


class TripletsAttributeDataModule(pl.LightningDataModule):
    def __init__(self, batch_size: int, buffer_size: int):
        super().__init__()
        self.batch_size = batch_size
        self.buffer_size = buffer_size
        self.train = None
        self.valid = None
        self.test = None

    def train_dataloader(self) -> DataLoader:
        assert self.train
        train = DataLoader(
            self.train,
            batch_sampler=TripletBatchRandomSampler(
                self.train.identities, self.batch_size, True, self.buffer_size
            ),
        )
        return train

    def val_dataloader(self) -> DataLoader:
        assert self.valid
        valid = DataLoader(
            self.valid,
            batch_sampler=TripletBatchRandomSampler(
                self.valid.identities, self.batch_size, False, self.buffer_size
            ),
        )
        return valid

    def test_dataloader(self) -> DataLoader:
        assert self.test
        test = DataLoader(
            self.test,
            batch_sampler=TripletBatchRandomSampler(
                self.test.identities, self.batch_size, False, self.buffer_size
            ),
        )
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
