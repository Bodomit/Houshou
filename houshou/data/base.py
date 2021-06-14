import os
import warnings
from typing import List, Optional, Set

import numpy as np
import pytorch_lightning as pl
import ruyaml as yaml
import torch
from torch.utils.data import DataLoader

from .samplers import TripletBatchRandomSampler


class TripletsAttributeDataModule(pl.LightningDataModule):

    MIN_BATCH_SIZE = 16

    def __init__(
        self,
        data_dir: str,
        batch_size: int,
        buffer_size: Optional[int],
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

        self.data_dir = self.parse_dataset_dir(data_dir)
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.attribute = attribute
        self.target_type = target_type

        self.num_workers = min(32, len(os.sched_getaffinity(0)))

    @property
    def batch_size(self):
        return self._batch_size

    @batch_size.setter
    def batch_size(self, batch_size: int):

        if batch_size < self.MIN_BATCH_SIZE and self.buffer_size is not None:
            msg = f"batch_size {batch_size} below minimum of {self.MIN_BATCH_SIZE}. "
            msg += f"Setting batch_size to {self.MIN_BATCH_SIZE}"
            warnings.warn(msg)
            batch_size = self.MIN_BATCH_SIZE

        self._batch_size = batch_size

        if self.train_sampler:
            self.train_sampler.batch_size = batch_size
        if self.valid_sampler:
            self.valid_sampler.batch_size = batch_size
        if self.test_sampler:
            self.test_sampler.batch_size = batch_size

    def parse_dataset_dir(self, dataset_dir: str) -> str:
        dataset_dir = os.path.expanduser(dataset_dir)

        if os.path.isabs(dataset_dir):
            return dataset_dir

        # Try reading the appconfig.yaml file to get the root directory.
        appconfig_path = os.path.abspath(
            os.path.join(__file__, "../../../appconfig.yaml")
        )
        try:
            with open(appconfig_path, "r") as infile:
                config = yaml.safe_load(infile)

                if "root_datasets_directory" in config:
                    return os.path.join(config["root_datasets_directory"], dataset_dir)
        except FileNotFoundError:
            warnings.warn(f"{appconfig_path} not found.")

        return dataset_dir

    def setup(self, stage: Optional[str]) -> None:
        super().setup(stage=stage)

        if self.buffer_size is None:
            return

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

    def _create_dataloader(self, dataset, batch_sampler, shuffle):
        if self.buffer_size is not None:
            return DataLoader(
                dataset, batch_sampler=batch_sampler, num_workers=self.num_workers
            )
        else:
            return DataLoader(
                dataset,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                shuffle=shuffle,
            )

    def train_dataloader(self) -> DataLoader:
        assert self.train
        train = self._create_dataloader(self.train, self.train_sampler, True)
        return train

    def val_dataloader(self) -> DataLoader:
        assert self.valid
        valid = self._create_dataloader(self.valid, self.valid_sampler, False)
        return valid

    def test_dataloader(self) -> DataLoader:
        assert self.test
        test = self._create_dataloader(self.test, self.test_sampler, False)
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

    @staticmethod
    def calc_attribute_support(attributes: torch.Tensor) -> torch.Tensor:
        _, c = attributes.unique(return_counts=True, dim=0)
        return c

    @staticmethod
    def get_val_set_classes(
        classes: Set[str], valid_split: float, valid_split_seed: int
    ) -> Set[str]:
        assert valid_split >= 0 and valid_split < 1.0
        n_valid_set_classes = int(len(classes) * valid_split)
        rng = np.random.default_rng(valid_split_seed)
        valid_classes = rng.choice(sorted(classes), size=n_valid_set_classes)
        return set(valid_classes)
