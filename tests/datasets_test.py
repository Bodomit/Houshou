import os
import unittest

import torch
from torch.utils.data.dataset import Dataset

from houshou.datasets import CelebA
from houshou.datasets.datasets import AttributeDataset

POSSIBLE_DATASET_ROOT_DIRS = ["/mnt/e/datasets"]


def get_root_dir() -> str:
    for dir in POSSIBLE_DATASET_ROOT_DIRS:
        if os.path.isdir(dir):
            return dir
    raise ValueError("No dataset directory found.")


class CelebADatasetTests(unittest.TestCase):
    def setUp(self):
        self.root = get_root_dir()

    def test_load_train(self):
        dataset = CelebA(self.root, "train")
        assert dataset is not None
        assert len(dataset) == 162484

        for image, target in dataset:
            assert image is not None
            assert len(target) == 2

    def test_load_test(self):
        dataset = CelebA(self.root, "test")
        assert dataset is not None
        assert len(dataset) == 19944

        for image, target in dataset:
            assert image is not None
            assert len(target) == 2


class AttributeDatasetTests(unittest.TestCase):
    def setUp(self):
        self.root = get_root_dir()

    def test_celeba_train_male_attribute(self):
        dataset: Dataset = CelebA(self.root, "train")
        dataset = AttributeDataset(dataset, ["Male"])

        for image, (identity, attributes) in dataset:
            assert image is not None
            assert identity.shape == torch.Size([])
            assert attributes.shape == torch.Size([1])
