import os
import unittest
import pytest

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from houshou.data import CelebA
from houshou.data import AttributeDataset
from houshou.data.samplers import TripletBatchRandomSampler

POSSIBLE_DATASET_ROOT_DIRS = ["/mnt/e/datasets", "~/datasets"]


def get_root_dir() -> str:
    for dir in POSSIBLE_DATASET_ROOT_DIRS:
        dir = os.path.abspath(os.path.expanduser(dir))
        if os.path.isdir(dir):
            return dir
    raise ValueError("No dataset directory found.")


@pytest.mark.local
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
            break

    def test_load_test(self):
        dataset = CelebA(self.root, "test")
        assert dataset is not None
        assert len(dataset) == 19944

        for image, target in dataset:
            assert image is not None
            assert len(target) == 2
            break


@pytest.mark.local
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
            break


@pytest.mark.local
class TripletFriendlyRandomSamplerTests(unittest.TestCase):
    def __init__(self, methodName: str) -> None:
        super().__init__(methodName=methodName)
        self.root = get_root_dir()

        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),  # Reads images scales to [0, 1]
                transforms.Lambda(lambda x: x * 2 - 1),  # Change range to [-1, 1]
                transforms.Resize((160, 160)),
                transforms.RandomHorizontalFlip(p=0.5),
            ]
        )

        dataset = CelebA(self.root, "train", transform=self.transform)
        self.dataset = AttributeDataset(dataset, ["Male"])

    def test_sampler_batchs_have_valid_triplets(self):
        sampler = TripletBatchRandomSampler(
            self.dataset, batch_size=32, drop_last=False, buffer_size=100
        )

        i = 0
        for i, batch in enumerate(sampler):
            identities = self.dataset.identity[batch]
            unique_identities, counts = identities.unique(return_counts=True)
            assert len(unique_identities) >= 2
            assert torch.any(counts >= 2)

        assert i == len(sampler) - 1

    def test_sampler_batchs_have_valid_triplets_drop_last(self):
        sampler = TripletBatchRandomSampler(
            self.dataset, batch_size=32, drop_last=True, buffer_size=100
        )

        i = 0
        for i, batch in enumerate(sampler):
            identities = self.dataset.identity[batch]
            unique_identities, counts = identities.unique(return_counts=True)
            assert len(unique_identities) >= 2
            assert torch.any(counts >= 2)

        assert i == len(sampler) - 1

    def test_sampler_batchs_have_valid_triplets_full(self):
        sampler = TripletBatchRandomSampler(
            self.dataset, batch_size=32, drop_last=True, buffer_size=100
        )

        for _, (yb, _) in DataLoader(self.dataset, batch_sampler=sampler):
            yb_unique, counts = yb.unique(return_counts=True)
            assert len(yb_unique) >= 2
            assert torch.any(counts >= 2)
            break
