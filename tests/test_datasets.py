import os
import unittest
from typing import Tuple

import pytest
import torch
from houshou.data import CelebA, Market1501, VGGFace2
from houshou.data.samplers import TripletBatchRandomSampler
from PIL.Image import Image
from torch.utils.data import DataLoader

POSSIBLE_DATASET_ROOT_DIRS = ["/mnt/e/datasets", "~/datasets"]


def get_root_dir() -> str:
    for dir in POSSIBLE_DATASET_ROOT_DIRS:
        dir = os.path.abspath(os.path.expanduser(dir))
        if os.path.isdir(dir):
            return dir
    raise ValueError("No dataset directory found.")


@pytest.mark.local
class CelebATests(unittest.TestCase):
    def setUp(self):
        self.root = get_root_dir()
        self.data_dir = os.path.join(self.root, "CelebA_MTCNN")
        self.batch_size = 2
        self.dataset_module = CelebA(
            self.batch_size,
            1000,
            ["Male", "Bags_Under_Eyes"],
            self.data_dir,
        )
        self.dataset_module.setup(None)

    def test_load_train(self):
        assert self.dataset_module.train is not None
        assert len(self.dataset_module.train) == 162484

        for image, target in self.dataset_module.train:
            assert image is not None
            assert isinstance(image, torch.Tensor)
            assert len(target) == 2
            assert isinstance(target[0], torch.Tensor)
            assert isinstance(target[1], torch.Tensor)
            assert isinstance(target[0].item(), int)
            assert target[1].dtype == torch.int64
            assert len(target[1]) == 2 and len(target[1].shape) == 1
            break

    def test_load_test(self):
        assert self.dataset_module.test is not None
        assert len(self.dataset_module.test) == 19944

        for image, target in self.dataset_module.test:
            assert image is not None
            assert isinstance(image, torch.Tensor)
            assert len(target) == 2
            assert isinstance(target[0], torch.Tensor)
            assert isinstance(target[1], torch.Tensor)
            assert isinstance(target[0].item(), int)
            assert target[1].dtype == torch.int64
            assert len(target[1]) == 2 and len(target[1].shape) == 1
            break

    def test_calc_attributes_support(self):
        assert self.dataset_module.test.attributes_support.shape[0] == 4


@pytest.mark.local
class TripletFriendlyRandomSamplerTests(unittest.TestCase):
    def __init__(self, methodName: str) -> None:
        super().__init__(methodName=methodName)
        self.root = get_root_dir()
        self.data_dir = os.path.join(self.root, "CelebA_MTCNN")

        self.dataset_module = CelebA(8, 1000, ["Male"], self.data_dir)
        self.dataset_module.setup(None)
        self.identities = self.dataset_module.train.identities

    def test_sampler_batchs_have_valid_triplets(self):
        sampler = TripletBatchRandomSampler(
            self.identities, batch_size=32, drop_last=False, buffer_size=100
        )

        i = 0
        for i, batch in enumerate(sampler):
            identities = self.identities[batch]
            unique_identities, counts = identities.unique(return_counts=True)
            assert len(unique_identities) >= 2
            assert torch.any(counts >= 2)

        assert i == len(sampler) - 1

    def test_sampler_batchs_have_valid_triplets_drop_last(self):
        sampler = TripletBatchRandomSampler(
            self.identities, batch_size=32, drop_last=True, buffer_size=100
        )

        i = 0
        for i, batch in enumerate(sampler):
            identities = self.identities[batch]
            unique_identities, counts = identities.unique(return_counts=True)
            assert len(unique_identities) >= 2
            assert torch.any(counts >= 2)

        assert i == len(sampler) - 1

    def test_sampler_batchs_have_valid_triplets_full(self):
        sampler = TripletBatchRandomSampler(
            self.identities, batch_size=32, drop_last=True, buffer_size=100
        )

        for _, (yb, _) in DataLoader(self.dataset_module.train, batch_sampler=sampler):
            yb_unique, counts = yb.unique(return_counts=True)
            assert len(yb_unique) >= 2
            assert torch.any(counts >= 2)
            break


class VGGFace2Tests(unittest.TestCase):
    def setUp(self):
        self.root = get_root_dir()
        self.data_dir = os.path.join(self.root, "vggface2_MTCNN")
        self.batch_size = 2
        self.dataset_module = VGGFace2(
            self.batch_size,
            1000,
            ["Male", "Bangs"],
            self.data_dir,
        )
        self.dataset_module.setup(None)

    def test_get_val_set_classes(self):
        classes = set(["1", "2", "3", "4", "5", "6", "7", "8", "9", "10"])
        valid_classes = VGGFace2.get_val_set_classes(classes, 0.1, 42)
        assert len(valid_classes) == 1

    def test_get_val_set_classes_obery_seed(self):
        classes1 = set(["1", "2", "3", "4", "5", "6", "7", "8", "9", "10"])
        classes2 = set(["10", "9", "8", "7", "6", "5", "4", "3", "2", "1"])
        valid_classes1 = VGGFace2.get_val_set_classes(classes1, 0.1, 42)
        valid_classes2 = VGGFace2.get_val_set_classes(classes2, 0.1, 42)
        valid_classes3 = VGGFace2.get_val_set_classes(classes1, 0.1, 12345)
        assert valid_classes1 == valid_classes2
        assert valid_classes3 != valid_classes1

    @pytest.mark.local
    def test_load_train_val(self):
        trian_classes = set(s[1] for s in self.dataset_module.train.samples)
        valid_classes = set(s[1] for s in self.dataset_module.valid.samples)

        assert set.intersection(trian_classes, valid_classes) == set([])

        for image, target in self.dataset_module.train:
            assert image is not None
            assert isinstance(image, torch.Tensor)
            assert len(target) == 2
            assert isinstance(target[0], torch.Tensor)
            assert isinstance(target[1], torch.Tensor)
            assert isinstance(target[0].item(), int)
            assert target[1].dtype == torch.int64
            assert len(target[1]) == 2 and len(target[1].shape) == 1
            break

        for image, target in self.dataset_module.valid:
            assert image is not None
            assert isinstance(image, torch.Tensor)
            assert len(target) == 2
            assert isinstance(target[0], torch.Tensor)
            assert isinstance(target[1], torch.Tensor)
            assert isinstance(target[0].item(), int)
            assert target[1].dtype == torch.int64
            assert len(target[1]) == 2 and len(target[1].shape) == 1
            break

    @pytest.mark.local
    def test_load_test(self):
        assert self.dataset_module.test
        assert len(self.dataset_module.test) == 169157

        for image, target in self.dataset_module.test:
            assert image is not None
            assert isinstance(image, torch.Tensor)
            assert len(target) == 2
            assert isinstance(target[0], torch.Tensor)
            assert isinstance(target[1], torch.Tensor)
            assert isinstance(target[0].item(), int)
            assert target[1].dtype == torch.int64
            assert len(target[1]) == 2 and len(target[1].shape) == 1
            break


class Market1501Tests(unittest.TestCase):
    def setUp(self):
        self.root = get_root_dir()
        self.data_dir = os.path.join(self.root, "Market-1501")
        self.batch_size = 2
        self.dataset_module = Market1501(
            self.batch_size,
            1000,
            ["gender", "backpack"],
            self.data_dir,
        )
        self.dataset_module.setup(None)

    def test_load_test(self):
        assert self.dataset_module.test
        assert len(self.dataset_module.test) == 13115
        assert len(self.dataset_module.query) == 3368

        for image, target in self.dataset_module.test:
            assert image is not None
            assert isinstance(image, torch.Tensor)
            assert len(target) == 2
            assert isinstance(target[0], torch.Tensor)
            assert isinstance(target[1], torch.Tensor)
            assert isinstance(target[0].item(), int)
            assert target[1].dtype == torch.int64
            assert len(target[1]) == 2 and len(target[1].shape) == 1
            break

    def test_load_train_val(self):
        
        train_classes = set(self.dataset_module.train.identities)
        valid_classes = set(self.dataset_module.valid.identities)

        assert set.intersection(train_classes, valid_classes) == set([])

        for image, target in self.dataset_module.train:
            assert image is not None
            assert isinstance(image, torch.Tensor)
            assert len(target) == 2
            assert isinstance(target[0], torch.Tensor)
            assert isinstance(target[1], torch.Tensor)
            assert isinstance(target[0].item(), int)
            assert target[1].dtype == torch.int64
            assert len(target[1]) == 2 and len(target[1].shape) == 1
            break

        for image, target in self.dataset_module.valid:
            assert image is not None
            assert isinstance(image, torch.Tensor)
            assert len(target) == 2
            assert isinstance(target[0], torch.Tensor)
            assert isinstance(target[1], torch.Tensor)
            assert isinstance(target[0].item(), int)
            assert target[1].dtype == torch.int64
            assert len(target[1]) == 2 and len(target[1].shape) == 1
            break
