import os
import unittest

from houshou.datasets import CelebA

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
