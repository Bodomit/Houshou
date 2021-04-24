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
        raise NotImplementedError
