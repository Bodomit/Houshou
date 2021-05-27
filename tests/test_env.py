import unittest
import torch

import pytest


class EnvironmentTests(unittest.TestCase):
    def test_torch(self):
        x = torch.rand(5, 3)
        assert x is not None

    @pytest.mark.local
    def test_torch_gpu(self):
        device = torch.device("cuda:0")
        assert device is not None
        assert torch.cuda.is_available()  # type: ignore
