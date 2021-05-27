import os
import unittest

from PIL import Image
from facenet_pytorch import MTCNN
import torch

from houshou.models import FullAttributeExtractionModel, MultiTaskTrainingModel


class MultiTaskTrainingModelTests(unittest.TestCase):
    def __init__(self, methodName: str) -> None:
        super().__init__(methodName=methodName)
        current_dir = os.path.dirname(os.path.realpath(__file__))
        image_path = os.path.join(current_dir, "resources", "0001_01.jpg")
        test_image_pil = Image.open(image_path)
        mtcnn = MTCNN(image_size=160, post_process=False)
        self.test_image = mtcnn(test_image_pil).unsqueeze(0)

    def test_forward_no_exception(self) -> None:
        model = MultiTaskTrainingModel()
        model.eval()
        with torch.no_grad():
            features, attribute = model(self.test_image)
            assert features is not None
            assert features.shape == torch.Size([1, 512])
            assert attribute is not None
            assert attribute.shape == torch.Size([1, 2])


class FullAttributeExtractionModelTests(unittest.TestCase):
    def __init__(self, methodName: str) -> None:
        super().__init__(methodName=methodName)
        current_dir = os.path.dirname(os.path.realpath(__file__))
        image_path = os.path.join(current_dir, "resources", "0001_01.jpg")
        test_image_pil = Image.open(image_path)
        mtcnn = MTCNN(image_size=160, post_process=False)
        self.test_image = mtcnn(test_image_pil).unsqueeze(0)

    def test_forward_no_exception(self) -> None:
        model = FullAttributeExtractionModel()
        model.eval()
        with torch.no_grad():
            attribute = model(self.test_image)
            assert attribute is not None
            assert attribute.shape == torch.Size([1, 2])
