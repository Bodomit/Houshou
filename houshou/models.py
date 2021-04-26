import torch
import torch.nn as nn
import torch.nn.functional as F

from facenet_pytorch import InceptionResnetV1


class FeatureModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.resnet = InceptionResnetV1(
            pretrained=None, classify=False, num_classes=None, dropout_prob=0.6
        )

    def forward(self, x):
        return self.resnet(x)


class AttributeExtractionModel(nn.Module):
    def __init__(self, n_outputs=2):
        super().__init__()
        self.full_model = nn.Sequential(
            nn.Flatten(),
            nn.LeakyReLU(),
            nn.Linear(512, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 32),
            nn.LeakyReLU(),
            nn.Linear(32, n_outputs),
        )

    def forward(self, x):
        return self.full_model(x)


class MultiTaskTrainingModel(nn.Module):
    def __init__(
        self,
        feature_model: nn.Module = None,
        attribute_model: nn.Module = None,
        reverse_attribute_gradient: bool = False,
    ):
        super().__init__()

        if feature_model:
            self.feature_model = feature_model
        else:
            self.feature_model = FeatureModel()

        if attribute_model:
            self.attribute_model = attribute_model
        else:
            self.attribute_model = AttributeExtractionModel()

        self.reverse_attribute_gradient = reverse_attribute_gradient

    def forward(self, x):
        features = self.feature_model(x)
        attribute = self.attribute_model(features)

        if self.reverse_attribute_gradient:
            attribute = GradReverse.apply(attribute)

        return features, attribute


class GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg(), None
