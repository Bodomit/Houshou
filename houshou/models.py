import torch
import torch.nn as nn

from facenet_pytorch import InceptionResnetV1

import pytorch_lightning as pl
from torch.nn.functional import dropout


class FeatureModel(pl.LightningModule):
    def __init__(self, dropout_prob: float = 0.6, **kwargs):
        super().__init__()
        self.save_hyperparameters()
        self.resnet = InceptionResnetV1(
            pretrained=None, classify=False, num_classes=None, dropout_prob=dropout_prob
        )

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("FeatureModel")
        parser.add_argument("--dropout-prob", type=float, default=0.6, metavar="FLOAT")
        return parent_parser

    def forward(self, x):
        return self.resnet(x)


class AttributeExtractionModel(pl.LightningModule):
    def __init__(self, n_outputs=2, **kwargs):
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

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("AttributeExtractionModel")
        parser.add_argument("--n-outputs", type=int, default=2, metavar="N1")
        return parent_parser

    def forward(self, x):
        return self.full_model(x)


class MultiTaskTrainingModel(pl.LightningModule):
    def __init__(
        self,
        feature_model_path: str = None,
        attribute_model_path: str = None,
        reverse_attribute_gradient: bool = False,
        **kwargs
    ):
        super().__init__()
        self.save_hyperparameters()

        if feature_model_path:
            self.feature_model = FeatureModel.load_from_checkpoint(feature_model_path)
        else:
            self.feature_model = FeatureModel(**kwargs)

        if attribute_model_path:
            self.attribute_model = AttributeExtractionModel.load_from_checkpoint(
                attribute_model_path
            )
        else:
            self.attribute_model = AttributeExtractionModel(**kwargs)

        self.reverse_attribute_gradient = reverse_attribute_gradient

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("MultiTaskTrainingModel")
        parser.add_argument(
            "--feature-model-path", type=str, default=None, metavar="PATH"
        )
        parser.add_argument(
            "--attribute-model-path", type=str, default=None, metavar="PATH"
        )
        parser.add_argument(
            "--reverse-attribute-gradient", type=bool, default=False, metavar="BOOL"
        )

        parser = FeatureModel.add_model_specific_args(parent_parser)
        parser = AttributeExtractionModel.add_model_specific_args(parent_parser)

        return parent_parser

    def forward(self, x):
        features = self.feature_model(x)

        if self.reverse_attribute_gradient:
            features = GradReverse.apply(features)

        attribute = self.attribute_model(features)
        return features, attribute


class FullAttributeExtractionModel(MultiTaskTrainingModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def forward(self, x):
        return super().forward(x)[1]


class GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg(), None
