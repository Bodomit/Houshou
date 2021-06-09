from typing import Optional, Type

import pytorch_lightning as pl
import torch
import torch.nn as nn
from facenet_pytorch import InceptionResnetV1
from pytorch_lightning.core.lightning import LightningModule


class FeatureModel(pl.LightningModule):
    def __init__(self, dropout_prob: float = 0.6, **kwargs):
        super().__init__()
        self.save_hyperparameters()
        self.resnet = InceptionResnetV1(
            pretrained=None, classify=False, num_classes=None, dropout_prob=dropout_prob
        )

    def forward(self, x):
        return self.resnet(x)


class ClassificationModel(FeatureModel):
    def __init__(self, n_classes: int, dropout_prob: float = 0.6, **kwargs) -> None:
        super().__init__(dropout_prob=dropout_prob, **kwargs)
        self.save_hyperparameters()
        self.n_classes = n_classes

        self.logits = nn.Linear(512, self.n_classes)

    def forward(self, x):
        features = self.resnet(x)
        logits = self.logits(features)

        return logits, features


class AttributeExtractionModel(pl.LightningModule):
    def __init__(self, n_outputs=2, **kwargs):
        super().__init__()
        self.full_model = nn.Sequential(
            nn.Flatten(),
            nn.LeakyReLU(),
            nn.Linear(512, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 64),
            nn.LeakyReLU(),
            nn.Linear(64, 64),
            nn.LeakyReLU(),
            nn.Linear(64, 32),
            nn.LeakyReLU(),
            nn.Linear(32, 32),
            nn.LeakyReLU(),
            nn.Linear(32, 16),
            nn.LeakyReLU(),
            nn.Linear(16, 16),
            nn.LeakyReLU(),
            nn.Linear(16, n_outputs),
        )

    def forward(self, x):
        return self.full_model(x)


class MultiTaskTrainingModel(pl.LightningModule):
    def __init__(
        self,
        feature_model_path: str = None,
        attribute_model_path: str = None,
        reverse_attribute_gradient: bool = False,
        classification_training_scenario: bool = False,
        **kwargs
    ):
        super().__init__()
        self.save_hyperparameters()

        if classification_training_scenario:
            model_type = ClassificationModel
        else:
            model_type = FeatureModel

        self.feature_model = self._load_or_create_model(
            model_type, feature_model_path, **kwargs
        )

        if attribute_model_path:
            self.attribute_model = AttributeExtractionModel.load_from_checkpoint(
                attribute_model_path
            )
        else:
            self.attribute_model = AttributeExtractionModel(**kwargs)

        self.reverse_attribute_gradient = reverse_attribute_gradient

    def _load_or_create_model(
        self,
        model_type: Type[pl.LightningModule],
        checkpoint_path: Optional[str],
        **kwargs
    ) -> pl.LightningModule:
        if checkpoint_path:
            return model_type.load_from_checkpoint(checkpoint_path)
        else:
            return model_type(**kwargs)

    def forward(self, x):
        if isinstance(self.feature_model, ClassificationModel):
            logits, features = self.feature_model(x)
            x = logits
        else:
            features = self.feature_model(x)
            x = features

        if self.reverse_attribute_gradient:
            features = GradReverse.apply(features)

        attribute = self.attribute_model(features)
        return x, attribute


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
