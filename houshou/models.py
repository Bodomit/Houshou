from typing import Optional, Type

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from facenet_pytorch import InceptionResnetV1


class FeatureModel(pl.LightningModule):
    def __init__(
        self,
        dropout_prob: float = 0.6,
        use_resnet18=False,
        use_resnet101=False,
        use_pretrained=False,
        **kwargs
    ):
        super().__init__()
        self.use_pretrained = use_pretrained
        self.save_hyperparameters()

        assert not (use_resnet18 and use_resnet101)
        self.resnet = None

        if use_pretrained:
            resnet = InceptionResnetV1(
                pretrained="vggface2",
                classify=False,
                num_classes=None,
                dropout_prob=dropout_prob,
            )
        elif use_resnet18:
            assert use_pretrained is False
            resnet = torch.hub.load(
                "pytorch/vision:v0.9.0", "resnet18", pretrained=False
            )
        elif use_resnet101:
            assert use_pretrained is False
            resnet = torch.hub.load(
                "pytorch/vision:v0.9.0", "resnet101", pretrained=False
            )
        else:
            assert use_pretrained is False
            resnet = torch.hub.load(
                "pytorch/vision:v0.9.0", "resnet50", pretrained=False
            )

        if use_pretrained:
            self.resnet = resnet
            self.needs_normalisation = False
        else:
            self.needs_normalisation = True
            self.resnet = nn.Sequential(
                *(list(resnet.children())[:-1]),
                nn.Dropout(p=dropout_prob),
                nn.Flatten(),
                nn.Linear(2048, 512, bias=False),
                nn.BatchNorm1d(512, eps=0.001, momentum=0.1, affine=True)
            )

        self.feature_model = self.resnet

    def forward(self, x):
        x = self.feature_model(x)
        if self.needs_normalisation:
            return F.normalize(x, p=2, dim=1)
        else:
            return x


class ClassificationModel(FeatureModel):
    def __init__(self, n_classes: int, **kwargs) -> None:
        super().__init__(**kwargs)
        self.save_hyperparameters()
        self.n_classes = n_classes

        self.logits = nn.Linear(512, self.n_classes)

    def forward(self, x):
        features = self.feature_model(x)
        logits = self.logits(features)

        return logits, features


class AttributeExtractionModel(pl.LightningModule):
    def __init__(
        self, n_inputs=512, n_outputs=2, use_short_attribute_branch=False, **kwargs
    ):
        super().__init__()
        super().save_hyperparameters()
        if use_short_attribute_branch:
            self.full_model = nn.Sequential(
                nn.Flatten(),
                nn.LeakyReLU(),
                nn.Linear(n_inputs, 128),
                nn.LeakyReLU(),
                nn.Linear(128, 32),
                nn.LeakyReLU(),
                nn.Linear(32, n_outputs),
            )
        else:
            self.full_model = nn.Sequential(
                nn.Flatten(),
                nn.LeakyReLU(),
                nn.Linear(n_inputs, 128),
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
        use_resnet18: bool = False,
        use_resnet101: bool = False,
        use_short_attribute_branch: bool = False,
        use_pretrained: bool = False,
        **kwargs
    ):
        super().__init__()
        self.save_hyperparameters()

        if classification_training_scenario:
            model_type = ClassificationModel
        else:
            model_type = FeatureModel

        self.feature_model = self._load_or_create_model(
            model_type,
            feature_model_path,
            use_resnet18=use_resnet18,
            use_resnet101=use_resnet101,
            use_short_attribute_branch=use_short_attribute_branch,
            use_pretrained=use_pretrained,
            **kwargs
        )

        if use_resnet101:
            n_feature_outputs = 2048
        else:
            n_feature_outputs = 512

        if attribute_model_path:
            self.attribute_model = AttributeExtractionModel.load_from_checkpoint(
                attribute_model_path, n_inputs=n_feature_outputs
            )
        else:
            self.attribute_model = AttributeExtractionModel(
                use_short_attribute_branch=use_short_attribute_branch,
                n_inputs=n_feature_outputs,
            )

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
