from typing import Optional

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from houshou.metrics import BalancedAccuracy
from houshou.models import AttributeExtractionModel, FeatureModel
from torchmetrics.classification import F1, Accuracy, Precision, Recall
from torchmetrics.collections import MetricCollection
from torchmetrics.functional import stat_scores


class AttributeExtractionTask(pl.LightningModule):
    def __init__(
        self,
        feature_model: Optional[FeatureModel],
        attribute_model: Optional[AttributeExtractionModel],
        learning_rate: float,
        n_outputs: int = 2,
        freeze_feature_model: bool = True
    ) -> None:
        super().__init__()
        self.feature_model = feature_model if feature_model else FeatureModel()
        self.attribute_model = attribute_model if attribute_model else AttributeExtractionModel(n_outputs=n_outputs)
        self.learning_rate = learning_rate
        self.n_outputs = n_outputs

        # Freeze the feature model.
        if freeze_feature_model:
            self.feature_model.freeze()

        # Get the metrics.
        metrics = MetricCollection(
            {
                "Accuracy": Accuracy(num_classes=n_outputs),
                "BalancedAccuracy": BalancedAccuracy(num_classes=n_outputs),
                "Precison": Precision(num_classes=n_outputs),
                "Recall": Recall(num_classes=n_outputs),
                "F1": F1(num_classes=n_outputs),
            }
        )
        self.train_metrics = metrics.clone(prefix="train_")
        self.val_metrics = metrics.clone(prefix="val_")
        self.test_metrics = metrics.clone(prefix="test_")

    def forward(self, x):
        x = self.feature_model(x)

        if isinstance(x, torch.Tensor):
            x = x
        elif len(x) == 2:
            _, x = x
        else:
            raise ValueError

        x = self.attribute_model(x)
        return x

    def training_step(self, batch, batch_idx):
        x, (_, a) = batch
        a = a.squeeze()

        a_hat = self(x)
        loss = F.cross_entropy(a_hat, a)

        self.log_metrics(self.train_metrics, a_hat, a, loss, True)

        return loss

    def validation_step(self, batch, batch_idx):
        x, (_, a) = batch
        a = a.squeeze()

        a_hat = self(x)
        loss = F.cross_entropy(a_hat, a)

        self.log_metrics(self.val_metrics, a_hat, a, loss, False)

    def test_step(self, batch, batch_idx):
        x, (_, a) = batch
        a = a.squeeze()

        a_hat = self(x)
        loss = F.cross_entropy(a_hat, a)

        self.log_metrics(self.test_metrics, a_hat, a, loss, False)

    def log_metrics(
        self,
        metric_collection: MetricCollection,
        attribute_pred: torch.Tensor,
        attribute: torch.Tensor,
        loss: torch.Tensor,
        log_on_step: bool,
    ):

        prefix = metric_collection.prefix
        a_hat_softmax = F.softmax(attribute_pred, dim=1)
        metrics = metric_collection(a_hat_softmax, attribute)

        tp, fp, tn, fn, _ = stat_scores(
            a_hat_softmax,
            attribute,
            reduce="micro",
            num_classes=self.n_outputs,
            multiclass=False,
        )
        stats = {"tp": tp, "fp": fp, "tn": tn, "fn": fn}
        stats = {f"{prefix}{k}": stats[k] for k in stats}

        combined_metrics = metrics | stats

        self.log(f"{prefix}loss", loss, on_step=log_on_step, on_epoch=True)
        self.log_dict(combined_metrics, on_step=log_on_step, on_epoch=True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)
