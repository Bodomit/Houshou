import torch
import torch.nn.functional as F
import pytorch_lightning as pl

from torchmetrics.utilities.data import to_onehot
from torchmetrics.collections import MetricCollection
from torchmetrics.classification import Accuracy, Precision, Recall, F1
from torchmetrics.functional import stat_scores

from houshou.models import AttributeExtractionModel, FeatureModel


class AttributeExtractionTask(pl.LightningModule):
    def __init__(
        self, feature_model: FeatureModel, learning_rate: float, n_outputs=2
    ) -> None:
        super().__init__()
        self.feature_model = feature_model
        self.learning_rate = learning_rate
        self.n_outputs = n_outputs
        self.attribute_model = AttributeExtractionModel(n_outputs=n_outputs)

        # Freeze the feature model.
        self.feature_model.freeze()

        # Get the metrics.
        metrics = MetricCollection(
            {
                "Accuracy": Accuracy(num_classes=n_outputs),
                "BalancedAccuracy": Accuracy(num_classes=n_outputs, average="weighted"),
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
        x = self.attribute_model(x)
        return x

    def training_step(self, batch, batch_idx):
        x, (_, a) = batch
        a = a.squeeze()

        a_hat = self(x)
        loss = F.cross_entropy(a_hat, a)

        self.log_metrics(self.train_metrics, a_hat, a, loss)

        return loss

    def validation_step(self, batch, batch_idx):
        x, (_, a) = batch
        a = a.squeeze()

        a_hat = self(x)
        loss = F.cross_entropy(a_hat, a)

        self.log_metrics(self.val_metrics, a_hat, a, loss)

    def test_step(self, batch, batch_idx):
        x, (_, a) = batch
        a = a.squeeze()

        a_hat = self(x)
        loss = F.cross_entropy(a_hat, a)

        self.log_metrics(self.test_metrics, a_hat, a, loss)

    def log_metrics(
        self,
        metric_collection: MetricCollection,
        attribute_pred: torch.Tensor,
        attribute: torch.Tensor,
        loss: torch.Tensor,
    ):

        prefix = metric_collection.prefix
        a_hat_softmax = F.softmax(attribute_pred, dim=1)
        a_onehot = to_onehot(attribute)
        metrics = metric_collection(a_hat_softmax, a_onehot)

        tp, fp, tn, fn, _ = stat_scores(
            a_hat_softmax, a_onehot, reduce="micro", num_classes=self.n_outputs
        )
        stats = {"tp": tp, "fp": fp, "tn": tn, "fn": fn}
        stats = {f"{prefix}{k}": stats[k] for k in stats}

        combined_metrics = metrics | stats

        self.log("{prefix}loss", loss, on_step=True, on_epoch=True)
        self.log_dict(combined_metrics, on_step=True, on_epoch=True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)
