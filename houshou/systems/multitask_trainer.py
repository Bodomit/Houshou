import os
import shutil

import torch
from torch.functional import Tensor
import torch.nn.functional as F
import pytorch_lightning as pl

from torch.utils.data.dataloader import DataLoader
from torchmetrics.classification import Accuracy, Precision, Recall, F1
from torchmetrics.classification import ConfusionMatrix
from torchmetrics.collections import MetricCollection

from houshou.losses import LOSS, get_loss
from houshou.metrics import BalancedAccuracy, CVThresholdingVerifier
from houshou.models import MultiTaskTrainingModel
from houshou.utils import save_cv_verification_results

from typing import Any, Tuple, Dict


class MultitaskTrainer(pl.LightningModule):
    def __init__(
        self,
        loss: str,
        lambda_value: float,
        learning_rate: float,
        verifier_args: Dict,
        weight_attributes: bool,
        **kwargs,
    ) -> None:
        super().__init__()
        self.save_hyperparameters("loss", "lambda_value", "learning_rate")

        self.model = MultiTaskTrainingModel(**kwargs)
        self.lambda_value = lambda_value
        self.learning_rate = learning_rate
        self.loss = get_loss(LOSS[loss], lambda_value=self.lambda_value)
        self.verifier = (
            CVThresholdingVerifier(**verifier_args) if verifier_args else None
        )
        self.weight_attributes = weight_attributes

        # Metrics
        metrics = MetricCollection(
            {
                "Accuracy": Accuracy(num_classes=2),
                "BalancedAccuracy": BalancedAccuracy(num_classes=2),
                "Precison": Precision(num_classes=2),
                "Recall": Recall(num_classes=2),
                "F1": F1(num_classes=2),
                "ConfusionMatrix": ConfusionMatrix(num_classes=2),
            }
        )
        self.train_metrics = metrics.clone(prefix="train_")
        self.valid_metrics = metrics.clone(prefix="valid_")

    def on_fit_start(self) -> None:
        # Create the log directory.
        dir = self.trainer.log_dir or self.trainer.default_root_dir
        if self.trainer.fast_dev_run:
            shutil.rmtree(dir, ignore_errors=True)
        os.makedirs(dir, exist_ok=True)

        # Calculate Attribute Weights
        dataloader = self.train_dataloader()
        assert isinstance(dataloader, DataLoader)
        support: torch.Tensor = dataloader.dataset.attributes_support  # type: ignore

        if self.weight_attributes:
            attribute_weights = 1 / support
            attribute_weights = (
                attribute_weights / attribute_weights.sum() * len(attribute_weights)
            )
            self.attribute_weights = attribute_weights
        else:
            self.attribute_weights = torch.ones_like(support, dtype=torch.float)
        self.attribute_weights = self.attribute_weights.to(self.device)

        # Set up the verification scenario tester.
        if self.verifier is not None:
            val_dataloader = self.val_dataloader()
            assert isinstance(val_dataloader, DataLoader)
            self.verifier.setup(val_dataloader)

    def training_step(self, batch, batch_idx):
        xb, (yb, ab) = batch
        ab = ab.squeeze()

        embeddings, attribute_pred = self.model(xb)
        total_loss, sub_losses = self.get_totalloss_with_sublosses(
            self.loss, yb, ab, embeddings, attribute_pred
        )

        self.log("loss", total_loss, on_step=True, on_epoch=True)
        self.log_dict(sub_losses, on_step=True, on_epoch=True)

        metrics = self.training_step_attribute_metrics(ab, attribute_pred)
        self.log_dict(metrics, on_step=True, on_epoch=True)

        return total_loss

    def training_step_attribute_metrics(
        self, attribute: torch.Tensor, attribute_pred: torch.Tensor
    ) -> Dict[str, Any]:
        softmaxed_attribute_pred = F.softmax(attribute_pred, dim=1)
        metrics = self.train_metrics(softmaxed_attribute_pred, attribute.squeeze())

        confusion_matrix = metrics["train_ConfusionMatrix"]
        del metrics["train_ConfusionMatrix"]

        confusion_matrix_dict = {
            "train_true_negative": confusion_matrix[0, 0],
            "train_false_positive": confusion_matrix[0, 1],
            "train_false_negative": confusion_matrix[1, 0],
            "train_true_positive": confusion_matrix[1, 1],
        }

        combine_metrics = metrics | confusion_matrix_dict
        return combine_metrics

    def validation_step(self, batch, batch_idx):
        xb, (yb, ab) = batch
        embeddings, attribute_pred = self.model(xb)
        total_loss, sub_losses = self.get_totalloss_with_sublosses(
            self.loss, yb, ab, embeddings, attribute_pred
        )

        self.log("valid_loss", total_loss)
        prefixed_sub_losses = {f"valid_{k}": sub_losses[k] for k in sub_losses}
        self.log_dict(prefixed_sub_losses)

        softmaxed_attribute_pred = F.softmax(attribute_pred, dim=1)
        metrics = self.valid_metrics(softmaxed_attribute_pred, ab.squeeze())

        confusion_matrix = metrics["valid_ConfusionMatrix"]
        del metrics["valid_ConfusionMatrix"]
        self.log_dict(metrics, on_step=True, on_epoch=True)

        confusion_matrix_dict = {
            "valid_true_negative": confusion_matrix[0, 0],
            "valid_false_positive": confusion_matrix[0, 1],
            "valid_false_negative": confusion_matrix[1, 0],
            "valid_true_positive": confusion_matrix[1, 1],
        }
        self.log_dict(confusion_matrix_dict)

    def validation_epoch_end(self, validation_step_outputs):
        assert isinstance(self.device, torch.device)

        if self.verifier is not None:

            try:
                auc, auc_per_attribute_pair = self.verifier.roc_auc(
                    self.model, self.device
                )
            except ValueError as ex:
                warning("Verification testing failed. Skipping: " + str(ex))
                return

            def newkey(attribute_pair: Tuple[int, int]):
                return f"valid_auc_{attribute_pair[0]}_{attribute_pair[1]}"

            labelled_apap = {
                newkey(k): auc_per_attribute_pair[k] for k in auc_per_attribute_pair
            }

            self.log("valid_auc", auc)
            self.log_dict(labelled_apap)

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer

    def get_totalloss_with_sublosses(
        self,
        loss_func,
        yb: torch.Tensor,
        ab: torch.Tensor,
        embeddings: torch.Tensor,
        pred_attribute: torch.Tensor,
        prefix: str = "loss/",
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:

        losses = loss_func(embeddings, pred_attribute, yb, ab, self.attribute_weights)

        if isinstance(losses, tuple):
            total_loss, sub_losses = losses
        else:
            total_loss = losses
            sub_losses = {}

        # Prefix "loss_" to each of the sublosses
        prefixed_sublosses = {f"{prefix}{k}": sub_losses[k] for k in sub_losses}
        return total_loss, prefixed_sublosses
