import torch
import torch.nn.functional as F
import pytorch_lightning as pl

from torch.utils.data.dataloader import DataLoader
from torchmetrics.classification import Accuracy, Precision, Recall, F1
from torchmetrics.classification import ConfusionMatrix
from torchmetrics.collections import MetricCollection

from houshou.losses import LOSS, get_loss
from houshou.metrics import CVThresholdingVerifier
from houshou.models import MultiTaskTrainingModel

from typing import Any, List, Optional, Tuple, Dict


class MultitaskTrainer(pl.LightningModule):
    def __init__(
        self,
        loss: str,
        lambda_value: float,
        learning_rate: float,
        verifier: CVThresholdingVerifier,
        **kwargs,
    ) -> None:
        super().__init__()
        self.save_hyperparameters("loss", "lambda_value", "learning_rate")

        self.model = MultiTaskTrainingModel(**kwargs)
        self.loss = get_loss(LOSS[loss])
        self.lambda_value = lambda_value
        self.learning_rate = learning_rate
        self.verifier = verifier

        # Metrics
        metrics = MetricCollection(
            {
                "Accuracy": Accuracy(num_classes=2),
                "BalancedAccuracy": Accuracy(num_classes=2, average="weighted"),
                "Precison": Precision(num_classes=2),
                "Recall": Recall(num_classes=2),
                "F1": F1(num_classes=2),
                "ConfusionMatrix": ConfusionMatrix(num_classes=2),
            }
        )
        self.train_metrics = metrics.clone(prefix="train_")
        self.valid_metrics = metrics.clone(prefix="valid_")

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("MultitaskTrainer")
        parser.add_argument("--loss", type=LOSS, required=True)
        parser.add_argument(
            "--lambda-value", type=float, metavar="FLOAT", required=True
        )
        parser.add_argument("--learning-rate", type=float, default=0.01)

        MultiTaskTrainingModel.add_model_specific_args(parent_parser)

        return parent_parser

    def on_fit_start(self) -> None:
        if self.verifier is not None:
            val_dataloader = self.val_dataloader()
            assert isinstance(val_dataloader, DataLoader)
            self.verifier.setup(val_dataloader)

    def training_step(self, batch, batch_idx):
        xb, (yb, ab) = batch
        embeddings, attribute_pred = self.model(xb)
        total_loss, sub_losses = self.get_totalloss_with_sublosses(
            self.loss, yb, ab, embeddings, attribute_pred
        )

        self.log("loss", total_loss, on_step=True, on_epoch=True)
        self.log_dict(sub_losses, on_step=True, on_epoch=True)

        softmaxed_attribute_pred = F.softmax(attribute_pred, dim=1)
        metrics = self.train_metrics(softmaxed_attribute_pred, ab.squeeze())

        confusion_matrix = metrics["train_ConfusionMatrix"]
        del metrics["train_ConfusionMatrix"]
        self.log_dict(metrics, on_step=True, on_epoch=True)

        confusion_matrix_dict = {
            "train_true_negative": confusion_matrix[0, 0],
            "train_false_positive": confusion_matrix[0, 1],
            "train_false_negative": confusion_matrix[1, 0],
            "train_true_positive": confusion_matrix[1, 1],
        }
        self.log_dict(confusion_matrix_dict)

        return total_loss

    def training_epoch_end(self, outputs: List[Any]) -> None:
        pass

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
            auc, auc_per_attribute_pair = self.verifier.roc_auc(self.model, self.device)

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
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:

        losses = loss_func(embeddings, pred_attribute, yb, ab)

        if isinstance(losses, tuple):
            total_loss, sub_losses = losses
        else:
            total_loss = losses
            sub_losses = {}

        # Prefix "loss_" to each of the sublosses
        prefixed_sublosses = {f"loss_{k}": sub_losses[k] for k in sub_losses}
        return total_loss, prefixed_sublosses
