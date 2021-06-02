from typing import Dict
import torch
import torch.nn.functional as F

from houshou.metrics import CVThresholdingVerifier
from .multitask_trainer import MultitaskTrainer


class TwoStageMultitaskTrainer(MultitaskTrainer):
    def __init__(
        self,
        loss: str,
        lambda_value: float,
        learning_rate: float,
        verifier_args: Dict,
        weight_attributes: bool,
        **kwargs
    ) -> None:
        super().__init__(
            loss,
            lambda_value,
            learning_rate,
            verifier_args,
            weight_attributes,
            **kwargs
        )

    def training_step(self, batch, batch_idx, optimizer_idx):
        xb, (yb, ab) = batch
        ab = ab.squeeze()

        embeddings, attribute_pred = self.model(xb)

        # Log metrics.
        metrics = self.training_step_attribute_metrics(ab, attribute_pred)
        self.log_dict(metrics)

        # Train attribute model only.
        if optimizer_idx == 0:
            attribute_loss = F.cross_entropy(attribute_pred, ab)
            self.log("loss/stage1", attribute_loss, on_step=True, on_epoch=True)
            return attribute_loss

        # Train on feature model only.
        if optimizer_idx == 1:
            assert isinstance(embeddings, torch.Tensor)
            total_loss, sub_losses = self.get_totalloss_with_sublosses(
                self.loss, yb, ab, embeddings, attribute_pred, prefix="loss/stage2/"
            )
            self.log("loss/stage2/total", total_loss, on_step=True, on_epoch=True)
            self.log_dict(sub_losses, on_step=True, on_epoch=True)

            return total_loss

    def configure_optimizers(self):
        optimizers = [
            torch.optim.Adam(
                self.model.attribute_model.parameters(), lr=self.learning_rate
            ),
            torch.optim.Adam(
                self.model.feature_model.parameters(), lr=self.learning_rate
            ),
        ]

        return optimizers
