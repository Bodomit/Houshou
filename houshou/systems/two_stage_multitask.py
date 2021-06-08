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
        self.automatic_optimization = False

    def training_step(self, batch, batch_idx):
        opt = self.optimizers()
        assert isinstance(opt, torch.optim.Optimizer)
        opt.zero_grad()

        # Get the batch.
        xb, (yb, ab) = batch
        ab = ab.squeeze()

        # Train attribute model only.
        self.model.feature_model.requires_grad_(False)
        self.model.attribute_model.requires_grad_(True)

        embeddings, attribute_pred = self.model(xb)

        attribute_loss = F.cross_entropy(attribute_pred, ab, self.attribute_weights)
        self.log("loss/stage1", attribute_loss, on_step=True, on_epoch=True)
        self.manual_backward(attribute_loss)
        opt.step()
        opt.zero_grad()

        # Train on same batch but feature model only.
        self.model.feature_model.requires_grad_(True)
        self.model.attribute_model.requires_grad_(False)

        embeddings, attribute_pred = self.model(xb)

        # Log metrics.
        metrics = self.training_step_attribute_metrics(ab, attribute_pred)
        self.log_dict(metrics)

        # Backprop the multi-task loss.
        assert isinstance(embeddings, torch.Tensor)
        total_loss, sub_losses = self.get_totalloss_with_sublosses(
            self.loss, yb, ab, embeddings, attribute_pred, prefix="loss/stage2/"
        )
        self.log("loss/stage2/total", total_loss, on_step=True, on_epoch=True)
        self.log_dict(sub_losses, on_step=True, on_epoch=True)
        self.manual_backward(total_loss)
        opt.step()
        opt.zero_grad()

    def configure_optimizers(self):
        return (
            torch.optim.Adam(
                self.model.attribute_model.parameters(), lr=self.learning_rate
            ),
        )
