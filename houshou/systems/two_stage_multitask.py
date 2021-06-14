from typing import Dict, Optional

import torch
import torch.nn.functional as F
from houshou.metrics import CVThresholdingVerifier

from .multitask_trainer import MultitaskTrainer


class TwoStageMultitaskTrainer(MultitaskTrainer):
    def __init__(
        self,
        loss_f: str,
        loss_a: str,
        lambda_value: float,
        learning_rate: float,
        verifier_args: Dict,
        weight_attributes: bool,
        classification_training_scenario: bool,
        n_classes: Optional[int] = None,
        **kwargs
    ) -> None:
        super().__init__(
            loss_f,
            loss_a,
            lambda_value,
            learning_rate,
            verifier_args,
            weight_attributes,
            classification_training_scenario,
            n_classes,
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
        self.model.feature_model.freeze()
        self.model.attribute_model.unfreeze()

        _, attribute_pred = self.model(xb)

        attribute_loss = F.cross_entropy(attribute_pred, ab, self.attribute_weights)
        self.log(
            "loss/stage1", attribute_loss, on_step=True, on_epoch=True, prog_bar=True
        )
        self.manual_backward(attribute_loss)
        opt.step()
        opt.zero_grad()

        # Train on same batch but feature model only.
        self.model.feature_model.unfreeze()
        self.model.attribute_model.freeze()

        embeddings_or_logits, attribute_pred = self.model(xb)

        # Log metrics.
        metrics = self.training_step_attribute_metrics(ab, attribute_pred)
        self.log_dict(metrics)

        # Backprop the multi-task loss.
        assert isinstance(embeddings_or_logits, torch.Tensor)
        total_loss, sub_losses = self.get_totalloss_with_sublosses(
            self.loss,
            yb,
            ab,
            embeddings_or_logits,
            attribute_pred,
            prefix="loss/stage2/",
        )
        self.log("loss/stage2/total", total_loss, on_step=True, on_epoch=True)
        self.log_dict(sub_losses, on_step=True, on_epoch=True, prog_bar=True)
        self.manual_backward(total_loss)
        opt.step()
        opt.zero_grad()

    def configure_optimizers(self):
        return (torch.optim.Adam(self.model.parameters(), lr=self.learning_rate),)
