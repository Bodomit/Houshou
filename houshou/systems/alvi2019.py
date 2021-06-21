from typing import Dict, Optional

import torch
import torch.nn.functional as F
from houshou.models import ClassificationModel

from .multitask_trainer import MultitaskTrainer


class Alvi2019(MultitaskTrainer):
    def __init__(
        self,
        loss_f: str,
        loss_a: str,
        lambda_value: float,
        learning_rate: float,
        verifier_args: Dict,
        weight_attributes: bool,
        classification_training_scenario: bool = True,
        n_classes: int = None,
        use_resnet18: bool = False,
        use_short_attribute_branch: bool = False,
        reverse_attribute_gradient: bool = False,
        grads_near_zero_threshold: float = 2e-11,
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
            use_resnet18,
            use_short_attribute_branch,
            reverse_attribute_gradient,
            **kwargs
        )
        self.grads_near_zero_threshold = grads_near_zero_threshold
        self.grads_are_near_zero = False
        assert isinstance(self.model.feature_model, ClassificationModel)

        self.automatic_optimization = False

    def attribute_grads_near_zero(self) -> bool:
        grad_total = 0.0
        for param in self.model.attribute_model.parameters():
            grad_total += param.grad.abs().sum()

        return grad_total < self.grads_near_zero_threshold

    def on_epoch_start(self) -> None:
        super().on_epoch_start()
        self.grads_are_near_zero = False

    def on_train_batch_start(self, batch, batch_idx: int, dataloader_idx: int) -> Optional[int]:
        super().on_train_batch_start(batch, batch_idx, dataloader_idx)
        # Stops the epoch early if on an even (Ls training) epoch and grads
        # are near zero.
        if self.current_epoch % 2 == 0 and self.grads_are_near_zero:
            return -1

    def training_step_Ls(self, batch, batch_idx):
        opt = self.optimizers()
        assert isinstance(opt, torch.optim.Optimizer)
        opt.zero_grad()

        # Get the batch.
        xb, (_, ab) = batch
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

        self.grads_are_near_zero = self.attribute_grads_near_zero()

        opt.step()
        opt.zero_grad()

    def training_step_Lp_Lconf(self, batch, batch_idx):
        opt = self.optimizers()
        assert isinstance(opt, torch.optim.Optimizer)
        opt.zero_grad()

        # Get the batch.
        xb, (yb, ab) = batch
        ab = ab.squeeze()

        # Train on feature model only.
        self.model.feature_model.unfreeze()
        self.model.attribute_model.freeze()

        logits, attribute_pred = self.model(xb)

        # Log metrics.
        metrics = self.training_step_attribute_metrics(ab, attribute_pred)
        self.log_dict(metrics)

        # Backprop the multi-task loss.
        assert isinstance(logits, torch.Tensor)
        total_loss, sub_losses = self.get_totalloss_with_sublosses(
            self.loss,
            yb,
            ab,
            logits,
            attribute_pred,
            prefix="loss/stage2/",
        )
        self.log("loss/stage2/total", total_loss, on_step=True, on_epoch=True)
        self.log_dict(sub_losses, on_step=True, on_epoch=True, prog_bar=True)
        self.manual_backward(total_loss)
        opt.step()
        opt.zero_grad()

    def training_step(self, batch, batch_idx):
        # If on an even Epoch (starting at 0), train Ls.
        # If on an odd Epoch (starting at 1), train combined loss.
        if self.current_epoch % 2 == 0:
            return self.training_step_Ls(batch, batch_idx)
        else:
            return self.training_step_Lp_Lconf(batch, batch_idx)

    def configure_optimizers(self):
        return (torch.optim.SGD(self.model.parameters(), lr=self.learning_rate),)
