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
        verifier: CVThresholdingVerifier,
        **kwargs
    ) -> None:
        super().__init__(loss, lambda_value, learning_rate, verifier, **kwargs)
        self.automatic_optimization = False

    def training_step(self, batch, batch_idx):
        opt = self.optimizers()
        assert isinstance(opt, torch.optim.Optimizer)
        opt.zero_grad()

        xb, (yb, ab) = batch
        ab = ab.squeeze()

        # Train attribute model only.
        self.model.feature_model.requires_grad_(False)
        self.model.attribute_model.requires_grad_(True)

        embeddings, attribute_pred = self.model(xb)

        attribute_loss = F.cross_entropy(attribute_pred, ab)
        self.manual_backward(attribute_loss)
        opt.step()
        opt.zero_grad()

        # Train on same batch but feature model only.
        self.model.feature_model.requires_grad_(True)
        self.model.attribute_model.requires_grad_(False)

        embeddings, attribute_pred = self.model(xb)

        assert isinstance(embeddings, torch.Tensor)
        total_loss, sub_losses = self.get_totalloss_with_sublosses(
            self.loss, yb, ab, embeddings, attribute_pred
        )

        self.manual_backward(total_loss)
        opt.step()

        sub_losses["stage1_attribute_loss"] = attribute_loss

        self.log("loss", total_loss, on_step=True, on_epoch=True)
        self.log_dict(sub_losses, on_step=True, on_epoch=True)

        metrics = self.training_step_attribute_metrics(ab, attribute_pred)
        self.log_dict(metrics)

        return total_loss
