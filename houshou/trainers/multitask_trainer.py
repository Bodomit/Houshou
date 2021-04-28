import torch
import torch.nn as nn
import pytorch_lightning as pl

from houshou.losses import SemiHardTripletMiner

from typing import Any, List, Tuple, Dict


class MultitaskTrainer(pl.LightningModule):
    def __init__(
        self,
        model: nn.Module,
        loss: SemiHardTripletMiner,
        lambda_value: float,
    ) -> None:
        super().__init__()
        self.model = model
        self.loss = loss
        self.lambda_value = lambda_value

    def training_step(self, batch, batch_idx):
        xb, (yb, ab) = batch
        embeddings, attribute_pred = self.model(xb)
        total_loss, sub_losses = self.get_totalloss_with_sublosses(
            self.loss, yb, ab, embeddings, attribute_pred
        )

        self.log("loss", total_loss)
        self.log_dict(sub_losses)

        return total_loss

    def training_epoch_end(self, outputs: List[Any]) -> None:
        # raise NotImplementedError()
        pass

    def validation_step(self, *args, **kwargs):
        # raise NotImplementedError
        pass

    def validation_epoch_end(self, validation_step_outputs):
        # raise NotImplementedError
        pass

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    def train_dataloader(self) -> Any:
        return super().train_dataloader()

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

        return total_loss, sub_losses
