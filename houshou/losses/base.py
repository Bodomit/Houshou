import abc

import torch
import torch.nn as nn


class LossBase(abc.ABC, nn.Module):
    @abc.abstractmethod
    def forward(
        self,
        embeddings: torch.Tensor,
        attribute_pred: torch.Tensor,
        labels: torch.Tensor,
        attribute: torch.Tensor,
        attribute_weights: torch.Tensor,
    ):
        pass


class WeightedMultitaskLoss(LossBase):
    def __init__(self, loss_f: LossBase, loss_a: LossBase, lambda_value: float):
        super().__init__()
        self.loss_f = loss_f
        self.loss_a = loss_a
        self.lambda_value = lambda_value

    def forward(
        self,
        embeddings_or_logits: torch.Tensor,
        attribute_pred: torch.Tensor,
        labels: torch.Tensor,
        attribute: torch.Tensor,
        attribute_weights: torch.Tensor,
    ):
        l_f = self.loss_f(
            embeddings_or_logits, attribute_pred, labels, attribute, attribute_weights
        )

        l_a = self.loss_a(
            embeddings_or_logits, attribute_pred, labels, attribute, attribute_weights
        )

        loss = (1 - self.lambda_value) * l_f + (self.lambda_value * l_a)

        return loss, {"l_f": l_f.item(), "l_a": l_a.item()}
