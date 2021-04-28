from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class TripletMiner(nn.Module):
    def __init__(self, device=None):
        super().__init__()

        if device:
            self.device = device
        else:
            self.device = torch.device("cpu")

    def pairwise_distances(self, feature: torch.Tensor, p=2):

        assert len(feature.shape) == 2
        num_data = feature.shape[0]
        new_shape = (num_data * num_data, feature.shape[1])

        f1 = feature.unsqueeze(1).repeat(1, num_data, 1).reshape(new_shape)
        f2 = (
            feature.unsqueeze(1)
            .transpose(0, 1)
            .repeat(1, num_data, 1)
            .reshape(new_shape)
        )

        pairwise_distances = F.pairwise_distance(f1, f2, p=p, eps=1e-16)
        pairwise_distances = pairwise_distances.reshape((num_data, num_data))

        # Explicitly set diagonals to zero.
        mask_offdiagonals = (
            torch.logical_not(torch.diagflat(torch.ones([num_data])))
            .to(torch.float32)
            .to(self.device)
        )
        pairwise_distances = torch.multiply(pairwise_distances, mask_offdiagonals)

        return pairwise_distances

    def masked_minimum(self, data: torch.Tensor, mask: torch.Tensor, dim: int = 1):
        axis_maximums = torch.max(data, dim, keepdim=True).values
        masked_minimums = (
            torch.min(
                torch.multiply(torch.subtract(data, axis_maximums), mask),
                dim,
                keepdim=True,
            ).values
            + axis_maximums
        )

        return masked_minimums

    def masked_maximum(self, data: torch.Tensor, mask: torch.Tensor, dim: int = 1):
        axis_minimums = torch.min(data, dim, keepdim=True).values
        masked_maximums = (
            torch.max(
                torch.multiply(torch.subtract(data, axis_minimums), mask),
                dim,
                keepdim=True,
            ).values
            + axis_minimums
        )

        return masked_maximums


class SemiHardTripletMiner(TripletMiner):
    def __init__(
        self,
        margin: float = 1.0,
        lambda_value: float = 0.0,
        device=None,
        penalty_loss_name: str = "penalty",
        **kwargs
    ):
        super().__init__(device=device)
        self.margin = margin
        self.lambda_value = lambda_value
        self.penalty_loss_name = penalty_loss_name

    def forward(
        self,
        embeddings: torch.Tensor,
        attribute_pred: Optional[torch.Tensor],
        labels: torch.Tensor,
        attribute: Optional[torch.Tensor] = None,
    ):
        embeddings = embeddings.to(torch.float32)
        labels = torch.unsqueeze(labels, 1)

        # Get pairwise distances.
        pdist_matrix = self.pairwise_distances(embeddings)

        # Build pairwise binary adjacency matrix.
        adjacency = torch.eq(labels, torch.transpose(labels, 0, 1))
        adjacency_not = torch.logical_not(adjacency).to(torch.float32)
        adjacency = adjacency.to(torch.float32)

        batch_size = labels.size()[0]

        # Compute the mask.
        pdist_matrix_tile = pdist_matrix.repeat(batch_size, 1)
        mask = torch.logical_and(
            adjacency_not.repeat(batch_size, 1),
            torch.greater(
                pdist_matrix_tile,
                torch.reshape(torch.transpose(pdist_matrix, 0, 1), [-1, 1]),
            ),
        )
        mask_final = torch.reshape(
            torch.greater(torch.sum(mask.to(torch.float32), 1, keepdim=True), 0.0),
            [batch_size, batch_size],
        )
        mask_final = torch.transpose(mask_final, 0, 1)

        # Negatives Outside: smallest D_an where D_an > D_ap
        negatives_outside = torch.reshape(
            self.masked_minimum(pdist_matrix_tile, mask), (batch_size, batch_size)
        )
        negatives_outside = torch.transpose(negatives_outside, 0, 1)

        # Negatives Inside
        negatives_inside = self.masked_maximum(pdist_matrix, adjacency_not).repeat(
            1, batch_size
        )
        semi_hard_negatives = torch.where(
            mask_final, negatives_outside, negatives_inside
        )

        loss_mat = torch.add(
            self.margin, torch.subtract(pdist_matrix, semi_hard_negatives)
        )

        mask_positives = adjacency - torch.diagflat(torch.ones(batch_size)).to(
            self.device
        )

        num_positives = torch.sum(mask_positives)

        triplet_loss = torch.div(
            torch.sum(torch.clamp(torch.mul(loss_mat, mask_positives), min=0.0)),
            num_positives,
        )

        if self.lambda_value > 0.0:
            # Calculate the penalty to try and force similar images apart.
            penalty = self.calc_penalty(
                embeddings,
                attribute_pred,
                labels,
                attribute,
                pdist_matrix=pdist_matrix,
                adjacency_not=adjacency_not,
            )
            total_loss = (1 - self.lambda_value) * triplet_loss + (
                self.lambda_value * penalty
            )
        else:
            total_loss = triplet_loss
            penalty = torch.tensor(0)

        return total_loss, {"triplet": triplet_loss, self.penalty_loss_name: penalty}

    def calc_penalty(
        self,
        embeddings: torch.Tensor,
        attribute_pred: Optional[torch.Tensor],
        labels: torch.Tensor,
        attribute: Optional[torch.Tensor],
        **kwargs
    ):
        raise NotImplementedError


class SHTWithCategoricalCrossEntropy(SemiHardTripletMiner):
    def __init__(self, **kwargs):
        super().__init__(
            **kwargs,
            penalty_loss_name="attribute",
        )
        self.loss = nn.CrossEntropyLoss()

    def calc_penalty(
        self,
        embeddings: torch.Tensor,
        attribute_pred: Optional[torch.Tensor],
        labels: torch.Tensor,
        attribute: Optional[torch.Tensor],
        pdist_matrix: torch.Tensor = None,
        adjacency_not: torch.Tensor = None,
        **kwargs
    ) -> torch.Tensor:

        assert attribute is not None
        assert attribute_pred is not None

        loss = self.loss(attribute_pred, attribute)
        return loss
