import torch
import torch.nn.functional as F

from .semihard_triplet_miner import SemiHardTripletMiner

from typing import Optional


class SHM_UniformKLDivergence(SemiHardTripletMiner):
    def __init__(self, lambda_value: float = 0.5, margin: float = 0.1, **kargs):
        super().__init__(lambda_value=lambda_value, margin=margin)
        self.target_distribution = None

    def calc_target_distribution(self, n_classes: int, device: torch.device):
        assert n_classes > 1
        avg_probability = 1 / n_classes
        probabilities = torch.tensor([avg_probability], device=device).repeat(n_classes)
        return probabilities

    def calc_penalty(  # type: ignore
        self,
        embeddings: torch.Tensor,
        attribute_pred: torch.Tensor,
        labels: torch.Tensor,
        attribute: Optional[torch.Tensor],
        pdist_matrix: torch.Tensor = None,
        adjacency_not: torch.Tensor = None,
        **kwargs
    ) -> torch.Tensor:

        if self.target_distribution is None:
            n_attribute_classes = attribute_pred.shape[1]
            self.target_distribution = self.calc_target_distribution(
                n_attribute_classes, attribute_pred.device
            )

        expanded_target = self.target_distribution.expand_as(attribute_pred)
        pred_log_softmax = F.log_softmax(attribute_pred, dim=1)
        loss = F.kl_div(pred_log_softmax, expanded_target, reduction="batchmean")
        loss = loss.clamp(min=0.0)
        return loss
