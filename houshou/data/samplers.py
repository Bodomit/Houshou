from enum import unique
import torch
from torch.utils.data import Sampler

from typing import Iterator

from houshou.data.datasets import AttributeDataset


class TripletFriendlyRandomSampler(Sampler[int]):
    def __init__(self, data_source: AttributeDataset) -> None:
        self.data_source = data_source
        assert torch.all(data_source.identity == data_source.identity.sort()[0])

    def __iter__(self) -> Iterator[int]:
        identity = self.data_source.identity
        id_unique, id_inverse = identity.unique(return_inverse=True)
        id_inverse = id_inverse.squeeze()

        # Map each sample's identity to another random integer (per identity) and sort.
        id_unique_perm = id_unique[torch.randperm(len(id_unique))]
        identity_with_mapped_ids = id_unique_perm[id_inverse]
        identity_map = identity_with_mapped_ids.argsort()
        identity_perm = identity[identity_map]

        # Ensure that only the order of the samples have changed.
        assert torch.all(
            identity.unique(return_counts=True)[1]
            == identity_perm.unique(return_counts=True)[1]
        )

        return iter(identity_map[i] for i in range(len(self.data_source)))

    def __len__(self) -> int:
        return len(self.data_source)
