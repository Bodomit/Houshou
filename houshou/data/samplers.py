import random

import torch
from torch.utils.data import Sampler, dataset

from typing import Iterator

from houshou.data.datasets import AttributeDataset


class TripletFriendlyRandomSampler(Sampler[int]):
    def __init__(self, data_source: AttributeDataset, buffer_size=1000) -> None:
        self.data_source = data_source
        self.buffer_size = buffer_size
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

        # Store indexes that will be sampled from.
        buffer = []
        real_idx = 0

        # Pre-populate the buffer.
        while len(buffer) < self.buffer_size:
            buffer.append(identity_map[real_idx])
            real_idx += 1

        # For each index requested, sample from the populated buffer.
        while True:
            try:
                # Randomly sample an index from the buffer
                buffer_idx = random.randint(0, len(buffer) - 1)
                dataset_idx = buffer[buffer_idx]
            except (ValueError, IndexError):
                # An error here means the buffer is empty, so stop iterating.
                return

            # Remove the index from the buffer and yield it.
            del buffer[buffer_idx]
            yield dataset_idx

            try:
                # Replenish the buffer.
                while len(buffer) < self.buffer_size:
                    buffer.append(identity_map[real_idx])
                    real_idx += 1
            except IndexError:
                # An index error here means there's no more data to fill the buffer.
                # Continute iterating while the buffer is exhausted.
                continue

    def __len__(self) -> int:
        return len(self.data_source)
