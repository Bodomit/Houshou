import warnings
import torch
from torch.utils.data import Sampler

from typing import Iterator, List


class TripletBatchRandomSampler(Sampler[List[int]]):
    def __init__(
        self,
        identities: torch.Tensor,
        batch_size: int,
        drop_last: bool,
        buffer_size: int,
        max_batch_retry: int = 100,
    ) -> None:
        self.identities = identities.squeeze()
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.buffer_size = buffer_size
        self.max_batch_retry = max_batch_retry
        assert torch.all(identities == identities.sort()[0])

    def __iter__(self) -> Iterator[List[int]]:
        id_unique, id_inverse = self.identities.unique(return_inverse=True)
        id_inverse = id_inverse.squeeze()

        # Map each sample's identity to another random integer (per identity) and sort.
        id_unique_perm = id_unique[torch.randperm(len(id_unique))]
        identity_with_mapped_ids = id_unique_perm[id_inverse]
        identity_map = identity_with_mapped_ids.argsort()
        identity_perm = self.identities[identity_map]

        # Ensure that only the order of the samples have changed.
        assert torch.all(
            self.identities.unique(return_counts=True)[1]
            == identity_perm.unique(return_counts=True)[1]
        )

        # Store indexes that will be sampled from.
        buffer = []
        real_idx = 0

        # Pre-populate the buffer.
        while len(buffer) < self.buffer_size:
            buffer.append(identity_map[real_idx])
            real_idx += 1

        # For each batch requested, sample batch_size indexes from the populated buffer.
        while True:
            buffer_idxs = None
            dataset_idxs = None

            # Randomly sample the buffer until a valid batch with triplets is found.
            valid_batch = False
            batch_retry_attempts = 0
            while not valid_batch:

                # Randomly sample indexs from the buffer, convert to dataset indexs.
                buffer_idxs = torch.randperm(len(buffer))[: self.batch_size]
                dataset_idxs = torch.tensor(
                    [buffer[b_idx] for b_idx in buffer_idxs], dtype=torch.long
                )

                # If batch is empty, exit the iterator.
                if dataset_idxs.shape[0] == 0:
                    return

                # Validate the batch.
                batch_identities = self.identities[dataset_idxs]
                unique_batch_identities, batch_counts = batch_identities.unique(
                    return_counts=True
                )
                valid_batch = len(unique_batch_identities) > 2 and torch.any(
                    batch_counts > 2
                )

                # If sampling is stuck and can't produce a valid batch, exit early.
                if batch_retry_attempts < self.max_batch_retry:
                    batch_retry_attempts += 1
                else:
                    warning_str = (
                        "Max retries for triplet batch validation reached: "
                        + str(self.max_batch_retry)
                        + ". Stopping Sampling early..."
                    )
                    warnings(warning_str)
                    return

            # Remove the indexs from the buffer.
            assert all([b_idx < len(buffer) for b_idx in buffer_idxs])
            for b_idx in sorted(buffer_idxs, reverse=True):  # type: ignore
                del buffer[b_idx]

            # Yield the batch (or stop iterating if the end is reached).
            if dataset_idxs.size()[0] == self.batch_size:
                yield dataset_idxs.tolist()
            elif self.drop_last is False:
                yield dataset_idxs.tolist()
            else:
                return

            try:
                # Replenish the buffer.
                while len(buffer) < self.buffer_size:
                    buffer.append(identity_map[real_idx])
                    real_idx += 1
            except IndexError:
                # An index error here means there's no more data to fill the buffer.
                # Continute iterating while the buffer is exhausted.
                continue

    def __len__(self):
        if self.drop_last:
            return len(self.identities) // self.batch_size  # type: ignore
        else:
            return (
                len(self.identities) + self.batch_size - 1
            ) // self.batch_size  # type: ignore
