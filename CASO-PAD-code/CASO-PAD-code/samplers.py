import math
import random
from torch.utils.data import BatchSampler


class BalancedBatchSampler(BatchSampler):
    """
    Balanced mini-batches for ConcatDataset.

    Example

        datasets = [OULU, RA, CASIA]

        batch_size = 30

    →

        10
        10
        10

    every batch.
    """

    def __init__(
        self,
        concat_dataset,
        batch_size,
        shuffle=True,
        drop_last=False,
    ):

        self.concat_dataset = concat_dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last

        assert hasattr(
            concat_dataset,
            "datasets",
        ), "BalancedBatchSampler requires ConcatDataset."

        self.dataset_indices = []

        start = 0

        for ds in concat_dataset.datasets:

            end = start + len(ds)

            self.dataset_indices.append(
                list(range(start, end))
            )

            start = end

        self.num_domains = len(
            self.dataset_indices
        )

        base = batch_size // self.num_domains
        extra = batch_size % self.num_domains
        self.samples_per_domain = []

        for i in range(self.num_domains):
            self.samples_per_domain.append(
                base + (1 if i < extra else 0)
            )


    def __iter__(self):

        pools = []

        for inds in self.dataset_indices:

            inds = inds.copy()

            if self.shuffle:
                random.shuffle(inds)

            pools.append(inds)

        while True:

            batch = []

            finished = True

            for pool, n in zip(
                pools,
                self.samples_per_domain,
            ):

                if len(pool) >= n:

                    finished = False

                    batch.extend(pool[:n])

                    del pool[:n]

            if finished:
                break

            if self.shuffle:
                random.shuffle(batch)

            yield batch


    def __len__(self):

        num_batches = []

        for inds, n in zip(
            self.dataset_indices,
            self.samples_per_domain,
        ):

            num_batches.append(
                len(inds) // n
            )

        return min(num_batches)