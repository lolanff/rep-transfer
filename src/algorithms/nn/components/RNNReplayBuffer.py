import numpy as np
from abc import abstractmethod
from typing import Any
from ReplayTables._utils.logger import logger
from ReplayTables.interface import Timestep, LaggedTimestep, Batch, Item
from ReplayTables.ingress.IndexMapper import IndexMapper
from ReplayTables.ingress.CircularMapper import CircularMapper
from ReplayTables.ingress.LagBuffer import LagBuffer
from ReplayTables.sampling.IndexSampler import IndexSampler
from ReplayTables.sampling.UniformSampler import UniformSampler
from ReplayTables.storage.BasicStorage import BasicStorage
from ReplayTables.storage.Storage import Storage
from ReplayTables.ReplayBuffer import ReplayBuffer

class RNNReplayBuffer(ReplayBuffer):
    def __init__(
            self,
            max_size: int,
            lag: int,
            rng: np.random.Generator,
            sequence_length: int,
            idx_mapper: IndexMapper | None = None,
            storage: Storage | None = None,
            sampler: IndexSampler | None = None,
    ):
        super().__init__(max_size, lag, rng, idx_mapper=idx_mapper, storage=storage, sampler=sampler)
        self.sequence_length = sequence_length
    # TODO: it is to be noted that it does not handle the discrepency in transition as new experience overwrite old ones in the circular buffer, that is, it does not recognize the boundary of the latest frame vs the next frame in order of idx who is the old one
    # returns flattened sequences
    def sample(self, n: int) -> Batch:
        idxs = self._rng.integers(0, self._idx_mapper.size - self.sequence_length, size=n, dtype=np.int64)
        idxs = (idxs[:, None] + np.arange(self.sequence_length)).ravel()

        samples = self._storage.get(idxs)
        return samples
