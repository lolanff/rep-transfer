import numpy as np
from abc import abstractmethod
from typing import Any, NamedTuple
from ReplayTables._utils.logger import logger
from ReplayTables.interface import Timestep, LaggedTimestep, Batch, Item, TransIds
from ReplayTables.ingress.IndexMapper import IndexMapper
from ReplayTables.ingress.CircularMapper import CircularMapper
from ReplayTables.ingress.LagBuffer import LagBuffer
from ReplayTables.sampling.IndexSampler import IndexSampler
from ReplayTables.sampling.UniformSampler import UniformSampler
from ReplayTables.storage.BasicStorage import BasicStorage
from ReplayTables.storage.Storage import Storage
from ReplayTables.ReplayBuffer import ReplayBuffer

class CarryBatch(NamedTuple):
    x: np.ndarray
    a: np.ndarray
    r: np.ndarray
    gamma: np.ndarray
    terminal: np.ndarray
    trans_id: TransIds
    xp: np.ndarray
    carry: np.ndarray
    
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
    
    # Returns flattened sequences
    def sample_sequences(self, n: int) -> CarryBatch:
        frontal_tid = self._lag_buffer._tid
        mapper_size = self._idx_mapper.size

        high = frontal_tid - self.sequence_length + 1
        low = max(frontal_tid - mapper_size - self.sequence_length, 0)
        idxs = self._rng.integers(low, high, size=n, dtype=np.int64)

        idxs = (idxs[:, None] + np.arange(self.sequence_length)).ravel()
        idxs = idxs % mapper_size
        items = self._storage.meta.get_items_by_idx(idxs)

        x = self._storage._load_states(items.sidxs)
        xp = self._storage._load_states(items.n_sidxs)

        return CarryBatch(
            x=x,
            a=self._storage._a[idxs],
            r=self._storage._r[idxs],
            gamma=self._storage._gamma[idxs],
            terminal=self._storage._term[idxs],
            trans_id=items.trans_ids,
            xp=xp,
            carry=np.array([self._storage._extras[i]['carry'] for i in idxs])
        )
