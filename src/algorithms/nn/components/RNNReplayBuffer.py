import numpy as np
from abc import abstractmethod
from typing import Any, NamedTuple, Tuple
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
    carryp: np.ndarray
    reset: np.ndarray
    pos: np.ndarray
    last_action_encoded: np.ndarray
    last_reward: np.ndarray
    
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
    def sample_sequences(self, n: int, sequence_length=None) -> CarryBatch:
        if sequence_length is None:
            sequence_length = self.sequence_length
        frontal_tid = self._lag_buffer._tid
        mapper_size = self._idx_mapper.size

        high = frontal_tid - sequence_length + 1
        low = max(frontal_tid - mapper_size - sequence_length, 0)
        idxs = self._rng.integers(low, high, size=n, dtype=np.int64)

        idxs = (idxs[:, None] + np.arange(sequence_length)).ravel()
        idxs = idxs % mapper_size
        items = self._storage.meta.get_items_by_idx(idxs)
        
        trans_ids = TransIds(np.array(items.trans_ids).reshape(n, sequence_length))

        x = self._storage._load_states(items.sidxs)
        xp = self._storage._load_states(items.n_sidxs)
       
        x = x.reshape(n, sequence_length, *x.shape[1:])
        xp = xp.reshape(n, sequence_length, *xp.shape[1:])
        
        a = self._storage._a[idxs].reshape(n, sequence_length)
        r = self._storage._r[idxs].reshape(n, sequence_length)
        gamma = self._storage._gamma[idxs].reshape(n, sequence_length)
        term = self._storage._term[idxs].reshape(n, sequence_length)

        extras = self._storage._extras
        carry, carryp, reset, pos, last_action_encoded, last_reward = zip(*((extras[i]['carry'], extras[i]['carryp'], extras[i]['reset'], extras[i]['pos'], extras[i].get('last_action_encoded'), extras[i].get('last_reward')) for i in idxs))

        carry = np.array(carry).reshape(n, sequence_length, -1)
        carryp = np.array(carryp).reshape(n, sequence_length, -1)
        reset = np.array(reset).reshape(n, sequence_length)
        
        # Handle None values in pos by replacing them with default position arrays
        pos = [p if p is not None else np.array([-1, -1]) for p in pos]
        pos = np.array(pos).reshape(n, sequence_length, -1)
        
        # Handle None values in last_action_encoded and last_reward
        last_action_encoded = [lae if lae is not None else np.array([0.0]) for lae in last_action_encoded]
        last_reward = [lr if lr is not None else 0.0 for lr in last_reward]
        last_action_encoded = np.array(last_action_encoded).reshape(n, sequence_length, -1)
        last_reward = np.array(last_reward).reshape(n, sequence_length)

        return CarryBatch(
            x=x,
            a=a,
            r=r,
            gamma=gamma,
            terminal=term,
            trans_id=trans_ids,
            xp=xp,
            carry=carry,
            carryp=carryp,
            reset=reset,
            pos=pos,
            last_action_encoded=last_action_encoded,
            last_reward=last_reward
        )
