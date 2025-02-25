# Adapted from https://github.com/erfanMhi/LTA-Representation-Properties/tree/main/core/component/auxiliary_tasks.py
import numpy as np
from representations.networks import NetworkBuilder
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import os
import seaborn as sns

from utils import torch_utils
from utils.hk import MultiLayerHead


class AuxTask(nn.Module):
    def __init__(self):
        super().__init__()
        
class VirtualVF5(AuxTask):
    def __init__(self):
        super().__init__()

    def compute_rewards_dones(self, state):
        pass

    def compute_loss(self, transition, phi, nphi, action_next):


        self.total_steps += 1

        return loss
    
    def _build_heads(self, builder: NetworkBuilder) -> None:
        pass
        
    # the network
    def aux_predictor(self, phi):
        pass

# TODO: add factory pattern
# class AuxFactory:
#     @classmethod
#     def get_aux_task():