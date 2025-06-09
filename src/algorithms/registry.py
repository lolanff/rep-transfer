from typing import Type
from algorithms.BaseAgent import BaseAgent

from algorithms.nn.DQN import DQN
from algorithms.nn.DQNAux import DQNAux
from algorithms.nn.DRQN import DRQN
from algorithms.nn.DRQNAux import DRQNAux

def getAgent(name) -> Type[BaseAgent]:
    if name.startswith("DQNAux"):
        return DQNAux
    elif name.startswith("DQN"):
        return DQN
    elif name.startswith("DRQNAux"):
        return DRQNAux
    elif name.startswith("DRQN"):
        return DRQN

    raise Exception('Unknown algorithm')
