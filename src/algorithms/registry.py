from typing import Type
from algorithms.BaseAgent import BaseAgent

from algorithms.nn.DQN import DQN
from algorithms.nn.DQNAux import DQNAux

def getAgent(name) -> Type[BaseAgent]:
    if name.startswith("DQNAux"):
        return DQNAux
    elif name.startswith("DQN"):
        return DQN

    raise Exception('Unknown algorithm')
