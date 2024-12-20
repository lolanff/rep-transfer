from typing import Type
from algorithms.BaseAgent import BaseAgent

from algorithms.nn.DQN import DQN

def getAgent(name) -> Type[BaseAgent]:
    if name.startswith("DQN"):
        return DQN

    raise Exception('Unknown algorithm')
