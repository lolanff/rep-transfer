from typing import Type
from algorithms.BaseAgent import BaseAgent

from algorithms.nn.DQN import DQN
from algorithms.nn.DQNAux import DQNAux
from algorithms.nn.DRQN import DRQN
from algorithms.nn.DRQNAux import DRQNAux
from algorithms.nn.PPO import PPO
from algorithms.nn.ARDRQN import ARDRQN
from algorithms.nn.ARDQN import ARDQN

def getAgent(name) -> Type[BaseAgent]:
    if name.startswith("DQNAux"):
        return DQNAux
    elif name.startswith("DQN"):
        return DQN
    elif name.startswith("DRQNAux"):
        return DRQNAux
    elif name.startswith("DRQN"):
        return DRQN
    elif name.startswith("PPO"):
        return PPO
    elif name.startswith("ARDRQN"):
        return ARDRQN
    elif name.startswith("ARDQN"):
        return ARDQN

    raise Exception('Unknown algorithm')
