import numpy as np
import torch

from argument import get_args

class BaseConfig(object):
    def __init__(self):
        pass

class Config(object):
    args = get_args()

    training = BaseConfig()
    training.device = "cuda:0" if torch.cuda.is_available() else "cpu"

    sim = BaseConfig()
    sim.render = False