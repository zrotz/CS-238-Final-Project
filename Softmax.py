from MDP import *
import math
from numpy import random as r

class Softmax:
    def __init__(self, meta : tuple):
        self._l = meta[0]  # precision paramter
        self._a = meta[1]  # precision factor

    def next_action(self, s : int, Q):
        weights = [math.exp(self._l * (Q[s][0] / 100)), math.exp(self._l * (Q[s][1] / 100))]
        t = sum(weights)
        norm_weights = [i/t for i in weights]
        return r.choice([0,1], p=norm_weights)
    
    def decay(self):
        if self._l < 50:
            self._l *= self._a