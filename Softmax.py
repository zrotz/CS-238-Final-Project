from MDP import *
import math
from numpy import random as r

class Softmax:
    def __init__(self, l : float, a : float):
        self._l = l  # precision paramter
        self._a = a  # precision factor

    def next_action(self, s : int, Q):
        weights = [math.exp(self._l * Q[s][0]), math.exp(self._l * Q[s][1])]
        t = sum(weights)
        norm_weights = [i/t for i in weights]
        return r.choice([0,1], p=norm_weights)
    
    def decay(self):
        self._l *= self._a