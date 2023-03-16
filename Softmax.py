from MDP import *
import math
from numpy import random as r

class Softmax:
    def __init__(self, l : float, a : float):
        self._l = l  # precision paramter
        self._a = a  # precision factor
        self._weights = [0.5, 0.5]  # initial probability for action 0,1

    def next_action(self, s : int, Q):
        weights = [math.exp(self._l * i) for i in self._weights]
        t = sum(weights)
        self._weights = [i/t for i in weights]
        return r.choice([0,1], p=self._weights)
    
    def decay(self):
        self._l *= self._a