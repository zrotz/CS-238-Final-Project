from MDP import *
import numpy as np
from math import sqrt, log, inf

class UCB1:
    def __init__(self, num_s : int, num_a : int, c : float, decay : float):
        self._N = np.zeros((num_s,num_a))  # initial counts for state, action pair
        self._c = c  # exploration parameter
        self._d = decay

    def __bonus(self, s : int, a : int):
        if self._N[s,a] == 0:
            return inf
        else:
            return self._c * sqrt(log(np.sum(self._N)) / self._N[s,a])
        
    def next_action(self, s : int, Q):
        a = np.argmax([Q[s,i] + self.__bonus(s,i) for i in range(0, len(Q[s,:]))])
        self.update_counts(s,a)
        return a
    
    def update_counts(self, s : int, a : int):
        self._N[s,a] += 1
    
    def decay(self):
        if self._c > 2:
            self._c *= self._d