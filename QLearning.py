from MDP import *

class QLearning:
    def __init__(self, model : MDP, lr : float):
        self._MDP = model
        r_len = model.S.get_rl() # road length
        self.Q = np.zeros((r_len*r_len*2, model.A)) # state space size, action space size
        self._lr = lr

    def update(self, s : int, a : int, r : int, sp : int):
        self.Q[s, a] += self._lr * (r + self._MDP.g*max(self.Q[sp, :]) - self.Q[s, a])