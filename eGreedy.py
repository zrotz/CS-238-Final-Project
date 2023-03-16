from MDP import *

class eGreedy:
    def __init__(self, model : MDP, meta : tuple):
        self._MDP = model
        self._eps = meta[0]
        self._dr = meta[1] # eps decay rate
        self._reset = meta[0] # store eps for reset

    def next_action(self, s : int, Q):
        if (np.max(Q[s, :]) == 0 and np.min(Q[s, :]) == 0) or np.random.random() < self._eps: # use random action at explored state
            return np.random.randint(0, self._MDP.A) # explore
        else:
            return np.argmax(Q[s, :]) # exploit
    
    def decay(self):
        self._eps = self._eps*self._dr if self._eps > 1e-2 else self._eps # decay until eps <= 1e-2