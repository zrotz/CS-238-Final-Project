import numpy as np
class MDP:
    class State:
        # Inputs: substate init., road length, ped. randomness
        def __init__(self, s0 : tuple): 
            self._bs = s0[0] # bus row loc
            self._ps = s0[1] # ped. row loc
            self._os = s0[2] # ped. in road
            self._reset = s0 # store init. for reset

        def reset(self):
            self._bs = self._reset[0]
            self._ps = self._reset[1]
            self._os = self._reset[2]

        def get_tup(self): # return MDP state as a tuple
            return (self._bs, self._ps, self._os)

        def set_tup(self, s : tuple):
            self._bs = s[0]
            self._ps = s[1]
            self._os = s[2]
        
        def get_idx(self, rlen): # needs road length
            return np.array([self._bs, self._ps, self._os]).dot(np.array([1, rlen, rlen**2]))

    class Transition:
        def __init__(self, env : tuple):
            self._l = env[0] # road length
            self._pr = env[1] # pedestrian randomness

        def get_length(self):
            return self._l # return road length

        def update(self, s, a):
            assert a in [0, 1] # valid action
            bs, ps, os = s.get_tup() # current state

            if np.random.rand() < self._pr: # pedestrian update
                os = 0 if bs == ps else 1
            else:
                ps = ps if os else np.clip(ps + np.random.choice([-1, 1]), 0, self._l-1)
                os = 0
            
            if bs < self._l-1:
                if a == 1: # bus update
                    bs = np.clip(bs+1, 0, self._l-1)
            else: # reach terminal state, teleport to start independent of action
                bs = 0
            
            return bs, ps, os
        
    class Reward:
        def __init__(self, cost : tuple):
            self._bc = cost[0] # base cost
            self._cc = cost[1] # collision cost
            self._tr = cost[2] # terminal reward

        def reward(self, s, a : int, rlen : int):
            bs, ps, os = s.get_tup() # return state tuple
            
            reward = self._bc if a == 0 else 0
            reward +=  self._tr if bs == (rlen - 1) else 0  # check terminal state
            # reward += self._bc if bs < rlen else self._tr # check terminal state

            reward += self._cc if (bs == ps) and (os == 1) else 0 # check collision
            # reward += self._cc if bs == ps else 0 # check collision
            return reward
            
    def __init__(self, s0 : tuple, A : int, cost : tuple, env : tuple, g : float):
        self.S = self.State(s0) # state object : state as tuple
        self.A = A # max action : assumes min action = 0
        self.R = self.Reward(cost) # reward object : reward function 
        self.T = self.Transition(env) # transition object : transition function
        self.g = g # discount
    
    