import numpy as np
class MDP:
    class State:
        # Inputs: substate init., road length, ped. randomness
        def __init__(self, s0 : tuple): 
            assert s0[1] > 1 # valid random interval
            self._bs = s0[0] # bus row loc
            self._ps = np.random.randint(0, s0[1]) # random row loc; s0[1] road length
            self._os = 0 if self._ps == 0 else s0[2] # ped. in road
            self._reset = s0 # store init. for reset

        def reset(self):
            self._bs = self._reset[0]
            self._ps = np.random.randint(0, self._reset[1])
            self._os = 0 if self._ps == 0 else np.random.randint(0, 2)

        def get_tup(self): # return MDP state as a tuple
            return (self._bs, self._ps, self._os)

        def get_bs(self):
            return self._bs
    
        def get_rl(self):
            return self._reset[1] # road length

        def set_tup(self, s : tuple):
            self._bs = s[0]
            self._ps = s[1]
            self._os = s[2]
        
        def get_idx(self): # needs road length
            rlen = self._reset[1]
            return np.array([self._bs, self._ps, self._os]).dot(np.array([1, rlen, rlen**2]))

        def tup_to_idx(self, bs : int, ps : int, os : int):
            return np.array([bs, ps, os]).dot(np.array([1, self.get_rl(), self.get_rl()**2]))


    class Transition:
        def __init__(self, pr : float):
            self._pr = pr # pedestrian randomness

        def update(self, s, a):
            assert a in [0, 1] # valid action
            bs, ps, os = s.get_tup() # current state
            rlen = s.get_rl() # road length

            if np.random.rand() < self._pr: # pedestrian update
                os = 0 if bs == ps else 1
            else:
                ps = ps if os else np.clip(ps + np.random.choice([-1, 1]), 0, rlen-1)
                os = 0
            
            if a == 1: # bus update
                bs = np.clip(bs+1, 0, rlen-1)
            
            return bs, ps, os
        
    class Reward:
        def __init__(self, cost : tuple):
            self._bc = cost[0] # base cost
            self._cc = cost[1] # collision cost
            self._tr = cost[2] # terminal reward

        def reward(self, s, a : int):
            bs, ps, os = s.get_tup() # return state tuple
            rlen = s.get_rl()-1

            # reward = self._bc if a == 0 else 0
            # reward +=  self._tr if bs == (rlen - 1) else 0  # check terminal state
            reward = self._bc if bs < rlen else self._tr # check terminal state

            reward += self._cc if bs == ps and os == 1 else 0 # check collision
            return reward
            
    def __init__(self, s0 : tuple, A : int, cost : tuple, pr : float, g : float):
        self.S = self.State(s0) # state object : state as tuple
        self.A = A # max action : assumes min action = 0
        self.R = self.Reward(cost) # reward object : reward function 
        self.T = self.Transition(pr) # transition object : transition function
        self.g = g # discount
    
    