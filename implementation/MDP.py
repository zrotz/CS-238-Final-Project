import numpy as np
import matplotlib.pyplot as plt

class MDP:
  def __init__(self, nstates, nactions, R_in, T_in, γ_in):
    self.S = np.arange(nstates) # state space
    self.A = np.arange(nactions) # action space
    self.R = R_in # reward matrix R[s, a]
    self.T = T_in # transition matrix T[a, s, s']
    self.γ = γ_in # discount
    self.k = 20 # value iteration count
    self.ped_pen = -100

  # Input: Current value function
  # Output: Update value function
  def vi(self, U):
    assert np.ma.size(U, axis = 0) == np.ma.size(self.S, axis = 0) # proper dimensions
    for _ in np.arange(self.k):
      for (i, s) in enumerate(self.S):
        U[i] = np.max(self.R[s, :].T + self.γ*(self.T[:, s, :] @ U))
    return U

  def set_R(self, iter):
    assert iter > 0
    if iter % 3 == 2:
      self.R[2, :] = self.ped_pen # pedestrian starts crossing row 2
    else:
      self.R[2, :] = 0 # road clear
  
  def step(self, U, s):
    a = self.__get_a(U, s) # get action
    if a > 0:
      return s+1
    else:
      plt.figure(1)
      plt.imshow(U, extent = (0, 1, 0, 15), origin = "lower")
      plt.colorbar()
      plt.savefig("./plots/reward_crossing.png")
      return s

  def __get_a(self, U, s):
    return np.argmax(self.R[s, :].T + self.γ*(self.T[:, s, :] @ U))