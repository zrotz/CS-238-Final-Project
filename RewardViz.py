from MDP import *
import matplotlib.pyplot as plt
import numpy as np

class RewardViz:
    def __init__(self, epoch : int, filename : str):
        self._x = np.arange(epoch)
        self._y = []
        self._filename = filename

    def update(self, y : int):
        self._y.append(y)
    
    def histogram(self):
        plt.hist(self._y, bins = 20)
        plt.xlabel("Reward")
        plt.ylabel("Count")
        plt.savefig("./results/hist.png")
        plt.clf()

    def plot_rewards(self):
        plt.plot(self._x, self._y)
        plt.xlabel("Iteration")
        plt.ylabel("Reward")
        plt.title("Rewards " + self._filename)
        plt.savefig(self._filename + ".png")
        plt.clf()

