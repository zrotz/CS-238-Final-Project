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
        p = np.percentile(self._y, [25, 50, 75, 95])
        print("\n25th, 50th, 75th, 95th percentiles: ", p)
        plt.hist(self._y, bins = np.max(self._y) - np.min(self._y), color = 'b')
        plt.axvline(p[0], label = "25th percentile", color = 'g')
        plt.axvline(p[1], label = "50th percentile", color = 'r')
        plt.axvline(p[2], label = "75th percentile", color = 'k')
        plt.axvline(p[3], label = "95th percentile", color = 'm')
        plt.xlabel("Reward")
        plt.ylabel("Count")
        plt.legend()
        plt.savefig("./results/hist.png")
        plt.clf()

    def plot_rewards(self):
        plt.plot(self._x, self._y)
        plt.xlabel("Iteration")
        plt.ylabel("Reward")
        plt.title("Rewards " + self._filename)
        plt.savefig(self._filename + ".png")
        plt.clf()

