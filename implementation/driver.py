import MDP
import numpy as np
import matplotlib.pyplot as plt

def main():
    nstates = 15 # discrete 15 segment road
    nactions = 2 # 0: staty, 1: forward

    R = np.zeros((nstates, nactions)) # R(s, a)
    R[-1, :] = 1 # reward at terminal state

    T = np.zeros((nactions, nstates, nstates)) # an array of transition matrices indexed by action
    T[0, :, :] = np.eye(nstates, nstates) # stay in place
    for i in np.arange(nstates-1):
        T[1, i, i+1] = 1
        T[1, -1, -1] = 1 # absorbing terminal state

    γ = 0.7

    # Build MDP
    mdp = MDP(nstates, nactions, R, T, γ)
    U = np.zeros((nstates, 1))
    debug = False

    if debug: # test one step vi
        U = mdp.vi(U) # VI with zero initial value

        plt.figure(1)
        plt.imshow(U, origin = "lower", extent = (0, 1, 0, 15))
        plt.colorbar()
    else:
        traj = [0] # initial position
    while True:
        mdp.set_R(len(traj))
        U = mdp.vi(U)
        traj.append(mdp.step(U, traj[-1]))
        if traj[-1] >= nstates-1:
            break
    plt.figure(2)
    plt.plot(np.arange(0, len(traj))+1, traj)
    plt.xlabel("Iteration")
    plt.ylabel("Bus State")

if __name__ == "__main__":
    main()