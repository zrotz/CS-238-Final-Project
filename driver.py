from MDP import *
from QLearning import *
from eGreedy import *
from Softmax import *
from UCB1 import *
from RewardViz import *
import matplotlib.pyplot as plt
from tqdm import tqdm

def exploration(model : MDP, lr : float, ex : eGreedy or Softmax or UCB1, epoch : int):
    viz_r = RewardViz(epoch, "exploration")
    ql = QLearning(model, lr)

    for _ in tqdm(np.arange(epoch)):
        total_r = 0

        while True:
            a = ex.next_action(model.S.get_idx(), ql.Q) # sample action from exp. policy
            bsp, psp, osp = model.T.update(model.S, a) # next state
            r = model.R.reward(model.S, a) # get next reward
            total_r += r
            ql.update(model.S.get_idx(), a, r, model.S.tup_to_idx(bsp, psp, osp)) # update ql
            if model.S.get_bs() < model.S.get_rl()-1:
                model.S.set_tup((bsp, psp, osp)) # update model state
            else:
                break

        model.S.reset()
        ex.decay()
        viz_r.update(total_r)
    viz_r.plot_rewards()
    return ql.Q


def exploitation(model : MDP, Q, epoch : int):
    viz_r = RewardViz(epoch, "exploitation")

    for _ in tqdm(np.arange(epoch)):
        model.S.reset()
        total_r = 0

        while True: 
            a = np.argmax(Q[model.S.get_idx(), :])
            r = model.R.reward(model.S, a)
            total_r += r
            bsp, psp, osp = model.T.update(model.S, a)
            if model.S.get_bs() < model.S.get_rl() - 1:
                model.S.set_tup((bsp, psp, osp))
            else:
                break
        viz_r.update(total_r)
    viz_r.plot_rewards()
    viz_r.histogram()

# Matt's visualization
def viz(model, Q):
    l = model.S.get_rl()
    Q0 = np.zeros((l, l, 2))
    Q1 = np.zeros((l, l, 2))

    for ps in np.arange(l):
        for bs in np.arange(l):
            osf = model.S.tup_to_idx(bs, ps, 0) # pedestrian not in street
            ost = model.S.tup_to_idx(bs, ps, 1) # pedestrian in street

            Q0[ps, bs, 0] = Q[osf, 0] # ped. on sidewalk
            Q0[ps, bs, 1] = Q[ost, 0] # ped. in road

            Q1[ps, bs, 0] = Q[osf, 1]
            Q1[ps, bs, 1] = Q[ost, 1]
   
    for i in np.arange(3): # assume road 15 units long
        for j in np.arange(5):
            plt.subplot(3, 5, 5*i + j+1)
            plt.title("ped. at " + str(i*5+j))
            plt.imshow(Q0[i*5+j, :, :], origin = "upper", vmin = Q0.min(), vmax = Q0.max())
    plt.suptitle("Action: stay [left col os = 0; right col os = 1]")
    plt.colorbar()
    plt.savefig("./results/stay.png")

    for i in np.arange(3): # assume road 15 units long
        for j in np.arange(5):
            plt.subplot(3, 5, i*5+j+1)
            plt.title("ped. at " + str(i*5+j))
            plt.imshow(Q1[i*5+j, :, :], origin = "upper", vmin = Q1.min(), vmax = Q1.max())
    plt.suptitle("Action: go [left col os = 0; right col os = 1]")
    plt.colorbar()
    plt.savefig("./results/go.png")

def main():
    strat = input("Enter exploration strategy (\"e\" = e-greedy; \"s\" = softmax; \"u\" = UCB1): ")
    assert(strat in ["e", "s", "u"])

    rlen = 15
    s0 = (0, rlen, 0) # bus at start, road length, ped. on sidewalk
    A = 2 # action space = [0, 1]; len(A) = 2
    cost = (-1, -200, 20) # base line and collision cost and terminal reward
    pr = 0.33 # pedestrian randomness (chance enters road)
    g = 0.6
    model = MDP(s0, A, cost, pr, g)
    lr = 0.2 # learning rate

    if strat == "e":
        meta = np.array([0.4, 0.9999], dtype = np.float64) # epsilon, decay rate
        ex = eGreedy(model, meta)
    elif strat == "s":
        meta = (0.1, 1.012)  # precision parameter, precision factor (>decay = higher num)
        ex = Softmax(meta)
    else:
        ex = UCB1(rlen, A, 50, 0.999)


    epoch = 1e5
    Q = exploration(model, lr, ex, epoch)
    file = input("Save Q to file [.npy]: ")
    np.save(file, Q)

    sim_number = 5e3
    exploitation(model, Q, sim_number)
    viz(model, Q)


if __name__ == "__main__":
    main()