from MDP import *
from QLearning import *
from eGreedy import *
import matplotlib.pyplot as plt
from tqdm import tqdm

def exploration(model : MDP, lr : float, meta : tuple, epoch : int):
    ql = QLearning(model, lr)
    eg = eGreedy(model, meta)
    rlen = model.T.get_length()

    for _ in tqdm(np.arange(epoch)):
        while eg.decay() > 1e-2:
            a = eg.next_action(model.S.get_idx(rlen), ql.Q) # sample action from exp. policy
            bsp, psp, osp = model.T.update(model.S, a) # next state
            r = model.R.reward(model.S, a, rlen-1) # get next reward
            ql.update(model.S.get_idx(rlen), a, r, MDP.State((bsp, psp, osp)).get_idx(rlen)) # update ql
            model.S.set_tup((bsp, psp, osp)) # update model state
        eg.reset()
        model.S.reset()
    return ql.Q

# Zach's visualization
def visualize_1(model : MDP, Q, rl : int):
    bus_location = 0
    ped_y = np.random.randint(1, rl)
    ped_x = np.random.randint(0, 3)

    # cur_state = model.State((bus_location, ped_y, 1 if ped_x == 1 else 0))
    cur_state = model.State((0, 4, 0))

    while cur_state._bs != (rl - 1): 
        a = np.argmax(Q[cur_state.get_idx(rl)][:])
        print(cur_state._bs, cur_state._ps, cur_state._os, a)

        bs, ps, os = model.T.update(cur_state, a)
        cur_state = model.State((bs, ps, os))


    # end state
    print(cur_state._bs, cur_state._ps, cur_state._os)

# Matt's visualization
def viz(Q, l):
    Q0 = np.zeros((l, l, 2))
    Q1 = np.zeros((l, l, 2))

    for ps in np.arange(l):
        for bs in np.arange(l):
            osf = MDP.State((bs, ps, 0)).get_idx(l) # pedestrian not in street
            ost = MDP.State((bs, ps, 1)).get_idx(l) # pedestrian in street

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
    plt.savefig("./stay.png")

    for i in np.arange(3): # assume road 15 units long
        for j in np.arange(5):
            plt.subplot(3, 5, i*5+j+1)
            plt.title("ped. at " + str(i*5+j))
            plt.imshow(Q1[i*5+j, :, :], origin = "upper", vmin = Q1.min(), vmax = Q1.max())
    plt.suptitle("Action: go [left col os = 0; right col os = 1]")
    plt.colorbar()
    plt.savefig("./go.png")



def main():
    s0 = (0, 5, 0) # bus at start, ped. at row 5 on sidewalk
    A = 2 # action space = [0, 1]; len(A) = 2
    cost = (-1, -200, 20) # base line and collision cost and terminal reward
    env = (15, 0.33) # road length, pedestrian randomness (chance enters road)
    g = 0.4
    model = MDP(s0, A, cost, env, g)

    lr = 0.2 # learning rate
    meta = (0.41, 0.97) # epsilon, decay rate
    epoch = 1e4
    Q = exploration(model, lr, meta, epoch)

    visualize_1(model, Q, 15)
    # viz(Q, 15)


if __name__ == "__main__":
    main()