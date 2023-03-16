from MDP import *
from QLearning import *
from eGreedy import *
import matplotlib.pyplot as plt
from tqdm import tqdm

def exploration(model : MDP, lr : float, meta : tuple, epoch : int):
    ql = QLearning(model, lr)
    eg = eGreedy(model, meta)

    for _ in tqdm(np.arange(epoch)):
        while True:
            a = eg.next_action(model.S.get_idx(), ql.Q) # sample action from exp. policy
            bsp, psp, osp = model.T.update(model.S, a) # next state
            r = model.R.reward(model.S, a) # get next reward
            ql.update(model.S.get_idx(), a, r, model.S.tup_to_idx(bsp, psp, osp)) # update ql
            if model.S.get_bs() < model.S.get_rl()-1:
                model.S.set_tup((bsp, psp, osp)) # update model state
            else:
                break
        model.S.reset()
        eg.decay()
    return ql.Q

# Zach's visualization
def visualize_1(model : MDP, Q, rl : int):
    bus_location = 0
    ped_y = np.random.randint(1, rl)
    ped_x = np.random.randint(0, 3)

    cur_state = model.State((bus_location, ped_y, 1 if ped_x == 1 else 0))
    # cur_state = model.State((0, 4, 0))

    while cur_state._bs != (rl - 1): 
        a = np.argmax(Q[cur_state.get_idx(rl)][:])
        print(cur_state._bs, cur_state._ps, cur_state._os, a)

        bs, ps, os = model.T.update(cur_state, a)
        cur_state = model.State((bs, ps, os))


    # end state
    print(cur_state._bs, cur_state._ps, cur_state._os)

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
    s0 = (0, 15, 0) # bus at start, road length, ped. on sidewalk
    A = 2 # action space = [0, 1]; len(A) = 2
    cost = (-1, -200, 20) # base line and collision cost and terminal reward
    pr = 0.33 # pedestrian randomness (chance enters road)
    g = 0.6
    model = MDP(s0, A, cost, pr, g)

    lr = 0.2 # learning rate
    meta = np.array([0.4, 0.9999], dtype = np.float64) # epsilon, decay rate
    epoch = 1e5
    Q = exploration(model, lr, meta, epoch)
    file = input("Save Q to file [.npy]: ")
    np.save(file, Q)

    visualize_1(model, Q, 15)
    viz(model, Q)


if __name__ == "__main__":
    main()

# 3/15/23 notes
# 1) randomly initialize pedestrian position
# 2) logic pedestrian not enter road when the bus is there => done
# 3) revise collision check in reward => done
# 4) decay epsilon with each journey, e = 0.7 dr = 0.999, stop at 1%
#       end journey at terminal state
# increase gamma or consider penalizing stay 