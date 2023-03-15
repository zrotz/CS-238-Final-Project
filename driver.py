from MDP import *
from QLearning import *
from eGreedy import *
import matplotlib.pyplot as plt

def exploration(model : MDP, lr : float, meta : tuple, epoch : int):
    ql = QLearning(model, lr)
    eg = eGreedy(model, meta)
    rlen = model.T.get_length()

    for _ in np.arange(epoch):
        while eg.decay() > 5e-2:
            a = eg.next_action(model.S.get_idx(rlen), ql.Q) # sample action from exp. policy
            bsp, psp, osp = model.T.update(model.S, a) # next state
            r = model.R.reward(model.S, a, rlen-1) # get next reward
            ql.update(model.S.get_idx(rlen), a, r, MDP.State((bsp, psp, osp)).get_idx(rlen)) # update ql
            model.S.set_tup((bsp, psp, osp)) # update model state
        eg.reset()
        model.S.reset()
    return ql.Q

def visualize_heatmap(model : MDP, Q, rl : int):
    bus_location = 0
    ped_y = np.random.randint(1, rl)
    ped_x = np.random.randint(0, 3)

    cur_state = model.State((bus_location, ped_y, 1 if ped_x == 1 else 0))

    while cur_state._bs != (rl - 1): 
        a = np.argmax(Q[cur_state.get_idx(rl)][:])
        print(cur_state._bs, cur_state._ps, cur_state._os, a)

        bs, ps, os = model.T.update(cur_state, a)
        cur_state = model.State((bs, ps, os))


    # end state
    print(cur_state._bs, cur_state._ps, cur_state._os)






def main():
    s0 = (0, 5, 0) # bus at start, ped. at row 5 on sidewalk
    A = 2 # action space = [0, 1]; len(A) = 2
    cost = (-1, -200, 20) # base line and collision cost and terminal reward
    env = (15, 0.67) # road length, pedestrian randomness (chance enters road)
    g = 0.4
    model = MDP(s0, A, cost, env, g)

    lr = 0.2 # learning rate
    meta = (0.41, 0.9) # epsilon, decay rate
    epoch = 1e4
    Q = exploration(model, lr, meta, epoch)

    visualize_heatmap(model, Q, 15)


if __name__ == "__main__":
    main()