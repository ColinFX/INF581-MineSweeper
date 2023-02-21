import numpy as np
import torch
from tqdm import trange

from game import MineSweeper
from Models.ddqn import DDQN


def test_hybrid(trained_model, width=9, height=9, bomb_no=10, rule='win7', simulation_no=20000, hybrid=True):
    """
    Experiment trained_model on designated environment and return win rate,
    action taken by the model only if intervention requested by the built-in auto_play mechanism.
    """
    env = MineSweeper(width, height, bomb_no, rule)
    won = 0
    with trange(1, simulation_no + 1) as steps:
        for _ in steps:
            env.reset()
            terminal = False
            reward = None
            while not terminal:
                # get action from trained model
                state_flatten = env.state.flatten()
                mask_flatten = (1 - env.fog).flatten()
                mask_flatten[state_flatten == -2] = 0
                state_flatten = np.maximum(state_flatten, -1)
                action = trained_model.act(state_flatten, mask_flatten)
                # take action on environment and auto_play until terminal or next intervention
                i = int(action / width)
                j = action % width
                _, terminal, reward = env.choose(i, j, auto_play=hybrid)
            if reward == 1:
                won += 1
    return won / simulation_no


if __name__ == "__main__":
    test_width = 6
    test_height = 6
    test_bomb_no = 6
    test_rule = 'win7'

    test_model = DDQN(test_width * test_height, test_width * test_height, cuda=False)
    test_model.load_state(torch.load("pre-trained/ddqn_dnn20000.pth"))

    win_rate = test_hybrid(test_model, test_width, test_height, test_bomb_no, test_rule)
    print("win rate =", win_rate)
