import time

import numpy as np
import torch
from tqdm import tqdm

from game import MineSweeper
from renderer import Render
from Models.stochastic import STOCHASTIC
from Models.ddqnCNN import DDQNCNNL
from Models.ddqn import DDQN
from Models.dqn import DQN
from Models.ppo import PPO
from Models.AC0 import AC0

model_list = {"DDQN": DDQN, "DQN": DQN, "PPO": PPO, "DDQNCNNL": DDQNCNNL, "STOCHASTIC": STOCHASTIC}  # First Release

weight_map = {"DDQNCNNL": "pre-trained/ddqncnnl_win7_19000.pth",
              "DDQN": "pre-trained/ddqn_dnn20000.pth",
              "DQN": "pre-trained/dqn_dnn10000.pth",
              "PPO": "pre-trained/ppo_dnn8000.pth"}

def test_hybrid(trained_model, width=9, height=9, bomb_no=10, rule='win7', simulation_no=2000, hybrid=True):
    """
    Experiment trained_model on designated environment and return win rate,
    action taken by the model only if intervention requested by the built-in auto_play mechanism.
    """
    env = MineSweeper(width, height, bomb_no, rule)
    won = 0
    lost = 0
    intervention = 0
    intervention_won = 0

    for game in tqdm(range(simulation_no)):
        env.reset()
        terminal = False
        first_click = True
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

            if not first_click and terminal:
                intervention += 1
                if reward == 1:
                    intervention_won += 1
            first_click = False

        if reward == 1:
            won += 1
        elif reward == -1:
            lost += 1
        else:
            raise AssertionError
    return won, lost, intervention, intervention_won


def test_hybrid_slow(trained_model, width=9, height=9, bomb_no=10, rule='win7', simulation_no=10, hybrid=True):
    """
    Experiment trained_model on designated environment and display step by step,
    action taken by the model only if intervention requested by the built-in auto_play mechanism.
    """
    for game in range(simulation_no):
        #time.sleep(1.5)
        print(">")
        env = MineSweeper(width, height, bomb_no, rule)
        renderer = Render(env.state)
        terminal = False
        first_click = True
        reward = None
        while not terminal:
            # time.sleep(1.5)
            if not first_click:
                print("Intervention")
            first_click = False
            state_flatten = env.state.flatten()
            mask_flatten = (1 - env.fog).flatten()
            mask_flatten[state_flatten == -2] = 0
            state_flatten = np.maximum(state_flatten, -1)
            action = trained_model.act(state_flatten, mask_flatten)
            i = int(action / width)
            j = action % width
            _, terminal, reward = env.choose(i, j, auto_play=hybrid)
            renderer.state = env.state
            renderer.draw()
            renderer.bugfix()
        if reward == 1:
            print("WON")
        else:
            print("LOST")


def load_weight(model_obj, model_type_, nb_cuda_):
    if nb_cuda_ == -1:
        model_obj.load_state(torch.load(weight_map[model_type_], map_location=torch.device("cpu")))
    else:
        model_obj.cuda()
        model_obj.load_state(torch.load(weight_map[model_type_]))
    return model_obj


if __name__ == "__main__":

    model_type = "DDQNCNNL"
    test_rule = 'win7'
    test_simulation_no = 20000
    nb_cuda = -1  # -1 = cpu; else specify cuda device: ex. 0 = cuda:0
    render_flag = True
    test_width, test_height, test_bomb_no = 9, 9, 10

    if model_type == "DDQNCNNL":
        test_model = model_list[model_type](width=test_width, height=test_height, action_dim=test_width*test_height, nb_cuda=nb_cuda)
        test_model = load_weight(test_model, model_type, nb_cuda)
    elif model_type in ["DDQN", "DQN", "PPO"]:
        test_width, test_height, test_bomb_no = 6, 6, 6
        test_model = model_list[model_type](inp_dim=test_width*test_height, action_dim=test_width*test_height)
        test_model = load_weight(test_model, model_type, nb_cuda)
    else:
        # STOCHASTIC
        test_model = model_list[model_type]()

    test_model.epsilon = 0

    if render_flag:
        test_hybrid_slow(test_model, test_width, test_height, test_bomb_no, test_rule, 50, hybrid=True)

    else:
        res = test_hybrid(test_model, test_width, test_height, test_bomb_no, test_rule, test_simulation_no, hybrid=False)
        print(f"{test_simulation_no} games simulated with {res[0]} won ({res[0]/test_simulation_no*100}%) and {res[1]} "
              f"lost ({res[1]/test_simulation_no*100}%).\n{res[2]} games ({res[2]/test_simulation_no*100}%) requested "
              f"intervention, out of which {res[3]} succeeded ({res[3]/res[2]*100}%).")

