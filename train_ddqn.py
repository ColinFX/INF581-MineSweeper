import multiprocessing

import torch
from numpy import float32, ndarray, asarray
from torch import FloatTensor, LongTensor
from torch.autograd import Variable

from Models.ddqn import DDQN, Buffer
from game import MineSweeper
from renderer import Render

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

'''
GAME PARAMS:
width       = width of board
height      = height of board
bomb_no     = bombs on map
parallel_n0 = number of MineSweeper play boards processed in parallel

AI PARAMS:
optimizer:
    lr          = learning rate at 0.002, weight decay 1e-5
    scheduler   = reduces learning rate to 0.95 of itself every 2000 steps
buffer  = stores the State, Action, Reward, Next State, Terminal?, and Masks for each state
gamma   = weightage of reward to future actions
epsilon = the randomness of the DDQN agent
    this is not decayed by linear or exponential methods, 
    RBED is used ( Reward Based Epsilon Decay )
        if reward_threshold is exceeded, epsilon becomes 0.9x of itself
        and next the reward_threshold is increased by reward_step
batch_size = set to 4096 decisions before each update

main() function PARAMS:
save_every : Saves the model every x steps
update_targ_every : Updates the target model to current model every x steps
    (Note I have to try interpolated tau style instead of hard copy)
epochs: self explanatory
logs: Win Rate, Reward, Loss and Epsilon are written to this file and can be visualized using ./Logs/plotter.py
'''


class ParallelDriver:
    def __init__(self, width, height, bomb_no, render_flag, parallel_no=1):
        self.width = width
        self.height = height
        self.bomb_no = bomb_no
        self.box_count = width * height
        self.parallel_no = parallel_no
        self.envs = [MineSweeper(self.width, self.height, self.bomb_no) for _ in range(self.parallel_no)]

        self.current_model = DDQN(self.box_count, self.box_count).to(device)
        self.target_model = DDQN(self.box_count, self.box_count).to(device)
        self.target_model.eval()
        self.optimizer = torch.optim.Adam(self.current_model.parameters(), lr=0.003, weight_decay=1e-5)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=2000, gamma=0.95)
        self.target_model.load_state_dict(self.current_model.state_dict())

        self.buffer = Buffer(65536)
        self.gamma = 0.99
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.90
        self.reward_threshold = 0.12
        self.reward_step = 0.01
        self.batch_size = 4096
        self.tau = 5e-5
        self.log = open("./Logs/ddqn_log.txt", 'w')

        # Only render pygame interface if parallel sampling is disabled
        self.render_flag = render_flag and parallel_no == 1
        if self.render_flag:
            self.Render = Render(self.envs[0].state)

    # Load from checkpoint for testing
    def load_models(self, iter_num):
        path = "./pre-trained/ddqn_dnn" + str(iter_num) + ".pth"
        weights = torch.load(path)
        self.current_model.load_state_dict(weights['current_state_dict'])
        self.target_model.load_state_dict(weights['target_state_dict'])
        self.optimizer.load_state_dict(weights['optimizer_state_dict'])
        self.current_model.epsilon = weights['epsilon']

    # Get an action from the DDQN model by supplying it State and Mask
    def get_actions(self, states: ndarray, masks: ndarray):
        # states = states.reshape(self.parallel_no, -1)
        # masks = masks.reshape(self.parallel_no, -1)
        # actions = self.current_model.act(states, masks)  # TODO: model need to support parallelism!
        # return actions
        actions = []
        for env_id in range(self.parallel_no):
            state = states[env_id]
            mask = masks[env_id]
            actions.append(self.current_model.act(state, mask))
        return asarray(actions)


    # Do the action and returns Next State, If terminal, Reward, Next Mask
    def do_step(self, action: ndarray, env_id: int):
        i = int(action / self.width)
        j = action % self.width
        next_state, terminal, reward = self.envs[env_id].choose(i, j)
        next_fog = 1 - self.envs[env_id].fog
        return next_state, terminal, reward, next_fog

    def do_step_parallel(self, actions):
        with multiprocessing.Pool() as pool:
            items = [(actions[env_id], env_id) for env_id in range(self.parallel_no)]
            next_states, terminals, rewards, next_fogs = zip(*pool.starmap(self.do_step, items))
        if self.render_flag:
            self.Render.state = self.envs[0].state
            self.Render.draw()
            self.Render.bugfix()
        return asarray(next_states), asarray(terminals), asarray(rewards), asarray(next_fogs)

    # Reward Based Epsilon Decay
    def epsilon_update(self, avg_reward):
        if avg_reward > self.reward_threshold:
            self.current_model.epsilon = max(self.epsilon_min, self.current_model.epsilon * self.epsilon_decay)
            self.reward_threshold += self.reward_step

    def TD_Loss(self):
        # Samples batch from buffer memory
        state, action, mask, reward, next_state, next_mask, terminal = self.buffer.sample(self.batch_size)

        # Converts the variabls to tensors for processing by DDQN
        state = Variable(FloatTensor(float32(state))).to(device)
        mask = Variable(FloatTensor(float32(mask))).to(device)
        next_state = FloatTensor(float32(next_state)).to(device)
        action = LongTensor(float32(action)).to(device)
        next_mask = FloatTensor(float32(next_mask)).to(device)
        reward = FloatTensor(reward).to(device)
        done = FloatTensor(terminal).to(device)

        # Predicts Q value for present and next state with current and target model
        q_values = self.current_model(state, mask)
        next_q_values = self.target_model(next_state, next_mask)
        q_crt_values = self.current_model(next_state, next_mask)
        action_max = torch.Tensor([torch.argmax(q_val) for q_val in q_crt_values]).to(device)

        q_value = q_values.gather(1, action.unsqueeze(1)).squeeze(1)
        next_q_value = torch.Tensor([next_q_values[i][int(action_max[i])] for i in range(action_max.shape[0])]).to(
            device)
        expected_q_value = reward + self.gamma * next_q_value * (1 - done)
        loss = (q_value - expected_q_value.detach()).pow(2).mean()
        loss_print = loss.item()

        # Back-propagates the Loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.scheduler.step()

        for target_param, local_param in zip(self.target_model.parameters(), self.current_model.parameters()):
            target_param.data.copy_(self.tau * local_param.data + (1.0 - self.tau) * target_param.data)
        return loss_print

    def save_checkpoints(self, batch_no):
        path = "./pre-trained/ddqn_dnn" + str(batch_no) + ".pth"
        torch.save({
            'epoch': batch_no,
            'current_state_dict': self.current_model.state_dict(),
            'target_state_dict': self.target_model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.current_model.epsilon
        }, path)

    def save_logs(self, batch_no, avg_reward, loss, wins):
        res = [
            str(batch_no),
            "\tAvg Reward: ",
            str(avg_reward),
            "\t Loss: ",
            str(loss),
            "\t Wins: ",
            str(wins),
            "\t Epsilon: ",
            str(self.current_model.epsilon)
        ]
        log_line = " ".join(res)
        print(log_line)
        self.log.write(log_line + "\n")
        self.log.flush()


if __name__ == "__main__":
    driver = ParallelDriver(6, 6, 6, False, parallel_no=2)
    states = asarray([env.state for env in driver.envs])
    epochs = 10000
    save_every = 2000
    count = 0
    running_reward = 0
    batch_no = 0
    wins = 0
    total = 0

    while (batch_no < epochs):
        # simple state action reward loop and writes the actions to buffer
        masks = 1 - asarray([env.fog for env in driver.envs])
        actions = driver.get_actions(states, masks)
        next_states, terminals, rewards, _ = driver.do_step_parallel(actions)

        for env_id in range(driver.parallel_no):
            driver.buffer.push(
                states[env_id].flatten(),
                actions[env_id],
                masks[env_id].flatten(),
                rewards[env_id],
                next_states[env_id].flatten(),
                (1 - driver.envs[env_id].fog).flatten(),
                terminals[env_id]
            )
            if terminals[env_id]:
                if rewards[env_id]:
                    wins += 1
                driver.envs[env_id].reset()
                states[env_id] = driver.envs[env_id].state
                masks[env_id] = driver.envs[env_id].fog
                total += 1

        states = next_states
        count += driver.parallel_no
        running_reward += sum(rewards)

        # Used for calculating win-rate for each batch

        if count >= driver.batch_size:
            # Computes the Loss
            driver.current_model.train()
            loss = driver.TD_Loss()
            driver.current_model.eval()

            # Calculates metrics
            batch_no += 1
            avg_reward = running_reward / count
            wins = wins * 100 / total
            driver.save_logs(batch_no, avg_reward, loss, wins)

            # Updates epsilon based on reward
            driver.epsilon_update(avg_reward)

            # Resets metrics for next batch calculation
            running_reward = 0
            count = 0
            wins = 0
            total = 0

            # Saves the model details to "./pre-trained" if 1000 batches have been processed
            if batch_no % save_every == 0:
                driver.save_checkpoints(batch_no)
