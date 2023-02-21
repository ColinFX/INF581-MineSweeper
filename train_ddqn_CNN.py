import torch
from numpy import float32
from torch import FloatTensor, LongTensor
from torch.autograd import Variable

from Models.ddqnCNN import DDQNCNN, Buffer,DDQNCNNL
from game import MineSweeper
from renderer import Render
from numpy import float32
from torch.autograd import Variable
from multiprocessing import Process
from torch import FloatTensor,LongTensor


'''
GAME PARAMS:
width   = width of board
height  = height of board
bomb_no = bombs on map
env     = minesweeper environment created from "game.py" with params

AI PARAMS:
optimizer:
    lr          = learning rate at 0.002, weight decay 1e-5
    scheduler   = reduces learning rate to 0.95 of itsemf every 2000 steps
buffer  = stores the State, Action, Reward, Next State, Terminal?, and Masks for each state
gamma   = weightage of reward to future actions
epsilon = the randomness of the DDQN agent
    this is not decayed by linear or exponential methods, 
    RBED is used ( Reward Based Epsilon Decay )
        if reward_threshold is exceeded, epsilon becomes 0.9x of itself
        and next the reward_threshold is increased by reward_step
batch_size = set to 2048 decisions before each update

Model Details:
I have made a pretty small model so that it executes fast and I can reiterate my parameters manually faster
DDQN with epsilon starting at 1 and reduces based on RBED
Has feature extractor layer
Has 2 heads to the model
    advantage and value
    combination of these 2 will give the q value of the state
IMPORTANT : I HAVE ADDED ACTION MASKING, WHICH IMPROVES PERFORMANCE


main() function PARAMS:
save_every : Saves the model every x steps
update_targ_every : Updates the target model to current model every x steps
    (Note I have to try interpolated tau style instead of hard copy)
epochs: self explanatory
logs: Win Rate, Reward, Loss and Epsilon are written to this file and can be visualized using ./Logs/plotter.py
'''


class Driver():

    def __init__(self, width, height, bomb_no, render_flag,continual_training,nb_cuda,use_lagre,env_type):

        self.width = width
        self.height = height
        self.bomb_no = bomb_no
        self.box_count = width*height
        self.nb_cuda = nb_cuda
        self.use_lagre = use_lagre
        self.env = MineSweeper(self.width,self.height,self.bomb_no,rule=env_type)
        if not self.use_lagre:
            self.current_model = DDQNCNN(self.width,self.height,self.box_count,self.nb_cuda).cuda(self.nb_cuda)
            self.target_model = DDQNCNN(self.width,self.height,self.box_count,self.nb_cuda).cuda(self.nb_cuda)
        else:
            self.current_model = DDQNCNNL(self.width,self.height,self.box_count,self.nb_cuda).cuda(self.nb_cuda)
            self.target_model = DDQNCNNL(self.width,self.height,self.box_count,self.nb_cuda).cuda(self.nb_cuda)
            
        self.target_model.eval()
        self.optimizer = torch.optim.Adam(self.current_model.parameters(), lr=1e-3, weight_decay=1e-5)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=2000, gamma=0.95)
        self.target_model.load_state_dict(self.current_model.state_dict())
        self.buffer = Buffer(100000)
        self.gamma = 0.99
        self.render_flag = render_flag
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.90
        self.reward_threshold = 0.12
        self.reward_step = 0.01
        self.batch_size = 4096
        self.tau = 5e-5
        if continual_training:
            self.log = open("./Logs/"+self.current_model.__class__.__name__.lower()+"_log_"+str(self.env.rule)+".txt", 'a')
        else:
            self.log = open("./Logs/"+self.current_model.__class__.__name__.lower()+"_log_"+str(self.env.rule)+".txt", 'w')

        if (self.render_flag):
            self.Render = Render(self.env.state)

    def load_models(self, number):
        path = "./pre-trained/"+self.current_model.__class__.__name__.lower()+"_" +str(self.env.rule) + "_" + str(number) + ".pth"
        weights = torch.load(path)
        self.current_model.load_state_dict(weights['current_state_dict'])
        self.target_model.load_state_dict(weights['target_state_dict'])
        self.optimizer.load_state_dict(weights['optimizer_state_dict'])
        self.current_model.epsilon = weights['epsilon']

    ### Get an action from the DDQN model by supplying it State and Mask
    def get_action(self, state, mask):
        state = state.flatten()
        mask = mask.flatten()
        action = self.current_model.act(state, mask)
        return action

    ### Does the action and returns Next State, If terminal, Reward, Next Mask
    def do_step(self, action):
        i = int(action / self.width)
        j = action % self.width
        if (self.render_flag):
            self.Render.state = self.env.state
            self.Render.draw()
            self.Render.bugfix()
        next_state, terminal, reward = self.env.choose(i, j)
        next_fog = 1 - self.env.fog
        return next_state, terminal, reward, next_fog

    ### Reward Based Epsilon Decay 
    def epsilon_update(self, avg_reward):
        if (avg_reward > self.reward_threshold):
            self.current_model.epsilon = max(self.epsilon_min, self.current_model.epsilon * self.epsilon_decay)
            self.reward_threshold += self.reward_step

    def TD_Loss(self):
        ### Samples batch from buffer memory
        state, action, mask, reward, next_state, next_mask, terminal = self.buffer.sample(self.batch_size)

        ### Converts the variabls to tensors for processing by DDQN
        state      = Variable(FloatTensor(float32(state))).cuda(self.nb_cuda)
        mask      = Variable(FloatTensor(float32(mask))).cuda(self.nb_cuda)
        next_state = FloatTensor(float32(next_state)).cuda(self.nb_cuda)
        action     = LongTensor(float32(action)).cuda(self.nb_cuda)
        next_mask      = FloatTensor(float32(next_mask)).cuda(self.nb_cuda)
        reward     = FloatTensor(reward).cuda(self.nb_cuda)
        done       = FloatTensor(terminal).cuda(self.nb_cuda)


        ### Predicts Q value for present and next state with current and target model
        q_values      = self.current_model(state,mask)
        next_q_values = self.target_model(next_state,next_mask)
        q_crt_values =  self.current_model(next_state,next_mask)
        action_max=torch.Tensor([torch.argmax(q_val) for q_val in q_crt_values]).cuda(self.nb_cuda)

        # Calculates Loss:
        #    If not Terminal:
        #        Loss = (reward + gamma*Q_val(next_state)) - Q_val(current_state)
        #    If Terminal:
        #        Loss = reward - Q_val(current_state)

        q_value          = q_values.gather(1, action.unsqueeze(1)).squeeze(1)
        next_q_value     = torch.Tensor([next_q_values[i][int(action_max[i])] for i in range(action_max.shape[0])]).cuda(self.nb_cuda)
        expected_q_value = reward + self.gamma * next_q_value * (1 - done)
        loss = (q_value - expected_q_value.detach()).pow(2).mean()
        loss_print = loss.item()   

        # Propagates the Loss
        self.optimizer.zero_grad()
        loss.backward()

        self.optimizer.step()
        self.scheduler.step()

        for target_param, local_param in zip(self.target_model.parameters(), self.current_model.parameters()):
            target_param.data.copy_(self.tau * local_param.data + (1.0 - self.tau) * target_param.data)
        return loss_print

    def save_checkpoints(self, batch_no):
        path = "./pre-trained/"+self.current_model.__class__.__name__.lower()+"_" +str(self.env.rule) + "_" + str(batch_no) + ".pth"
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


def main():
    #driver = Driver(6, 6, 6, False)
    continual_training = False
    checkpoint_number = 10000
    nb_cuda = 7
    use_lagre = False
    env_type = "default"

    driver = Driver(6, 6, 6, False,continual_training,nb_cuda,use_lagre,env_type)
    state = driver.env.state
    epochs = 100000
    save_every = 1000
    count = 0
    running_reward = 0
    batch_no = 0
    wins = 0
    total = 0

    if continual_training:
        driver.load_models(checkpoint_number)
        epochs += checkpoint_number
        batch_no += checkpoint_number

    while (batch_no < epochs):

        # simple state action reward loop and writes the actions to buffer
        mask = 1 - driver.env.fog
        action = driver.get_action(state, mask)
        next_state, terminal, reward, _ = driver.do_step(action)
        driver.buffer.push(state.flatten(), action, mask.flatten(), reward, next_state.flatten(),
                           (1 - driver.env.fog).flatten(), terminal)
        state = next_state
        count += 1
        running_reward += reward

        # Used for calculating winrate for each batch
        if (terminal):
            if (reward == 1):
                wins += 1
            driver.env.reset()
            state = driver.env.state
            mask = driver.env.fog
            total += 1

        if (count == driver.batch_size):
            # Computes the Loss
            driver.current_model.train()
            loss = driver.TD_Loss()
            driver.current_model.eval()

            # Calculates metrics
            batch_no += 1
            avg_reward = running_reward / driver.batch_size
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
            if (batch_no % save_every == 0):
                driver.save_checkpoints(batch_no)


main()
