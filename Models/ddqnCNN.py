import random
from collections import deque

import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from collections import deque
import numpy as np
import random


class DDQNCNN(nn.Module):

    def __init__(self, width,height, action_dim,nb_cuda):
        super(DDQNCNN, self).__init__()
        self.epsilon = 0.99
        self.width = width
        self.height = height
        self.nb_cuda = nb_cuda
        self.feature = nn.Sequential(
            nn.Unflatten(1,(1,self.width,self.height)),
            nn.Conv2d(in_channels=1,out_channels=4,kernel_size=(3,3),stride=1,padding=1),
            nn.MaxPool2d(kernel_size=(3,3),padding=1,stride=1),
            nn.Conv2d(in_channels=4,out_channels=16,kernel_size=(3,3),stride=1,padding=1),
            nn.MaxPool2d(kernel_size=(3,3),padding=1,stride=1),
            nn.Flatten(),
            nn.Linear(16*self.width*self.height,128),
            nn.ReLU()
        )

        self.advantage = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim)
        )

        self.value = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    ### This is important, masks invalid actions
    def masked_softmax(self, vec, mask, dim=1, epsilon=1e-5):
        exps = torch.exp(vec)
        masked_exps = exps * mask.float()
        masked_sums = masked_exps.sum(dim, keepdim=True) + epsilon
        return (masked_exps / masked_sums)

    def forward(self, x, mask):
        x = x / 8
        x = self.feature(x)
        advantage = self.masked_softmax(self.advantage(x), mask)
        value = self.masked_softmax(self.value(x), mask)
        return value + advantage - advantage.mean()

    def act(self, state, mask):
        #epsilon greedy policy
        bruh = random.random()
        if bruh > self.epsilon:
            state   = torch.FloatTensor(state).unsqueeze(0).cuda(self.nb_cuda)
            mask   = torch.FloatTensor(mask).unsqueeze(0).to(self.nb_cuda)
            q_value = self.forward(state,mask)
            # print(q_value)
            action  = q_value.max(1)[1].data[0].item()
        else:
            indices = np.nonzero(mask)[0]
            randno = random.randint(0, len(indices) - 1)
            action = indices[randno]
        return action

    def load_state(self,info):
        self.load_state_dict(info['current_state_dict'])
        
class DDQNCNNL(DDQNCNN):
    def __init__(self, width, height, action_dim, nb_cuda):
        super(DDQNCNNL,self).__init__(width, height, action_dim, nb_cuda)
        self.feature = nn.Sequential(
            nn.Unflatten(1,(1,self.width,self.height)),
            nn.Conv2d(in_channels=1,out_channels=4,kernel_size=(3,3),stride=1,padding=1),
            nn.MaxPool2d(kernel_size=(3,3),padding=1,stride=1),
            nn.Conv2d(in_channels=4,out_channels=16,kernel_size=(3,3),stride=1,padding=1),
            nn.MaxPool2d(kernel_size=(3,3),padding=1,stride=1),
            nn.Conv2d(in_channels=16,out_channels=32,kernel_size=(3,3),stride=1,padding=1),
            nn.MaxPool2d(kernel_size=(3,3),padding=1,stride=1),
            nn.Conv2d(in_channels=32,out_channels=64,kernel_size=(3,3),stride=1,padding=1),
            nn.MaxPool2d(kernel_size=(3,3),padding=1,stride=1),
            nn.Flatten(),
            nn.Linear(64*self.width*self.height,256),
            nn.ReLU()
        )

        self.advantage = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim)
        )

        self.value = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )
            


class Buffer():
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, mask, reward, new_state, new_mask, terminal):
        self.buffer.append((state, action, mask, reward, new_state, new_mask, terminal))

    def sample(self, batch_size):
        states, actions, masks, rewards, new_states, new_mask, terminals = zip(*random.sample(self.buffer, batch_size))
        return states, actions, masks, rewards, new_states, new_mask, terminals
