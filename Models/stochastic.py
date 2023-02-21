import random


class STOCHASTIC:
    def act(self, state, mask):
        return random.choice([i for i in range(len(mask)) if mask[i] == 1])
