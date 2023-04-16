import h5py
import numpy as np


class ExpGenerator:
    def __init__(self, exp_file, batch_size, num_planes, board_size):
        self.exp_file = exp_file
        self.batch_size = batch_size
        self.num_planes = num_planes
        self.board_size = board_size
        self.num_moves = board_size * board_size
        self.length = self.length()

    def __len__(self):
        return self.length

    def length(self):
        with h5py.File(self.exp_file, 'r') as f:
            return int(f['experience']['states'].shape[0] / self.batch_size)

    def next(self):
        return self.__next__()

    def __next__(self):
        with h5py.File(self.exp_file, 'r') as f:
            states = np.zeros((self.batch_size, self.board_size, self.board_size, self.num_planes))
            policy_target = np.zeros((self.batch_size, self.num_moves))
            value_target = np.zeros((self.batch_size,))
            for i in range(self.batch_size):
                state = f['experience/states'][i]
                action = f['experience/actions'][i]
                reward = f['experience/rewards'][i]
                advantage = f['experience/advantages'][i]
                states[i] = state
                policy_target[i][action] = advantage
                value_target[i] = reward
            targets = [policy_target, value_target]
            yield states, targets


