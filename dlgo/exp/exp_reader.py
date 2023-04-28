import h5py
import numpy as np
import tensorflow as tf
keras = tf.keras


class ExpGenerator:
    def __init__(self, exp_file, batch_size, num_planes, board_size, seed=None):
        self.exp_file = exp_file
        self.batch_size = batch_size
        self.num_planes = num_planes
        self.board_size = board_size
        self.num_moves = board_size * board_size
        self.length = self._length()
        self.seed = seed
        self.num_classes = self.board_size * self.board_size

    def __len__(self):
        return self._length()

    def _length(self):
        with h5py.File(self.exp_file, 'r') as f:
            return int(f['experience']['states'].shape[0] / self.batch_size)

    def num_states(self):
        with h5py.File(self.exp_file, 'r') as f:
            return len(f['experience/states'])

    # def generate(self):
    #     while True:
    #         for item in self._generate():
    #             yield item

    def generate(self):
        while True:
            for states, targets in self._generate():
                x = tf.convert_to_tensor(states, dtype=tf.float32)
                y1 = tf.convert_to_tensor(targets[0], dtype=tf.float32)
                y2 = tf.convert_to_tensor(targets[1], dtype=tf.float32)
                yield x, (y1, y2)

    def _generate(self):
        with h5py.File(self.exp_file, 'r') as f:
            max_advantage = f['experience/max_advantage'][()]
            num_samples = self.num_states()
            indices = np.arange(num_samples)
            num_batches = num_samples // self.batch_size
            for i in range(num_batches):
                batch_indices = indices[i * self.batch_size:(i + 1) * self.batch_size]
                states = f['experience/states'][batch_indices]
                states = states.astype('float32')
                policy_target = np.zeros((self.batch_size, self.num_moves), dtype='float32')
                value_target = np.zeros((self.batch_size,), dtype='float32')
                for j, idx in enumerate(batch_indices):
                    action = f['experience/actions'][idx]
                    reward = f['experience/rewards'][idx]
                    advantage = f['experience/advantages'][idx]
                    if max_advantage > 0:
                        policy_target[j][action] = advantage / max_advantage
                    else:
                        policy_target[j][action] = advantage
                    value_target[j] = reward
                targets = [policy_target, value_target]
                yield states, targets



