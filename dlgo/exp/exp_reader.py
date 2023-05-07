import h5py
import numpy as np
import tensorflow as tf


class ExpReader:
    def __init__(self, exp_file: str, batch_size: int, num_planes: int, board_size: int, seed=1234, client='ac'):
        self.exp_file = exp_file
        self.batch_size = batch_size
        self.num_planes = num_planes
        self.board_size = board_size
        self.num_moves = board_size * board_size
        self.seed = seed
        self.num_classes = self.board_size * self.board_size
        self.client = client

    def __len__(self):
        with h5py.File(self.exp_file, 'r') as f:
            return f['experience']['states'].shape[0] // self.batch_size

    def num_states(self):
        with h5py.File(self.exp_file, 'r') as f:
            return len(f['experience/states'])

    def generate(self):
        while True:
            if self.client == 'ac':
                for states, targets in self._generate():
                    x = tf.convert_to_tensor(states, dtype=tf.float32)
                    y1 = tf.convert_to_tensor(targets[0], dtype=tf.float32)
                    y2 = tf.convert_to_tensor(targets[1], dtype=tf.float32)
                    yield x, (y1, y2)
            elif self.client == 'pg':
                for states, targets in self._generate():
                    x = tf.convert_to_tensor(states, dtype=tf.float32)
                    y = tf.convert_to_tensor(targets, dtype=tf.float32)
                    yield x, y
            else:  # self.client == 'value'
                for states, value in self._generate():
                    x = tf.convert_to_tensor(states, dtype=tf.float32)
                    y = tf.convert_to_tensor(value, dtype=tf.float32)
                    yield x, y

    def _generate(self):
        with h5py.File(self.exp_file, 'r') as f:
            max_advantage = f['experience/max_advantage'][()]
            num_samples = self.num_states()
            indices = np.arange(num_samples)
            num_batches = num_samples // self.batch_size
            assert num_batches != 0, f"Experience database too small compared to the number of batches ({num_samples} samples)."
            for i in range(num_batches):
                batch_indices = indices[i * self.batch_size:(i + 1) * self.batch_size]
                states = f['experience/states'][batch_indices]
                states = states.astype('float32')
                ac_policy_target = np.zeros((self.batch_size, self.num_moves), dtype='float32')
                pg_policy_target = np.zeros((self.batch_size, self.num_moves), dtype='float32')
                value_target = np.zeros((self.batch_size,), dtype='float32')
                for j, idx in enumerate(batch_indices):
                    action = f['experience/actions'][idx]
                    reward = f['experience/rewards'][idx]
                    pg_policy_target[j][action] = reward
                    advantage = f['experience/advantages'][idx]
                    if max_advantage > 0:
                        ac_policy_target[j][action] = round(advantage / max_advantage, 2)
                    else:
                        ac_policy_target[j][action] = advantage
                    value_target[j] = reward
                targets = [ac_policy_target, value_target]
                if self.client == 'ac':
                    yield states, targets
                elif self.client == 'pg':
                    yield states, pg_policy_target
                else:  # self.client == 'value'
                    value_target[value_target == -1] = 0
                    yield states, value_target

