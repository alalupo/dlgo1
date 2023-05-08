import h5py
import numpy as np
import tensorflow as tf


class ZeroExpReader:
    def __init__(self, exp_file: str, batch_size: int, seed=1234):
        self.exp_file = exp_file
        self.batch_size = batch_size
        self.seed = seed

    def __len__(self):
        with h5py.File(self.exp_file, 'r') as f:
            return f['experience']['states'].shape[0] // self.batch_size

    def num_states(self):
        with h5py.File(self.exp_file, 'r') as f:
            return len(f['experience/states'])

    def generate(self):
        while True:
            for states, targets in self._generate():
                x = tf.convert_to_tensor(states, dtype=tf.float32)
                y1 = tf.convert_to_tensor(targets[0], dtype=tf.float32)
                y2 = tf.convert_to_tensor(targets[1], dtype=tf.float32)
                yield x, (y1, y2)

    def _generate(self):
        with h5py.File(self.exp_file, 'r') as f:
            num_samples = self.num_states()
            indices = np.arange(num_samples)
            num_batches = num_samples // self.batch_size
            assert num_batches != 0, f"Experience database too small compared to the number of batches ({num_samples} samples)."
            for i in range(num_batches):
                batch_indices = indices[i * self.batch_size:(i + 1) * self.batch_size]

                states = f['experience/states'][batch_indices]
                states = states.astype('float32')

                visits = f['experience/visits'][batch_indices]
                visits = visits.astype('float32')
                visits_sums = np.sum(visits, axis=1)
                print(f'visits_sum: {visits_sums}')
                visits_sums = visits_sums.reshape(self.batch_size, 1)
                action_target = visits / visits_sums

                value_target = f['experience/rewards'][batch_indices]
                value_target = value_target.astype('float32')

                targets = [action_target, value_target]
                yield states, targets

