import numpy as np

from dlgo.rl.experience import EpisodeExperienceCollector


class ExpGenerator:
    def __init__(self, exp_file, batch_size, num_planes):
        self.exp_file = exp_file
        self.batch_size = batch_size
        self.num_planes = num_planes
        self.collector = EpisodeExperienceCollector(self.exp_file, self.num_planes)
        self.__batch_index = 0

    def __len__(self):
        # steps_per_epoch, if unspecified, will use the len(generator) as a number of steps.
        # hence this
        return len(self.collector) / self.batch_size

    def next(self):
        return self.__next__()

    def __next__(self):
        for i in range(0, len(self.collector), self.batch_size):
            start = i
            stop = i + self.batch_size
            states = np.array(self.exp_file['experience']['states'][start:stop]),
            actions = np.array(self.exp_file['experience']['actions'][start:stop]),
            rewards = np.array(self.exp_file['experience']['rewards'][start:stop]),
            advantages = np.array(self.exp_file['experience']['advantages'][start:stop])
            yield states, actions, rewards, advantages
