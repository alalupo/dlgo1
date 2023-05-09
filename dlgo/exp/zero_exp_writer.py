import logging

import h5py
import numpy as np


class ZeroExpWriter:
    def __init__(self, h5file: str, board_size: int, num_planes: int):
        self.h5file = h5file
        self.board_size = board_size
        self.num_planes = num_planes

        self.states = []
        self.visit_counts = []
        self.rewards = []
        self._current_episode_states = []
        self._current_episode_visit_counts = []

    def __len__(self):
        with h5py.File(self.h5file, "r") as f:
            return int(f['experience/states'].shape[0])

    @staticmethod
    def get_buffer_data(states, visits, rewards):
        return np.array(states), np.array(visits), np.array(rewards)

    def show_size(self):
        logging.info(
            f'EXPERIENCE COLLECTOR\'S SIZE: {round(np.array(self._current_episode_states).nbytes / 1000000, 2)} MB')

    def begin_episode(self):
        self._current_episode_states = []
        self._current_episode_visit_counts = []

    def record_decision(self, state, visit_counts):
        self._current_episode_states.append(state)
        self._current_episode_visit_counts.append(visit_counts)

    def complete_episode(self, reward):
        num_states = len(self._current_episode_states)
        logging.info(f'EXPERIENCE COLLECTOR: num_states in current episode: {num_states}')
        self.states += self._current_episode_states
        self.visit_counts += self._current_episode_visit_counts
        self.rewards += [reward for _ in range(num_states)]

        states, visits, rewards = self.get_buffer_data(self._current_episode_states,
                                                       self._current_episode_visit_counts,
                                                       self.rewards)
        self.show_size()
        self._current_episode_states = []
        self._current_episode_visit_counts = []
        with h5py.File(self.h5file, "a") as f:
            self.serialize(f, states, visits, rewards)

    def serialize(self, h5file, states, visits, rewards):
        group_name = 'experience'
        if group_name in h5file.keys():
            h5file['experience/states'].resize((h5file['experience/states'].shape[0] + states.shape[0]), axis=0)
            h5file['experience/states'][-states.shape[0]:] = states
            h5file['experience/visits'].resize((h5file['experience/visits'].shape[0] + visits.shape[0]), axis=0)
            h5file['experience/visits'][-visits.shape[0]:] = visits
            h5file['experience/rewards'].resize((h5file['experience/rewards'].shape[0] + rewards.shape[0]), axis=0)
            h5file['experience/rewards'][-rewards.shape[0]:] = rewards
        else:
            h5file.create_group('experience')
            h5file['experience'].create_dataset(
                name='states',
                data=states,
                maxshape=(None, self.board_size, self.board_size, self.num_planes))
            h5file['experience'].create_dataset(
                name='visits',
                data=visits,
                maxshape=(None, self.board_size * self.board_size + 1))
            h5file['experience'].create_dataset(name='rewards', data=rewards, maxshape=(None,))
