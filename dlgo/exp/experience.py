import logging

import h5py
import numpy as np

__all__ = [
    'EpisodeExperienceCollector',
]


class EpisodeExperienceCollector(object):
    def __init__(self, h5file, board_size, num_planes):
        self.str_h5file = h5file
        self.board_size = board_size
        self.num_planes = num_planes
        self._current_episode_states = []
        self._current_episode_actions = []
        self._current_episode_estimated_values = []
        self.max_advantage = 0.0

    def __len__(self):
        with h5py.File(self.str_h5file, "r") as f:
            return int(f['experience/states'].shape[0])

    @staticmethod
    def get_buffer_data(states, actions, rewards, advantages):
        return np.array(states), np.array(actions), np.array(rewards), np.array(advantages)

    def show_size(self):
        logging.info(f'EXPERIENCE COLLECTOR: {round(np.array(self._current_episode_states).nbytes / 1000000, 2)} MB')

    def begin_episode(self):
        self._current_episode_states = []
        self._current_episode_actions = []
        self._current_episode_estimated_values = []

    def record_decision(self, state, action, estimated_value=0):
        self._current_episode_states.append(state)
        self._current_episode_actions.append(action)
        self._current_episode_estimated_values.append(estimated_value)

    def complete_episode(self, reward):
        num_states = len(self._current_episode_states)
        logging.info(f'EXPERIENCE COLLECTOR: num_states in current episode: {num_states}')
        rewards = [reward for _ in range(num_states)]
        advantages = []
        for i in range(num_states):
            advantage = round(reward - self._current_episode_estimated_values[i], 2)
            if advantage > self.max_advantage:
                self.max_advantage = advantage
            advantages.append(advantage)

        states, actions, rewards, advantages = self.get_buffer_data(self._current_episode_states,
                                                                    self._current_episode_actions,
                                                                    rewards,
                                                                    advantages)
        self.show_size()
        self._current_episode_states = []
        self._current_episode_actions = []
        self._current_episode_estimated_values = []
        with h5py.File(self.str_h5file, "a") as f:
            self.serialize(f, states, actions, rewards, advantages)

    def serialize(self, h5file, states, actions, rewards, advantages):
        group_name = 'experience'
        if group_name in h5file.keys():
            h5file['experience/states'].resize((h5file['experience/states'].shape[0] + states.shape[0]), axis=0)
            h5file['experience/states'][-states.shape[0]:] = states
            h5file['experience/actions'].resize((h5file['experience/actions'].shape[0] + actions.shape[0]), axis=0)
            h5file['experience/actions'][-actions.shape[0]:] = actions
            h5file['experience/rewards'].resize((h5file['experience/rewards'].shape[0] + rewards.shape[0]), axis=0)
            h5file['experience/rewards'][-rewards.shape[0]:] = rewards
            h5file['experience/advantages'].resize(
                (h5file['experience/advantages'].shape[0] + advantages.shape[0]), axis=0)
            h5file['experience/advantages'][-advantages.shape[0]:] = advantages
            if 'max_advantage' in h5file['experience'].keys():
                saved_max_advantage = h5file['experience/max_advantage'][()]
                if saved_max_advantage > self.max_advantage:
                    h5file['experience/max_advantage'][()] = saved_max_advantage
                else:
                    h5file['experience/max_advantage'][()] = self.max_advantage
            else:
                h5file['experience'].create_dataset(name='max_advantage', data=self.max_advantage, dtype='float32')
                saved_max_advantage = h5file['experience/max_advantage'][()]
                if saved_max_advantage > self.max_advantage:
                    h5file['experience/max_advantage'][()] = saved_max_advantage
                else:
                    h5file['experience/max_advantage'][()] = self.max_advantage


        else:
            h5file.create_group('experience')
            h5file['experience'].create_dataset(
                name='states',
                data=states,
                maxshape=(None, self.board_size, self.board_size, self.num_planes))
            h5file['experience'].create_dataset(name='actions', data=actions, maxshape=(None,))
            h5file['experience'].create_dataset(name='rewards', data=rewards, maxshape=(None,))
            h5file['experience'].create_dataset(name='advantages', data=advantages, maxshape=(None,))
            h5file['experience'].create_dataset(name='max_advantage', data=self.max_advantage, dtype='float32')
