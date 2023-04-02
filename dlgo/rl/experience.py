import numpy as np

__all__ = [
    'ExperienceCollector',
    'ExperienceBuffer',
    'combine_experience',
    'load_experience',
    'get_buffer'
]


class ExperienceCollector(object):
    def __init__(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.advantages = []
        self._current_episode_states = []
        self._current_episode_actions = []
        self._current_episode_estimated_values = []

    def get_buffer_data(self):
        return np.array(self.states), \
               np.array(self.actions), \
               np.array(self.rewards), \
               np.array(self.advantages)

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
        self.states += self._current_episode_states
        self.actions += self._current_episode_actions
        self.rewards += [reward for _ in range(num_states)]

        for i in range(num_states):
            advantage = reward - self._current_episode_estimated_values[i]
            self.advantages.append(advantage)

        self._current_episode_states = []
        self._current_episode_actions = []
        self._current_episode_estimated_values = []


class ExperienceBuffer(object):
    def __init__(self, states, actions, rewards, advantages):
        self.states = states
        self.actions = actions
        self.rewards = rewards
        self.advantages = advantages

    def serialize(self, h5file):
        group_name = 'experience'
        # for name in h5file:
        #     print(f'name: {name}')
        # print(f'The h5 file keys: {list(h5file.keys())}')
        if group_name in h5file.keys():
            # grp = h5file['experience']
            # print(f'grp name: {grp.name}')
            # dset1 = h5file['experience/states']
            # print(f'states shape: {dset1.shape}')
            # print(f'states.shape[0] shape: {dset1.shape[0]}')
            # print(f'self.states.shape = {self.states.shape}')
            # print(f'self.states.shape[0] = {self.states.shape[0]}')
            # print(f'states dtype: {dset1.dtype}')
            h5file['experience/states'].resize((h5file['experience/states'].shape[0] + self.states.shape[0]), axis=0)
            h5file['experience/states'][-self.states.shape[0]:] = self.states
            h5file['experience/actions'].resize((h5file['experience/actions'].shape[0] + self.actions.shape[0]), axis=0)
            h5file['experience/actions'][-self.actions.shape[0]:] = self.actions
            h5file['experience/rewards'].resize((h5file['experience/rewards'].shape[0] + self.rewards.shape[0]), axis=0)
            h5file['experience/rewards'][-self.rewards.shape[0]:] = self.rewards
            h5file['experience/advantages'].resize(
                (h5file['experience/advantages'].shape[0] + self.advantages.shape[0]), axis=0)
            h5file['experience/advantages'][-self.advantages.shape[0]:] = self.advantages
            # print(f'>>>The resized datasets:')
            # print(f'states shape: {dset1.shape}')
            # print(f'states dtype: {dset1.dtype}')
            # dset2 = h5file['experience/actions']
            # print(f'actions shape: {dset2.shape}')
            # print(f'actions dtype: {dset2.dtype}')
            # dset3 = h5file['experience/rewards']
            # print(f'rewards shape: {dset3.shape}')
            # print(f'rewards dtype: {dset3.dtype}')
            # dset4 = h5file['experience/advantages']
            # print(f'advantages shape: {dset4.shape}')
            # print(f'advantages dtype: {dset4.dtype}')
        else:
            h5file.create_group('experience')
            h5file['experience'].create_dataset('states', data=self.states, maxshape=(None, 19, 19, 11))
            h5file['experience'].create_dataset('actions', data=self.actions, maxshape=(None,))
            h5file['experience'].create_dataset('rewards', data=self.rewards, maxshape=(None,))
            h5file['experience'].create_dataset('advantages', data=self.advantages, maxshape=(None,))
            # print(f'>>>The freshly created datasets:')
            # dset1 = h5file['experience/states']
            # print(f'states shape: {dset1.shape}')
            # print(f'states dtype: {dset1.dtype}')
            # dset2 = h5file['experience/actions']
            # print(f'actions shape: {dset2.shape}')
            # print(f'actions dtype: {dset2.dtype}')
            # dset3 = h5file['experience/rewards']
            # print(f'rewards shape: {dset3.shape}')
            # print(f'rewards dtype: {dset3.dtype}')
            # dset4 = h5file['experience/advantages']
            # print(f'advantages shape: {dset4.shape}')
            # print(f'advantages dtype: {dset4.dtype}')


def combine_experience(collectors):
    print(f'>>Concatenating states...')
    print(f'Collectors\' shapes: {[np.array(c.states).shape for c in collectors]}')
    print(f'Collectors\' combined_states size: {[round(np.array(c.states).nbytes / 1000000, 2) for c in collectors]}')
    combined_states = np.concatenate([np.array(c.states) for c in collectors])
    print(f'The shape of the concatenated array: {combined_states.shape}')
    print(f'The concatenated array size: {round(combined_states.nbytes / 1000000, 2)}')

    print(f'>>Concatenating actions...')
    print(f'Collectors\' combined_actions size: {[round(np.array(c.actions).nbytes / 1000000, 2) for c in collectors]}')
    combined_actions = np.concatenate([np.array(c.actions) for c in collectors])

    print(f'>>Concatenating rewards...')
    print(f'Collectors\' combined_rewards size: {[round(np.array(c.rewards).nbytes / 1000000, 2) for c in collectors]}')
    combined_rewards = np.concatenate([np.array(c.rewards) for c in collectors])

    print(f'>>Combining advantages...')
    print(
        f'Collectors\' combined_advantages size: {[round(np.array(c.advantages).nbytes / 1000000, 2) for c in collectors]}')
    combined_advantages = np.concatenate([np.array(c.advantages) for c in collectors])

    print(f'>>Returning buffer...')
    return ExperienceBuffer(
        combined_states,
        combined_actions,
        combined_rewards,
        combined_advantages)


def get_buffer(collector):
    states, actions, rewards, advantages = collector.get_buffer_data()
    return ExperienceBuffer(
        states,
        actions,
        rewards,
        advantages)


def load_experience(h5file):
    return ExperienceBuffer(
        states=np.array(h5file['experience']['states']),
        actions=np.array(h5file['experience']['actions']),
        rewards=np.array(h5file['experience']['rewards']),
        advantages=np.array(h5file['experience']['advantages']))
