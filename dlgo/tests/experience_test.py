import unittest
import pathlib
import numpy as np
import h5py

from dlgo.rl.experience import EpisodeExperienceCollector


class EpisodeExperienceCollectorTest(unittest.TestCase):
    def setUp(self):
        self.board_size = 19
        self.num_planes = 11
        self.file1 = pathlib.Path('tmp_experience1.h5')
        self.file2 = pathlib.Path('tmp_experience2.h5')
        if pathlib.Path(self.file1).is_file():
            pathlib.Path.unlink(self.file1)
        if pathlib.Path(self.file2).is_file():
            pathlib.Path.unlink(self.file2)
        self.collector1 = EpisodeExperienceCollector(self.file1, self.board_size, self.num_planes)
        self.collector2 = EpisodeExperienceCollector(self.file2, self.board_size, self.num_planes)

    def tearDown(self):
        # Clean up the test file
        if pathlib.Path(self.file1).is_file():
            pathlib.Path.unlink(self.file1)
        if pathlib.Path(self.file2).is_file():
            pathlib.Path.unlink(self.file2)

    def test_begin_episode(self):
        self.collector1.begin_episode()
        self.assertEqual(len(self.collector1._current_episode_states), 0)
        self.assertEqual(len(self.collector1._current_episode_actions), 0)
        self.assertEqual(len(self.collector1._current_episode_estimated_values), 0)

    def test_record_decision(self):
        state = np.zeros((self.board_size, self.board_size, self.num_planes))
        state[3, 15, 0] = 1
        action = 1
        estimated_value = 1.0
        self.collector1.record_decision(state, action, estimated_value)
        self.assertEqual(len(self.collector1._current_episode_states), 1)
        self.assertEqual(len(self.collector1._current_episode_actions), 1)
        self.assertEqual(len(self.collector1._current_episode_estimated_values), 1)

    def test_complete_episode(self):
        state = np.zeros((self.board_size, self.board_size, self.num_planes))
        state[3, 15, 0] = 1
        action = 2
        estimated_value = 1.0
        reward = 0.5
        self.collector1.begin_episode()
        self.collector1.record_decision(state, action, estimated_value)
        self.collector1.complete_episode(reward)
        with h5py.File(self.file1, 'r') as f:
            self.assertIn('experience', f.keys())
            self.assertIn('states', f['experience'].keys())
            self.assertIn('actions', f['experience'].keys())
            self.assertIn('rewards', f['experience'].keys())
            self.assertIn('advantages', f['experience'].keys())

    def test_combine_experience(self):
        state = np.zeros((self.board_size, self.board_size, self.num_planes))
        state[3, 15, 0] = 1
        action = 1
        estimated_value = 1.0
        reward = 0.5
        self.collector1.begin_episode()
        self.collector1.record_decision(state, action, estimated_value)
        self.collector1.complete_episode(reward)
        state = np.zeros((self.board_size, self.board_size, self.num_planes))
        state[3, 3, 0] = 1
        action = 2
        estimated_value = 1.1
        reward = 0.7
        self.collector2.begin_episode()
        self.collector2.record_decision(state, action, estimated_value)
        self.collector2.complete_episode(reward)
        combined_states, combined_actions, combined_rewards, combined_advantages = \
            self.collector1.combine_experience([self.collector1, self.collector2])

        self.assertTrue(np.all(combined_actions == np.array([1, 2])))
        self.assertTrue(np.all(combined_rewards == np.array([0.5, 0.7])))
        self.assertTrue(np.all(combined_advantages == np.array([-0.5, -0.4])))


