import logging.config
import unittest
from pathlib import Path
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import h5py
import numpy as np
import tensorflow as tf

keras = tf.keras

from dlgo.exp.exp_reader import ExpReader
from exp.exp_writer import ExpWriter
from dlgo.tools.board_decoder import BoardDecoder
from init_ac_agent import Initiator
from self_play import SelfPlayer
from dlgo.encoders.base import get_encoder_by_name

logger = logging.getLogger('acTrainingLogger')


class EpisodeExperienceCollectorTest(unittest.TestCase):
    def setUp(self):
        self.board_size = 9
        self.encoder = get_encoder_by_name('simple', self.board_size)
        self.num_planes = self.encoder.num_planes
        self.num_games = 1
        self.project_path = Path.cwd()
        self.exp_path = self.project_path / 'exp' / f'exp_{self.num_games}_experience_test.h5'
        self.model_name = 'model_experience_test.h5'
        self.model_full_path = self.project_path / 'models' / self.model_name
        self.collector = ExpWriter(self.exp_path, self.board_size, self.num_planes)
        # self.clean_up()

    def tearDown(self):
        pass
        # self.clean_up()

    def clean_up(self):
        self.unlink_file(self.model_full_path)
        self.unlink_file(self.exp_path)

    def unlink_file(self, file):
        if Path(file).is_file():
            Path.unlink(file)

    def test_begin_episode(self):
        self.collector.begin_episode()
        self.assertEqual(len(self.collector._current_episode_states), 0)
        self.assertEqual(len(self.collector._current_episode_actions), 0)
        self.assertEqual(len(self.collector._current_episode_estimated_values), 0)

    def test_record_decision(self):
        state = np.zeros((self.board_size, self.board_size, self.num_planes))
        state[3, 8, 0] = 1
        action = 1
        estimated_value = 1.0
        self.collector.record_decision(state, action, estimated_value)
        self.assertEqual(len(self.collector._current_episode_states), 1)
        self.assertEqual(len(self.collector._current_episode_actions), 1)
        self.assertEqual(len(self.collector._current_episode_estimated_values), 1)

    def test_complete_episode(self):
        state = np.zeros((self.board_size, self.board_size, self.num_planes))
        state[3, 8, 0] = 1
        action = 2
        estimated_value = 1.0
        reward = 0.5
        self.collector.begin_episode()
        self.collector.record_decision(state, action, estimated_value)
        self.collector.complete_episode(reward)
        with h5py.File(self.exp_path, 'r') as f:
            self.assertIn('experience', f.keys())
            self.assertIn('states', f['experience'].keys())
            self.assertIn('actions', f['experience'].keys())
            self.assertIn('rewards', f['experience'].keys())
            self.assertIn('advantages', f['experience'].keys())

    def test_visualize_selfplay(self):
        initiator = Initiator(self.board_size, self.model_name)
        initiator.create_model()
        player = SelfPlayer(self.board_size, self.model_name, self.num_games)
        player.play()

        gen = ExpReader(self.exp_path, 32, self.num_planes, self.board_size)
        length = gen.num_states()
        print(f'The length of the experience file (the number of states/positions): {length}')
        next_batch = gen.generate()

        for i, (states, targets) in enumerate(next_batch):
            if i >= length // 32:
                break
            print(
                f'Batch {i}: states shape={states.shape}, policy target shape={targets[0].shape}, value target shape={targets[1].shape}')

    def test_visualize_experience(self):
        print(f'Collector length: {len(self.collector)}')
        with h5py.File(self.exp_path, 'r') as f:
            # two equivalent methods for acquiring the volume of the experience file
            num_states = len(f['experience/states'])
            num_states2 = f['experience']['states'].shape[0]
            print(f"There are {num_states} or {num_states2} 'states' elements in the exp1 file.")

        print(f'*' * 40)
        print(f'THE CONTENT OF THE EXP FILE:')
        print(f'*' * 40)

        gen = ExpReader(self.exp_path, 32, self.num_planes, self.board_size, seed=1234)
        length = gen.num_states()
        print(f'LENGTH: {length}')
        next_batch = gen.generate()

        for i, (states, targets) in enumerate(next_batch):
            if i >= length // 32:
                break
            print(
                f'Batch {i}: states shape={states.shape}, policy target shape={targets[0].shape}, value target shape={targets[1].shape}')
            for item in range(32):
                print(f'ITEM: {item}')
                decoder = BoardDecoder(states[item])
                decoder.print()
                print(f'')
                move = np.argmax(targets[0][item], axis=None, out=None)
                point = self.encoder.decode_point_index(move)
                # print(move)
                print(point)
                print(f'')
                print(f'POLICY TARGET (THE ACTOR):')
                print(f'{targets[0][item]}')
                print(f'VALUE TARGET (THE CRITIC):')
                print(f'{targets[1][item]}')

    def print_h5_structure(self, obj, indent=0):
        """
        Recursively prints the HDF5 file structure.
        """
        indent_str = "  " * indent
        if isinstance(obj, h5py.File):
            print(indent_str + "<File>")
            for key in obj.keys():
                self.print_h5_structure(obj[key], indent=indent + 1)
        elif isinstance(obj, h5py.Dataset):
            print(indent_str + "<Dataset: {} (dtype: {})>".format(obj.name, obj.dtype))
        elif isinstance(obj, h5py.Group):
            print(indent_str + "<Group: {}>".format(obj.name))
            for key in obj.keys():
                self.print_h5_structure(obj[key], indent=indent + 1)


if __name__ == '__main__':
    unittest.main()
