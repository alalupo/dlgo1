import logging.config
import unittest
from pathlib import Path

import h5py
import numpy as np
import tensorflow as tf

keras = tf.keras
from keras.models import load_model, save_model

from dlgo.exp.exp_reader import ExpGenerator
from dlgo.goboard_fast import GameState
from dlgo.rl.experience import EpisodeExperienceCollector
from dlgo.tools.board_decoder import BoardDecoder
from init_ac_agent import Initiator
from self_play import SelfPlayer
from dlgo.encoders.base import get_encoder_by_name
from tests.callback_debug import DebugCallback
from train_ac import ACTrainer

logging.config.fileConfig('log_confs/test_logging.conf')
logger = logging.getLogger('acTrainingLogger')


class EpisodeExperienceCollectorTest(unittest.TestCase):
    def setUp(self):
        self.board_size = 19
        self.num_planes = 11
        self.file1 = Path('tmp_experience1.h5')
        self.file2 = Path('tmp_experience2.h5')
        if Path(self.file1).is_file():
            Path.unlink(self.file1)
        if Path(self.file2).is_file():
            Path.unlink(self.file2)
        self.collector1 = EpisodeExperienceCollector(self.file1, self.board_size, self.num_planes)
        self.collector2 = EpisodeExperienceCollector(self.file2, self.board_size, self.num_planes)

    def tearDown(self):
        # Clean up the test file
        if Path(self.file1).is_file():
            Path.unlink(self.file1)
        if Path(self.file2).is_file():
            Path.unlink(self.file2)

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

    def test_selfplay(self):
        project_path = Path.cwd()
        model_dir = project_path / 'models'
        model_name = 'model_ac_test.h5'

        num_games = 1
        exp_path = project_path / 'exp' / f'exp_{num_games}_ac_test.h5'

        model_full_path = model_dir / model_name
        self.unlink_file(model_full_path)
        self.unlink_file(exp_path)

        board_size = 9
        initiator = Initiator(board_size, model_name)
        initiator.create_model()
        player = SelfPlayer(board_size, model_name, num_games)
        player.play()

        print(f'*' * 40)
        print(f'THE CONTENT OF THE EXP FILE:')
        print(f'*' * 40)

        gen = ExpGenerator(exp_path, 32, self.num_planes, board_size)
        length = gen.num_states()
        print(f'The length of the experience file (the number of states/positions): {length}')
        next_batch = gen.generate()

        for i, (states, targets) in enumerate(next_batch):
            if i >= length // 32:
                break
            print(
                f'Batch {i}: states shape={states.shape}, policy target shape={targets[0].shape}, value target shape={targets[1].shape}')

    def test_visualize_exp(self):
        project_path = Path.cwd()
        exp_path = project_path / 'exp' / 'exp_1_ac_test.h5'
        board_size = 9

        with h5py.File(exp_path, 'r') as f:
            # two equivalent methods for acquiring the volume of the experience file
            num_states = len(f['experience/states'])
            num_states2 = f['experience']['states'].shape[0]
            print(f"There are {num_states} or {num_states2} 'states' elements in the exp1 file.")

        print(f'*' * 40)
        print(f'THE CONTENT OF THE EXP FILE:')
        print(f'*' * 40)

        gen = ExpGenerator(exp_path, 32, self.num_planes, board_size, seed=1234)
        length = gen.num_states()
        print(f'LENGTH: {length}')
        next_batch = gen.generate()

        for i, (states, targets) in enumerate(next_batch):
            if i >= length // 32:
                break
            print(
                f'Batch {i}: states shape={states.shape}, policy target shape={targets[0].shape}, value target shape={targets[1].shape}')
            # for item in range(32):
            #     if item == 0:
            #         print(f'Dtypes: {states[item].dtype}, {targets[0][item].dtype}, {targets[1][item].dtype}')
            for item in range(32):
                # if item == 0:
                print(f'ITEM: {item}')
                decoder = BoardDecoder(states[item])
                decoder.print()
                print(f'')
                move = np.argmax(targets[0][item], axis=None, out=None)
                encoder = get_encoder_by_name('simple', board_size)
                point = encoder.decode_point_index(move)
                # print(move)
                print(point)
                print(f'')
                print(f'POLICY TARGET (THE ACTOR):')
                print(f'{targets[0][item]}')
                print(f'VALUE TARGET (THE CRITIC):')
                print(f'{targets[1][item]}')

    def unlink_file(self, file):
        if Path(file).is_file():
            Path.unlink(file)

    def test_pg_agent(self):
        project_path = Path.cwd()
        model_dir = project_path / 'models'
        model_name = 'model_test_pg_agent.h5'

        model_full_path = model_dir / model_name

        exp_path1 = project_path / 'exp' / 'exp_pg_agent_test1.h5'
        exp_path2 = project_path / 'exp' / 'exp_pg_agent_test2.h5'

        self.unlink_file(model_full_path)
        self.unlink_file(exp_path1)
        self.unlink_file(exp_path2)

        board_size = 9
        game_state = GameState.new_game(board_size)
        moves = []
        encoder = get_encoder_by_name('simple', board_size)

        initiator = Initiator(board_size, model_name)
        initiator.create_model()
        player = SelfPlayer(board_size, model_name, 3)

        pg_bot1 = player.create_bot(1)
        collector1 = EpisodeExperienceCollector(exp_path1, board_size, 11)
        pg_bot1.set_collector(collector1)
        collector1.begin_episode()

        next_move = pg_bot1.select_move(game_state)
        print(f'NEXT MOVE: {next_move}')
        print(f'NEXT MOVE ENCODED: {encoder.encode_point(next_move.point)}')
        first_move = encoder.decode_point_index(encoder.encode_point(next_move.point))
        print(f'NEXT MOVE DECODED BACK: {first_move}')
        collector1.complete_episode(reward=1)
        with h5py.File(exp_path1, 'r') as f:
            state = f['experience/states'][0]
            action = f['experience/actions'][0]
            reward = f['experience/rewards'][0]
            advantage = f['experience/advantages'][0]
        print(f'STATE:')
        decoder = BoardDecoder(state)
        decoder.print()
        print(f'')
        print(f'ACTION: {action}')
        first_move_from_exp = encoder.decode_point_index(action)
        print(f'ACTION DECODED: {first_move_from_exp}')
        print(f'REWARD: {reward}')
        print(f'ADVANTAGE: {advantage}')

        moves.append(next_move)
        game_state = game_state.apply_move(next_move)
        pg_bot2 = player.create_bot(2)
        collector2 = EpisodeExperienceCollector(exp_path2, board_size, 11)
        pg_bot2.set_collector(collector2)
        collector2.begin_episode()

        next_move = pg_bot2.select_move(game_state)
        print(f'NEXT MOVE: {next_move}')
        print(f'NEXT MOVE ENCODED: {encoder.encode_point(next_move.point)}')
        print(f'NEXT MOVE DECODED BACK: {encoder.decode_point_index(encoder.encode_point(next_move.point))}')
        collector2.complete_episode(reward=1)
        with h5py.File(exp_path2, 'r') as f:
            state = f['experience/states'][0]
            action = f['experience/actions'][0]
            reward = f['experience/rewards'][0]
            advantage = f['experience/advantages'][0]
        print(f'STATE:')
        decoder = BoardDecoder(state)
        decoder.print()
        print(f'')
        print(f'ACTION: {action}')
        second_move_from_exp = encoder.decode_point_index(action)
        print(f'ACTION DECODED: {second_move_from_exp}')
        print(f'REWARD: {reward}')
        print(f'ADVANTAGE: {advantage}')
        self.assertEqual(first_move, first_move_from_exp)

    def test_ac_agent(self):
        # run test_selfplay() first
        project_path = Path.cwd()
        model_name = 'model_ac_test.h5'

        exp_path = project_path / 'exp' / 'exp_1_ac_test.h5'

        board_size = 9
        game_state = GameState.new_game(board_size)
        moves = []
        encoder = get_encoder_by_name('simple', board_size)

        player = SelfPlayer(board_size, model_name, 3)

        ac_bot1 = player.create_bot(1)
        collector1 = EpisodeExperienceCollector(exp_path, board_size, 11)
        ac_bot1.set_collector(collector1)
        collector1.begin_episode()

        next_move = ac_bot1.select_move(game_state)
        print(f'NEXT MOVE: {next_move}')
        print(f'NEXT MOVE ENCODED: {encoder.encode_point(next_move.point)}')
        first_move = encoder.decode_point_index(encoder.encode_point(next_move.point))
        print(f'NEXT MOVE DECODED BACK: {first_move}')
        collector1.complete_episode(reward=1)
        with h5py.File(exp_path, 'r') as f:
            state = f['experience/states'][0]
            action = f['experience/actions'][0]
            reward = f['experience/rewards'][0]
            advantage = f['experience/advantages'][0]
        print(f'STATE:')
        decoder = BoardDecoder(state)
        decoder.print()
        print(f'')
        print(f'ACTION: {action}')
        first_move_from_exp = encoder.decode_point_index(action)
        print(f'ACTION DECODED: {first_move_from_exp}')
        print(f'REWARD: {reward}')
        print(f'ADVANTAGE: {advantage}')

        moves.append(next_move)
        game_state = game_state.apply_move(next_move)
        ac_bot2 = player.create_bot(2)
        collector2 = EpisodeExperienceCollector(exp_path, board_size, 11)
        ac_bot2.set_collector(collector2)
        collector2.begin_episode()

        next_move = ac_bot2.select_move(game_state)
        print(f'NEXT MOVE: {next_move}')
        print(f'NEXT MOVE ENCODED: {encoder.encode_point(next_move.point)}')
        print(f'NEXT MOVE DECODED BACK: {encoder.decode_point_index(encoder.encode_point(next_move.point))}')
        collector2.complete_episode(reward=1)
        with h5py.File(exp_path, 'r') as f:
            state = f['experience/states'][0]
            action = f['experience/actions'][0]
            reward = f['experience/rewards'][0]
            advantage = f['experience/advantages'][0]
        print(f'STATE:')
        decoder = BoardDecoder(state)
        decoder.print()
        print(f'')
        print(f'ACTION: {action}')
        second_move_from_exp = encoder.decode_point_index(action)
        print(f'ACTION DECODED: {second_move_from_exp}')
        print(f'REWARD: {reward}')
        print(f'ADVANTAGE: {advantage}')
        self.assertEqual(first_move, first_move_from_exp)

    def test_inputs_outputs(self):

        tf.compat.v1.reset_default_graph()
        tf.keras.backend.clear_session()
        # tf.debugging.set_log_device_placement(True)

        model_path = Path.cwd() / 'models' / 'model_ac_test.h5'
        new_model_path = Path.cwd() / 'models' / 'new_model_ac_test.h5'
        exp_path = Path.cwd() / 'exp' / 'exp_1_ac_test.h5'

        with h5py.File(exp_path, 'r') as f:
            generator = ExpGenerator(exp_path, 32, 11, 9, seed=1234)
            steps_per_epoch = len(generator)
            print(f'Len(generator) = {steps_per_epoch}')

        trainer = ACTrainer(9, model_path, new_model_path, 0.001, 32, exp_path)
        trainer.train()

        # with h5py.File(new_model_path, 'r') as f:
        #     self.print_h5_structure(f)

        model = load_model(new_model_path)

        print(f'TEST Model name: {model.name}')
        print(f'TEST Model inputs: {model.inputs}')
        print(f'TEST Model outputs: {model.outputs}')

        generator = ExpGenerator(exp_path, 32, 11, 9, seed=1234)
        steps_per_epoch = len(generator)

        debug_callback = DebugCallback(model)  # pass model to DebugCallback constructor
        history = model.fit(
            generator.generate(),
            steps_per_epoch=steps_per_epoch,
            callbacks=[debug_callback],
            batch_size=32,
            epochs=1
        )

        print(f'TEST Model name: {model.name}')
        print(f'TEST Model inputs: {model.inputs}')
        print(f'TEST Model input shape: {model.input_shape}')
        print(f'TEST Model outputs: {model.outputs}')
        # print(f'{model.summary()}')

        gen = ExpGenerator(exp_path, 32, 11, 9, seed=1234)
        length = gen.num_states()
        print(f'LENGTH: {length}')

        # def generator():
        #     for states, targets in gen.generate():
        #         yield states, targets

        def generator():
            for states, targets in gen.generate():
                x = tf.convert_to_tensor(states, dtype=tf.float32)
                y1 = tf.convert_to_tensor(targets[0], dtype=tf.float32)
                y2 = tf.convert_to_tensor(targets[1], dtype=tf.float32)
                yield x, (y1, y2)

        ds = tf.data.Dataset.from_generator(
            generator,
            output_signature=(
                tf.TensorSpec(shape=(32, 9, 9, 11), dtype=tf.float32),
                (
                    tf.TensorSpec(shape=(32, 81), dtype=tf.float32),
                    tf.TensorSpec(shape=(32,), dtype=tf.float32)
                )
            )
        )

        for i, (x, y) in enumerate(ds):
            if i >= length // 32:
                break
            print(
                f'Batch {i}: states shape={x.shape}, policy target shape={y[0].shape}, value target shape={y[1].shape}'
            )
            for item in range(32):
                # if item == 0:
                print(f'ITEM: {item}')
                decoder = BoardDecoder(x[item])
                decoder.print()
                print(f'')
                move = np.argmax(y[0][item], axis=None, out=None)
                encoder = get_encoder_by_name('simple', 9)
                point = encoder.decode_point_index(move)
                # print(move)
                print(point)
                print(f'')
                print(f'POLICY TARGET (THE ACTOR):')
                print(f'{y[0][item]}')
                print(f'VALUE TARGET (THE CRITIC):')
                print(f'{y[1][item]}')

        # new_state = [[[0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0.],
        #               [0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0.],
        #               [0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0.],
        #               [0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0.],
        #               [0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0.],
        #               [0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0.],
        #               [0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0.],
        #               [0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0.],
        #               [0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0.]],
        #
        #              [[0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0.],
        #               [0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0.],
        #               [0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0.],
        #               [0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0.],
        #               [0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0.],
        #               [0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0.],
        #               [0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0.],
        #               [0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0.],
        #               [0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0.]],
        #
        #              [[0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0.],
        #               [0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0.],
        #               [0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0.],
        #               [0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0.],
        #               [0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0.],
        #               [0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0.],
        #               [0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0.],
        #               [0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0.],
        #               [0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0.]],
        #
        #              [[0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0.],
        #               [0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0.],
        #               [0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0.],
        #               [0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0.],
        #               [0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0.],
        #               [0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0.],
        #               [0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0.],
        #               [0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0.],
        #               [0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0.]],
        #
        #              [[0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0.],
        #               [0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0.],
        #               [0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0.],
        #               [0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0.],
        #               [0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0.],
        #               [0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0.],
        #               [0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0.],
        #               [0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0.],
        #               [0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0.]],
        #
        #              [[0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0.],
        #               [0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0.],
        #               [0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0.],
        #               [0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0.],
        #               [0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0.],
        #               [0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0.],
        #               [0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0.],
        #               [0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0.],
        #               [0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0.]],
        #
        #              [[0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0.],
        #               [0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0.],
        #               [0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0.],
        #               [0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0.],
        #               [0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0.],
        #               [0., 0., 0., 1., 0., 0., 0., 0., 0., 1., 0.],
        #               [0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0.],
        #               [0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0.],
        #               [0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0.]],
        #
        #              [[0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0.],
        #               [0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0.],
        #               [0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0.],
        #               [0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0.],
        #               [0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0.],
        #               [0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0.],
        #               [0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0.],
        #               [0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0.],
        #               [0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0.]],
        #
        #              [[0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0.],
        #               [0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0.],
        #               [0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0.],
        #               [0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0.],
        #               [0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0.],
        #               [0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0.],
        #               [0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0.],
        #               [0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0.],
        #               [0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0.]]]
        #
        # new_states = np.array(new_state)
        # print(new_states.shape)

        # with h5py.File(exp_path, 'r') as f:
        #     for i in range(1):
        #         state = f['experience/states'][i]
        #         action = f['experience/actions'][i]
        #         reward = f['experience/rewards'][i]
        #         advantage = f['experience/advantages'][i]
        #         print(f'state: {state}')
        #         print(f'action: {action}')
        #         print(f'reward: {reward}')
        #         print(f'advantage: {advantage}')

        # gen = ExpGenerator(exp_path, 32, 11, 9, seed=1234)
        # length = gen.num_states()
        # print(f'LENGTH: {length}')
        # next_batch = gen.generate()
        #
        # for i, (states, targets) in enumerate(next_batch):
        #     if i >= length // 32:
        #         break
        #     print(
        #         f'Batch {i}: states shape={states.shape}, policy target shape={targets[0].shape}, value target shape={targets[1].shape}')
        #     x, y = map_func(states, targets)
        #     print(f'x.shape = {x.shape}, {x.dtype}')
        #     policy, value = y
        #     print(f'policy = {policy.shape}, {policy.dtype}')
        #     print(f'value = {value.shape}, {value.dtype}')
        #
        # # Define the output signature of the dataset
        # output_signature = (
        #     tf.TensorSpec(shape=(32, 9, 9, 11), dtype=tf.float32),
        #     (
        #         tf.TensorSpec(shape=(32, 81), dtype=tf.float32),
        #         tf.TensorSpec(shape=(32,), dtype=tf.float32)
        #     )
        # )
        #
        # # Create the dataset
        # dataset = tf.data.Dataset.from_generator(
        #     generator=generator.generate,
        #     output_signature=output_signature
        # )
        #
        # dataset = dataset.map(map_func)
        # print(f'DATASET:')
        # print(dataset)
        #
        # # example usage of the dataset
        # print(f'ELEMENT SPEC:')
        # print(dataset.element_spec)
        #
        # for element in dataset.as_numpy_iterator():
        #     print(element)

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
