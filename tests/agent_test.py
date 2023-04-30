import logging.config
import sys
import unittest
import time
from pathlib import Path
import os

import h5py
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

keras = tf.keras
from keras.models import load_model
from keras.callbacks import ModelCheckpoint

from dlgo.gotypes import Player
from dlgo.exp.exp_reader import ExpGenerator
from dlgo.goboard_fast import GameState
from dlgo.rl.experience import EpisodeExperienceCollector
from dlgo.tools.board_decoder import BoardDecoder
from init_ac_agent import Initiator
from self_play import SelfPlayer
from dlgo.encoders.base import get_encoder_by_name
from tests.callback_debug import DebugCallback
from train_ac import ACTrainer

logger = logging.getLogger('acTrainingLogger')


class ACAgentTest(unittest.TestCase):
    def setUp(self):
        self.board_size = 5
        self.encoder = get_encoder_by_name('simple', self.board_size)
        self.num_planes = self.encoder.num_planes
        self.num_games = 1000
        self.learning_rate = 0.001
        self.batch_size = 32
        self.project_path = Path.cwd()
        self.exp_path = self.project_path / 'exp' / f'exp_{self.num_games}_ac_agent_test.h5'
        # self.exp_path = self.project_path / 'exp' / f'exp_5000a_ac_bs5_mac_v1.h5'
        self.model_name = 'model_ac_agent_test.h5'
        self.new_model_name = 'new_model_ac_agent_test.h5'
        self.model_path = self.project_path / 'models' / self.model_name
        self.new_model_path = self.project_path / 'models' / self.new_model_name
        self.collector = EpisodeExperienceCollector(self.exp_path, self.board_size, self.num_planes)
        self.prepare_exp_by_self_play()

    def tearDown(self):
        pass

    def prepare_exp_by_self_play(self):
        initiator = Initiator(self.board_size, self.model_name)
        initiator.create_model()
        player = SelfPlayer(self.board_size, self.model_name, self.num_games)
        player.play()
        os.system('clear')

    def create_bot(self, player, num):
        ac_bot = player.create_bot(num)
        collector = EpisodeExperienceCollector(self.exp_path, self.board_size, self.num_planes)
        ac_bot.set_collector(collector)
        return ac_bot

    def make_move(self, bot, game_state, move_num):
        next_move = bot.select_move(game_state)
        print(f'NEXT MOVE: {next_move}')
        print(f'NEXT MOVE ENCODED: {self.encoder.encode_point(next_move.point)}')
        move = self.encoder.decode_point_index(self.encoder.encode_point(next_move.point))
        print(f'NEXT MOVE DECODED BACK: {move}')

        bot.collector.complete_episode(reward=1)

        with h5py.File(self.exp_path, 'r') as f:
            state = f['experience/states'][move_num]
            action = f['experience/actions'][move_num]
            reward = f['experience/rewards'][move_num]
            advantage = f['experience/advantages'][move_num]

        print(f'STATE:')
        decoder = BoardDecoder(state)
        decoder.print()
        print(f'')
        print(f'ACTION: {action}')
        move_from_exp = self.encoder.decode_point_index(action)
        print(f'ACTION DECODED: {move_from_exp}')
        print(f'REWARD: {reward}')
        print(f'ADVANTAGE: {advantage}')

        game_state = game_state.apply_move(next_move)
        return game_state, move, move_from_exp

    def test_ac_agent_writing_to_experience_correctly(self):
        print(f'=' * 40)
        print(f'>>>Testing if ACAgent writes to experience file correctly...')
        game_state = GameState.new_game(self.board_size)

        player = SelfPlayer(self.board_size, self.model_name, self.num_games)

        ac_bot1 = self.create_bot(player, 1)
        ac_bot1.collector.begin_episode()

        ac_bot2 = self.create_bot(player, 2)
        ac_bot2.collector.begin_episode()

        moves_made = []
        moves_read_from_experience = []

        for i in range(0, 10, 2):
            game_state, move_made, move_read_from_experience = self.make_move(ac_bot1, game_state, i)
            moves_made.append(move_made)
            moves_read_from_experience.append(move_read_from_experience)
            game_state, move_made, move_read_from_experience = self.make_move(ac_bot2, game_state, i + 1)
            moves_made.append(move_made)
            moves_read_from_experience.append(move_read_from_experience)

        for made, read in zip(moves_made, moves_read_from_experience):
            self.assertEqual(made, read)

    def test_ac_agent_creating_correct_new_model(self):
        trainer = ACTrainer(self.board_size, self.model_path, self.new_model_path, self.learning_rate, self.batch_size,
                            self.exp_path)
        trainer.train()

        #  a new model created through training
        model = load_model(self.new_model_path)

        self.assertEqual((None, self.board_size, self.board_size, self.num_planes), model.input_shape)
        self.assertEqual((None, self.board_size * self.board_size), model.outputs[0].shape)
        self.assertEqual((None, 1), model.outputs[1].shape)
        self.assertEqual(tf.float32, model.inputs[0].dtype)
        self.assertEqual(tf.float32, model.outputs[0].dtype)
        self.assertEqual(tf.float32, model.outputs[1].dtype)

    def test_ac_agent_creating_correct_new_model(self):
        trainer = ACTrainer(self.board_size, self.model_path, self.new_model_path, self.learning_rate, self.batch_size,
                            self.exp_path)
        trainer.train()

        #  a new model created through training
        model = load_model(self.new_model_path)

        self.assertEqual((None, self.board_size, self.board_size, self.num_planes), model.input_shape)
        self.assertEqual((None, self.board_size * self.board_size), model.outputs[0].shape)
        self.assertEqual((None, 1), model.outputs[1].shape)
        self.assertEqual(tf.float32, model.inputs[0].dtype)
        self.assertEqual(tf.float32, model.outputs[0].dtype)
        self.assertEqual(tf.float32, model.outputs[1].dtype)

    def test_ac_agent_mse(self):
        def calculate_mse_values(model, gen, steps):
            mse_values = 0.0
            num_batches = 0
            print(f'STEPS: {steps}')
            for states, targets in gen:
                actions, values = model(states)
                predicted_values = values.numpy()
                mse_values += np.mean(np.square(targets[1].numpy() - predicted_values))
                num_batches += 1
                if num_batches > steps:
                    break
            mse_values /= num_batches
            return mse_values

        initiator = Initiator(self.board_size, self.model_name)
        initiator.create_model()

        model = load_model(self.model_path)

        # trainer = ACTrainer(self.board_size, self.model_path, self.new_model_path, self.learning_rate, self.batch_size,
        #                     self.exp_path)
        # trainer.train()

        # trained_model = load_model(self.new_model_path)

        generator = ExpGenerator(self.exp_path, self.batch_size, self.num_planes, self.board_size, seed=1234)
        steps_per_epoch = len(generator)
        print(f'The length of the generator (steps per epoch) = {steps_per_epoch}')

        # Define a function to calculate MSE of the values

        # Calculate the initial MSE of the model
        print(f'>>>Calculating inital mse...')
        initial_mse = calculate_mse_values(model, generator.generate(), steps_per_epoch)

        # Train the model on the experience
        print(f'>>>Model training...')
        model.fit(generator.generate(), epochs=1, steps_per_epoch=steps_per_epoch)

        # Calculate the final MSE of the model
        print(f'>>>Calculating final mse...')
        final_mse = calculate_mse_values(model, generator.generate(), steps_per_epoch)

        # Check if the MSE has decreased
        if final_mse < initial_mse:
            print("MSE has decreased from {:.4f} to {:.4f}".format(initial_mse, final_mse))
        else:
            print("MSE has not decreased. Initial MSE: {:.4f}, Final MSE: {:.4f}".format(initial_mse, final_mse))

    def test_ac_agent_predicting_on_position(self):

        # test_board = np.array([[
        #     0, 0, 0, 0, 0, 0, 0, 0, 0,
        #     0, 0, 0, 0, 0, 0, 0, 0, 0,
        #     0, 0, 0, 0, 0, 0, 0, 0, 0,
        #     0, 1, -1, 1, -1, 0, 0, 0, 0,
        #     0, 1, -1, 1, -1, 0, 0, 0, 0,
        #     0, 0, 1, -1, 0, 0, 0, 0, 0,
        #     0, 0, 0, 0, 0, 0, 0, 0, 0,
        #     0, 0, 0, 0, 0, 0, 0, 0, 0,
        #     0, 0, 0, 0, 0, 0, 0, 0, 0,
        # ]])
        test_board = np.array([[
            0,  0,  0,  0,  0,
            0,  1, -1,  1, -1,
            0,  1, -1,  1, -1,
            0,  0,  1, -1,  0,
            0,  0,  0,  0,  0,
        ]])
        print(f'test_board[3][1]: {np.reshape(test_board, (self.board_size, self.board_size))[3][1]}')
        print(f'test_board[3][2]: {np.reshape(test_board, (self.board_size, self.board_size))[3][2]}')
        print(f'Test board\'s shape: {test_board.shape}')

        board_tensor = self.encoder.encode_tensor(test_board, Player.black)
        board_tensor = np.transpose(board_tensor, (1, 2, 0))
        print(f'')
        print(f'POSITION DECODED:')
        decoder = BoardDecoder(board_tensor)
        decoder.print()
        print(f'')
        X = np.array([board_tensor])

        initiator = Initiator(self.board_size, self.model_name)
        initiator.create_model()

        model = load_model(self.model_path)
        actions, values = model(X)
        move_probs, estimated_value, ranked_moves, ranked_moves_repeated = self.get_ac_agent_parameters(actions, values)
        self.print_out_ac_agent_parameters(actions, move_probs, estimated_value, ranked_moves, ranked_moves_repeated)

        trainer = ACTrainer(self.board_size, self.model_path, self.new_model_path, self.learning_rate, self.batch_size,
                            self.exp_path)
        trainer.train()

        #  a new model created through training
        model = load_model(self.new_model_path)
        actions, values = model(X)

        move_probs, estimated_value, ranked_moves, ranked_moves_repeated = self.get_ac_agent_parameters(actions, values)
        self.print_out_ac_agent_parameters(actions, move_probs, estimated_value, ranked_moves, ranked_moves_repeated)

    def get_ac_agent_parameters(self, actions, values):
        eps = 1e-5
        num_moves = self.board_size * self.board_size
        candidates = np.arange(num_moves)

        actions = actions.numpy()

        move_probs = actions[0]
        move_probs = np.clip(move_probs, eps, 1 - eps)
        move_probs = move_probs / np.sum(move_probs)

        estimated_value = values[0][0]
        estimated_value = estimated_value.numpy()

        ranked_moves = np.random.choice(candidates, num_moves, replace=False, p=move_probs)
        ranked_moves_repeated = np.random.choice(candidates, num_moves, replace=True, p=move_probs)
        return move_probs, estimated_value, ranked_moves, ranked_moves_repeated

    def print_out_ac_agent_parameters(self, actions, move_probs, estimated_value, ranked_moves, ranked_moves_repeated):
        print(f'')
        print(f'*' * 40)
        print(f'')
        print(f'ACTIONS:')
        print(f'{actions}')
        print(f'')
        print(f'THE SUM OF THE MOVE_PROBS:')
        x = np.sum(move_probs)
        print(f'{x}')
        print(f'')
        print(f'MOVE PROBS:')
        move_probs = np.reshape(move_probs, (self.board_size, self.board_size))
        float_formatter = "{:.4f}".format
        row = []
        for i in range(self.board_size):
            for j in range(self.board_size):
                a = float_formatter(move_probs[i][j])
                row.append(f'{a} ')
            print(f'{row}')
            row = []
        print(f'')
        print(f'THE SUM OF THE MOVE_PROBS:')
        x = np.sum(move_probs)
        print(f'{x}')
        print(f'ESTIMATED VALUE:')
        print(f'{estimated_value}')
        print(f'')
        print(f'RANKED MOVES:')
        print(f'{ranked_moves}')
        print(f'')
        print(f'')
        print(f'RANKED MOVES REPEATED:')
        print(f'{ranked_moves_repeated}')
        print(f'')
        print(f'ESTIMATED VALUE:')
        print(f'{estimated_value}')
        print(f'')


    def test_ac_agent_learning_on_correct_data(self):
        # tf.compat.v1.reset_default_graph()
        # tf.keras.backend.clear_session()
        # tf.debugging.set_log_device_placement(True)

        trainer = ACTrainer(self.board_size, self.model_path, self.new_model_path, self.learning_rate, self.batch_size,
                            self.exp_path)
        trainer.train()

        #  a new model created through training
        model = load_model(self.new_model_path)

        generator = ExpGenerator(self.exp_path, self.batch_size, self.num_planes, self.board_size, seed=1234)
        steps_per_epoch = len(generator)
        print(f'The length of the generator (steps per epoch) = {steps_per_epoch}')

        debug_callback = DebugCallback(model)  # pass model to DebugCallback constructor

        history = model.fit(
            generator.generate(),
            steps_per_epoch=steps_per_epoch,
            callbacks=[debug_callback],
            batch_size=self.batch_size,
            epochs=1
        )

        print(f'HISTORY KEYS:')
        print(f'{history.history.keys()}')
        print(f'HISTORY PARAMS:')
        print(f'{history.params}')

        print(f'METRICS NAMES: {model.metrics_names}')
        print(f'{model.metrics[0]}')
        print(f'METRIC RESULTS: {model.get_metrics_result()}')
        layer = model.get_layer(index=1)
        print(f'LAYER NAME: {layer.name}')

        num_states = generator.num_states()
        print(f'Generator num_states: {num_states}')

        def generator_method():
            for states, targets in generator.generate():
                x = tf.convert_to_tensor(states, dtype=tf.float32)
                y1 = tf.convert_to_tensor(targets[0], dtype=tf.float32)
                y2 = tf.convert_to_tensor(targets[1], dtype=tf.float32)
                yield x, (y1, y2)

        ds = tf.data.Dataset.from_generator(
            generator_method,
            output_signature=(
                tf.TensorSpec(shape=(self.batch_size, self.board_size, self.board_size, self.num_planes),
                              dtype=tf.float32),
                (
                    tf.TensorSpec(shape=(self.batch_size, self.board_size * self.board_size), dtype=tf.float32),
                    tf.TensorSpec(shape=(self.batch_size,), dtype=tf.float32)
                )
            )
        )

        for i, (x, y) in enumerate(ds):
            if i >= num_states // self.batch_size:
                break
            print(
                f'Batch {i}: states shape={x.shape}, policy target shape={y[0].shape}, value target shape={y[1].shape}'
            )
            print(f'=' * 40)
            decoder = BoardDecoder(x[500])
            decoder.print()
            print(f'Move:')
            move = np.argmax(y[0][500], axis=None, out=None)
            point = self.encoder.decode_point_index(move)
            print(point)
            print(f'')
            print(f'POLICY TARGET (THE ACTOR):')
            print(f'{y[0][500]}')
            print(f'VALUE TARGET (THE CRITIC):')
            print(f'{y[1][500]}')

        print(f'DATASET:')
        print(ds)

        # print(f'ELEMENT SPEC:')
        # print(ds.element_spec)
        #
        # for element in ds.as_numpy_iterator():
        #     print(element)
        #     break

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

        # Define a function to map the generator output to TensorFlow format
        # def map_func(x, y1, y2):
        #     # Convert the numpy arrays to TensorFlow format
        #     x = tf.convert_to_tensor(x, dtype=tf.float32)
        #     y1 = tf.convert_to_tensor(y1, dtype=tf.float32)
        #     y2 = tf.convert_to_tensor(y2, dtype=tf.float32)
        #
        #     return (x, (y1, y2))
        #
        # # Define the output signature of the dataset
        # output_signature = (
        #     tf.TensorSpec(shape=(32, 9, 9, 11), dtype=tf.float32),
        #     (
        #         tf.TensorSpec(shape=(32, 81), dtype=tf.float32),
        #         tf.TensorSpec(shape=(32,), dtype=tf.float32)
        #     )
        # )

        # print(f'DATASET:')
        # print(ds)
        #
        # example usage of the dataset
        # print(f'ELEMENT SPEC:')
        # print(ds.element_spec)
        #
        # for element in ds.as_numpy_iterator():
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
