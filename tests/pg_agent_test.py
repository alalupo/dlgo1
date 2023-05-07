import logging.config
import os
import unittest
from pathlib import Path

import h5py
import tensorflow as tf

keras = tf.keras
from keras.models import load_model

from dlgo.exp.exp_reader import ExpReader
from dlgo.goboard_fast import GameState
from exp.exp_writer import ExpWriter
from dlgo.tools.board_decoder import BoardDecoder
from dlgo.encoders.base import get_encoder_by_name
from scripts.policy_rl_trainer import RLTrainer
from scripts.simulate import Dispatcher

logger = logging.getLogger('acTrainingLogger')


class PgAgentTest(unittest.TestCase):
    def setUp(self):
        self.board_size = 19
        self.encoder = get_encoder_by_name('simple', self.board_size)
        self.num_planes = self.encoder.num_planes
        self.num_games = 5
        self.learning_rate = 0.001
        self.batch_size = 32
        self.project_path = Path(__file__).parent
        self.model_sl_name = 'model_sl_strong_improved3_10000_1_epoch1_24proc.h5'
        self.model_rl_name = self.model_sl_name.replace('sl', 'rl')
        self.model_sl_path = self.project_path / 'models' / self.model_sl_name
        self.model_rl_path = self.project_path / 'models' / self.model_rl_name
        self.exp_path = self.project_path / 'exp' / self.model_sl_name.replace('model_', 'exp_')
        self.big_file_exp_path = self.project_path / 'exp' / 'exp5_rl_bs19_1000.h5'
        self.collector = ExpWriter(str(self.exp_path), self.board_size, self.num_planes)
        self.prepare_exp_by_self_play()

    def tearDown(self):
        pass

    def prepare_exp_by_self_play(self):
        dispatcher = Dispatcher(self.board_size, self.num_games, self.model_sl_path)
        dispatcher.run_simulations()
        os.system('clear')

    def create_bot(self, dispatcher):
        bot = dispatcher.create_bot()
        collector = ExpWriter(str(self.exp_path), self.board_size, self.num_planes)
        bot.set_collector(collector)
        return bot

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

        print(f'STATE:')
        decoder = BoardDecoder(state)
        decoder.print()
        print(f'')
        print(f'ACTION: {action}')
        move_from_exp = self.encoder.decode_point_index(action)
        print(f'ACTION DECODED: {move_from_exp}')
        print(f'REWARD: {reward}')

        game_state = game_state.apply_move(next_move)
        return game_state, move, move_from_exp

    def test_pg_agent_writing_to_experience_correctly(self):
        print(f'=' * 40)
        print(f'>>>Testing if PgAgent writes to experience file correctly...')
        game_state = GameState.new_game(self.board_size)

        dispatcher = Dispatcher(self.board_size, self.num_games, self.model_sl_path)

        bot1 = self.create_bot(dispatcher)
        bot1.collector.begin_episode()

        bot2 = self.create_bot(dispatcher)
        bot2.collector.begin_episode()

        moves_made = []
        moves_read_from_experience = []

        for i in range(0, 10, 2):
            game_state, move_made, move_read_from_experience = self.make_move(bot1, game_state, i)
            moves_made.append(move_made)
            moves_read_from_experience.append(move_read_from_experience)
            game_state, move_made, move_read_from_experience = self.make_move(bot2, game_state, i + 1)
            moves_made.append(move_made)
            moves_read_from_experience.append(move_read_from_experience)

        for made, read in zip(moves_made, moves_read_from_experience):
            self.assertEqual(made, read)

    def test_pg_agent_creating_correct_new_model(self):
        with h5py.File(str(self.exp_path), 'r') as f:
            len_from_file = len(f['experience/states'])

        reader = ExpReader(str(self.exp_path), self.batch_size, self.num_planes, self.board_size, seed=1234,
                           client='pg')
        len_from_reader = reader.num_states()

        trainer = RLTrainer(str(self.model_sl_path), [str(self.exp_path)])
        trainer.train()

        #  a new model created through training
        model = load_model(self.model_rl_path)

        self.assertEqual(len_from_file, len_from_reader)
        self.assertEqual((None, self.board_size, self.board_size, self.num_planes), model.input_shape)
        self.assertEqual((None, self.board_size * self.board_size), model.outputs.shape)
        self.assertEqual(tf.float32, model.inputs[0].dtype)
        self.assertEqual(tf.float32, model.outputs[0].dtype)


if __name__ == '__main__':
    unittest.main()
