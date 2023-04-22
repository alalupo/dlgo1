import argparse
import logging.config
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from collections import namedtuple

import h5py
import tensorflow as tf
keras = tf.keras
from keras.models import load_model
from keras.optimizers import SGD

from dlgo import rl
from dlgo import scoring
from dlgo.agent.pg import PolicyAgent
from dlgo.encoders.base import get_encoder_by_name
from dlgo.goboard_fast import GameState
from dlgo.gotypes import Player
from dlgo.tools.file_finder import FileFinder
from dlgo.utils import print_board

logging.config.fileConfig('log_confs/selfplay_logging.conf')
logger = logging.getLogger('selfplayLogger')


class GameRecord(namedtuple('GameRecord', 'moves winner margin')):
    pass


def main():
    logger.info('SELF PLAY: Logging started')

    parser = argparse.ArgumentParser()
    parser.add_argument('--board-size', '-size', type=int, required=True)
    parser.add_argument('--learning-model', '-model', required=True)
    parser.add_argument('--num-games', '-n', type=int, default=10)

    args = parser.parse_args()
    board_size = args.board_size
    model = args.learning_model
    num_games = args.num_games

    logger.info(f'MODEL NAME: {model}')
    logger.info(f'GAMES: {num_games}')
    logger.info(f'BOARD SIZE: {board_size}')

    player = SelfPlayer(board_size, model, num_games)
    player.play()
    logger.info('SELF PLAY: Logging finished')


class SelfPlayer:
    def __init__(self, board_size, model, num_games):
        self.board_size = board_size
        self.rows, self.cols = self.board_size, self.board_size
        self.encoder = get_encoder_by_name('simple', self.board_size)
        self.model_path = self.get_model_path(model)
        self.num_games = num_games
        self.exp_name = self.get_exp_name(model)
        self.exp_path = self.get_exp_path()
        logger.info(f'=== NEW SelfPlay OBJECT CREATED ===')
        logger.info(f'ENCODER: {self.encoder.name()}')

    @staticmethod
    def get_model_path(model):
        finder = FileFinder()
        return finder.get_model_full_path(model)

    def get_exp_name(self, model):
        finder = FileFinder()
        name = finder.get_new_prefix_name_from_model(model, 'exp_')
        return f'{name}_{self.board_size}_{self.num_games}'

    def get_exp_path(self):
        finder = FileFinder()
        return finder.get_exp_full_path(self.exp_name)

    def play(self):
        print(f'>>>Creating two bots from the model: {self.model_path}')
        agent1 = self.create_bot(1)
        agent2 = self.create_bot(2)
        collector1 = rl.EpisodeExperienceCollector(self.exp_path, self.board_size, self.encoder.num_planes)
        collector2 = rl.EpisodeExperienceCollector(self.exp_path, self.board_size, self.encoder.num_planes)
        agent1.set_collector(collector1)
        agent2.set_collector(collector2)

        for i in range(self.num_games):
            print(f'Simulating game {i+1}/{self.num_games}...')
            collector1.begin_episode()
            collector2.begin_episode()
            game_record = self.simulate_game(agent1, agent2, self.board_size)
            print(f'Game {i+1} is over. Saving the episode...')
            if game_record.winner == Player.black:
                collector1.complete_episode(reward=1)
                collector2.complete_episode(reward=-1)
            else:
                collector2.complete_episode(reward=1)
                collector1.complete_episode(reward=-1)
        print(f'>>> {self.num_games} games completed.')

    def create_bot(self, number):
        print(f'>>>Creating bot {number}...')
        model = self.get_model()
        return PolicyAgent(model, self.encoder)

    def get_model(self):
        model_file = None
        try:
            model_file = open(self.model_path, 'r')
        finally:
            model_file.close()
        with h5py.File(self.model_path, "r") as model_file:
            model = load_model(model_file)
            model.compile(
                loss='categorical_crossentropy',
                optimizer=SGD(learning_rate=0.0001, clipnorm=1.0))
        return model

    @staticmethod
    def simulate_game(black_player, white_player, board_size):
        moves = []
        game = GameState.new_game(board_size)
        agents = {
            Player.black: black_player,
            Player.white: white_player,
        }
        while not game.is_over():
            next_move = agents[game.next_player].select_move(game)
            moves.append(next_move)
            game = game.apply_move(next_move)

        print_board(game.board)
        game_result = scoring.compute_game_result(game)
        logger.info(game_result)

        return GameRecord(
            moves=moves,
            winner=game_result.winner,
            margin=game_result.winning_margin,
        )


if __name__ == '__main__':
    main()
