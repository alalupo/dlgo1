import argparse
import logging.config
from collections import namedtuple
from pathlib import Path
import sys
import os

import h5py
import tensorflow as tf

keras = tf.keras
from keras.models import load_model

project_path = Path(__file__).resolve().parent.parent
os.chdir(project_path)
sys.path.append(str(project_path))
sys.path.append(str(Path.cwd() / 'dlgo'))

from dlgo import scoring
from dlgo.goboard_fast import GameState
from dlgo.gotypes import Player
from dlgo.exp.exp_writer import ExpWriter
from dlgo.agent.pg import PolicyAgent
from dlgo.encoders.base import get_encoder_by_name

logger = logging.getLogger('selfplayLogger')


def cleaning(file):
    if Path(file).is_file():
        Path.unlink(file)


def main():
    logger.info('SIMULATOR: Logging started')

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', '-m', required=True)
    parser.add_argument('--num-games', '-n', type=int, default=10)

    args = parser.parse_args()
    model_name = args.model
    num_games = args.num_games

    logger.info(f'MODEL NAME: {model_name}')
    logger.info(f'GAMES: {num_games}')

    dispatcher = Dispatcher(19, num_games, model_name)
    dispatcher.run_simulations()
    logger.info('SIMULATOR: Logging finished')


class GameRecord(namedtuple('GameRecord', 'moves winner margin')):
    pass


class Dispatcher:
    def __init__(self, board_size, num_games, model_name):
        self.board_size = board_size
        self.num_games = num_games
        self.model_name = model_name
        self.model_dir = Path.cwd() / 'models'
        self.encoder = get_encoder_by_name('simple', self.board_size)
        self.limit = 1000
        self.exp_paths = []

    def run_simulations(self):
        rl_agent = self.create_bot(self.model_name)
        opponent = self.create_bot(self.model_name)

        for i in range(0, self.num_games, self.limit):
            games = min(self.num_games - i, self.limit)
            simulator = Simulator(self.board_size, games)
            path = simulator.build_experience(rl_agent, opponent)
            self.exp_paths.append(path)

        print(f'>>> Experience files saved:')
        for exp in self.exp_paths:
            print(f'{exp}')

    def create_bot(self, model_name):
        print(f'>>>Creating bot {model_name}...')
        path = str(self.model_dir / model_name)
        model = self.get_model(path)
        return PolicyAgent(model, self.encoder)

    def get_model(self, model_path):
        model_file = None
        try:
            model_file = open(model_path, 'r')
        finally:
            model_file.close()
        with h5py.File(model_path, "r") as model_file:
            model = load_model(model_file)
        return model


class Simulator:
    def __init__(self, board_size, num_games):
        self.board_size = board_size
        self.encoder = get_encoder_by_name('simple', self.board_size)
        self.num_planes = self.encoder.num_planes
        self.num_games = num_games
        self.exp_name = f'exp_rl_bs{self.board_size}_{num_games}.h5'
        self.exp_path = str(Path.cwd() / 'exp' / self.exp_name)
        cleaning(Path.cwd() / 'exp' / self.exp_name)

    def build_experience(self, rl_agent, opponent):

        collector1 = ExpWriter(self.exp_path, self.board_size, self.num_planes)
        collector2 = ExpWriter(self.exp_path, self.board_size, self.num_planes)
        rl_agent.set_collector(collector1)
        opponent.set_collector(collector2)

        color1 = Player.black
        for i in range(self.num_games):
            print(f'Simulating game {i + 1}/{self.num_games}...')
            collector1.begin_episode()
            collector2.begin_episode()

            if color1 == Player.black:
                black_player, white_player = rl_agent, opponent
            else:
                white_player, black_player = opponent, rl_agent

            game_record = self.simulate_game(black_player, white_player)

            print(f'Game {i + 1} is over. Saving the episode...')
            if game_record.winner == color1:
                collector1.complete_episode(reward=1)
                collector2.complete_episode(reward=-1)
            else:
                collector2.complete_episode(reward=1)
                collector1.complete_episode(reward=-1)
            color1 = color1.other

        print(f'>>> {self.num_games} games completed.')
        return self.exp_path

    def simulate_game(self, black_player, white_player):
        moves = []
        game = GameState.new_game(self.board_size)
        agents = {
            Player.black: black_player,
            Player.white: white_player,
        }
        while not game.is_over():
            next_move = agents[game.next_player].select_move(game)
            moves.append(next_move)
            game = game.apply_move(next_move)

        # print_board(game.board)
        game_result = scoring.compute_game_result(game)
        print(f'GAME RESULT: {game_result}')

        return GameRecord(
            moves=moves,
            winner=game_result.winner,
            margin=game_result.winning_margin,
        )


if __name__ == '__main__':
    main()
