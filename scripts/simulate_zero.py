import argparse
import logging.config
import os
import sys
from collections import namedtuple
from pathlib import Path

import h5py
import numpy as np
import tensorflow as tf

keras = tf.keras
from keras.models import load_model

project_path = Path(__file__).resolve().parent.parent
os.chdir(project_path)
sys.path.append(str(project_path))
sys.path.append(str(project_path / 'dlgo'))

from dlgo import scoring
from dlgo.goboard_fast import GameState
from dlgo.gotypes import Player
from dlgo.exp.zero_exp_writer import ZeroExpWriter
from dlgo.zero.agent import ZeroAgent
from dlgo.zero.encoder import ZeroEncoder

logger = logging.getLogger('zeroTrainingLogger')


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

    model_dir_path = Path.cwd() / 'models'

    dispatcher = Dispatcher(19, num_games, model_dir_path, model_name)
    dispatcher.run_simulations()
    logger.info('SIMULATOR: Logging finished')


class GameRecord(namedtuple('GameRecord', 'moves winner margin')):
    pass


class Dispatcher:
    def __init__(self, board_size: int, num_games: int, model_dir_path: Path, model_name: str):
        self.board_size = board_size
        self.num_games = num_games
        self.model_dir_path = model_dir_path
        self.model_name = model_name
        self.model_path = Path(self.model_dir_path / self.model_name)
        self.encoder = ZeroEncoder(self.board_size)
        self.limit = 1000
        self.rounds_per_move = 100
        self.exp_paths = []

    def run_simulations(self):
        """ Optimally, thousands of games for each training batch """
        agent = self.create_bot()
        opponent = self.create_bot()

        for i in range(0, self.num_games, self.limit):
            games = min(self.num_games - i, self.limit)
            simulator = Simulator(self.board_size, games,
                                  Path(self.model_dir_path / f'exp{i / self.limit}_{self.model_name}'))
            path = simulator.build_experience(agent, opponent)
            self.exp_paths.append(path)

        print(f'>>> Experience files saved:')
        for exp in self.exp_paths:
            print(f'{exp}')

    def create_bot(self):
        print(f'>>> Creating bot for model {self.model_path}...')
        model = self.get_model()
        return ZeroAgent(model, self.encoder, rounds_per_move=self.rounds_per_move, c=2.0)

    def get_model(self):
        model_file = None
        try:
            model_file = open(str(self.model_path), 'r')
        finally:
            model_file.close()
        with h5py.File(str(self.model_path), "r") as model_file:
            model = load_model(model_file)
        return model


class Simulator:
    def __init__(self, board_size: int, num_games: int, exp_path: Path):
        self.board_size = board_size
        self.encoder = ZeroEncoder(self.board_size)
        self.num_planes = self.encoder.num_planes
        self.num_games = num_games
        self.exp_path = str(exp_path)
        cleaning(Path(self.exp_path))

    def build_experience(self, agent, opponent):

        collector1 = ZeroExpWriter(self.exp_path, self.board_size, self.num_planes)
        collector2 = ZeroExpWriter(self.exp_path, self.board_size, self.num_planes)
        agent.set_collector(collector1)
        opponent.set_collector(collector2)

        color1 = Player.black
        for i in range(self.num_games):
            print(f'Simulating game {i + 1}/{self.num_games}...')
            collector1.begin_episode()
            collector2.begin_episode()

            if color1 == Player.black:
                black_player, white_player = agent, opponent
            else:
                white_player, black_player = opponent, agent

            game_record = self.simulate_game(black_player, white_player)

            print(f'Game {i + 1} is over. Saving the episode...')
            if game_record.winner == color1:
                collector1.complete_episode(reward=1)
                collector2.complete_episode(reward=-1)
            else:
                collector2.complete_episode(reward=1)
                collector1.complete_episode(reward=-1)
            # color1 = color1.other

        print(f'>>> {self.num_games} games completed.')
        print(f'>>> {len(collector1)} states saved.')
        return self.exp_path

    def simulate_game(self, black_player, white_player):
        moves = []
        game = GameState.new_game(self.board_size)
        agents = {
            Player.black: black_player,
            Player.white: white_player,
        }
        m = 1
        while not game.is_over():
            if m % 2 == 0:
                print(f'{int(np.ceil(m / 2))} : ', end='\r')
            next_move = agents[game.next_player].select_move(game)
            moves.append(next_move)
            game = game.apply_move(next_move)
            m += 1

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
