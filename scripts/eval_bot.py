import argparse
import os
import sys
from collections import namedtuple
from pathlib import Path

import h5py

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf

keras = tf.keras
from keras.models import load_model

this_directory = os.path.dirname(__file__)
project_directory = os.path.dirname(this_directory)
sys.path.append(project_directory)
sys.path.append(this_directory)

from dlgo import scoring
from dlgo.encoders.base import get_encoder_by_name
from dlgo.goboard_fast import GameState, Player
from dlgo.zero.agent import ZeroAgent
from dlgo.zero.encoder import ZeroEncoder


class GameRecord(namedtuple('GameRecord', 'moves winner')):
    pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--board-size', '-size', type=int, default=19, required=False)
    parser.add_argument('--model1', '-m1', required=True)
    parser.add_argument('--model2', '-m2', required=True)
    parser.add_argument('--num-games', '-n', type=int, default=100, required=False)

    args = parser.parse_args()
    board_size = args.board_size
    model1 = args.model1
    model2 = args.model2
    num_games = args.num_games

    evaluator = Evaluator(board_size, model1, model2, num_games)
    evaluator.play()


class Evaluator:
    def __init__(self, board_size, model1, model2, num_games):
        self.board_size = board_size
        self.model1_path = self.get_model_path(model1)
        self.model2_path = self.get_model_path(model2)
        self.model1 = self.get_model(self.model1_path)
        self.model2 = self.get_model(self.model2_path)
        self.num_games = num_games
        self.encoder = ZeroEncoder(self.board_size)

    @staticmethod
    def get_model_path(model):
        return str(Path(project_directory) / 'models' / model)

    @staticmethod
    def get_model(model_path):
        model_file = None
        try:
            model_file = open(model_path, 'r')
        finally:
            model_file.close()
        with h5py.File(model_path, "r") as model_file:
            model = load_model(model_file)
        return model

    def play(self):

        agent1 = ZeroAgent(self.model1, self.encoder)
        agent2 = ZeroAgent(self.model2, self.encoder)

        wins = 0
        losses = 0
        color1 = Player.black
        for i in range(self.num_games):
            print(f'Playing game {i + 1} / {self.num_games}...')
            if color1 == Player.black:
                black_player, white_player = agent1, agent2
            else:
                white_player, black_player = agent1, agent2
            game_record = self.simulate_game(black_player, white_player)
            if game_record.winner == color1:
                wins += 1
                print(f'Total result so far: {wins}/{wins + losses}')
            else:
                losses += 1
            color1 = color1.other
        print(f'*' * 40)
        print(f'AGENT 1 FINAL RECORD: {wins} / {wins + losses}')
        print(f'*' * 40)

    def simulate_game(self, black_player, white_player):
        moves = []
        game = GameState.new_game(self.board_size)
        agents = {
            Player.black: black_player,
            Player.white: white_player
        }
        while not game.is_over():
            next_move = agents[game.next_player].select_move(game)
            moves.append(next_move)
            game = game.apply_move(next_move)
        game_result = scoring.compute_game_result(game)
        print(game_result)
        return GameRecord(
            moves=moves,
            winner=game_result.winner
        )


if __name__ == '__main__':
    main()
