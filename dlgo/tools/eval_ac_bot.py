import argparse
import h5py
from collections import namedtuple
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
keras = tf.keras
from keras.models import load_model

from dlgo.tools.file_finder import FileFinder
from dlgo.rl.ac import ACAgent
from dlgo import scoring
from dlgo.encoders.base import get_encoder_by_name
from dlgo.goboard_fast import GameState, Player


class GameRecord(namedtuple('GameRecord', 'moves winner')):
    pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--board-size', '-size', type=int, required=True)
    parser.add_argument('--model1', required=True)
    parser.add_argument('--model2', required=True)
    parser.add_argument('--num-games', '-n', type=int, default=10)

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
        self.encoder = get_encoder_by_name('simple', self.board_size)

    @staticmethod
    def get_model_path(model):
        finder = FileFinder()
        return finder.get_model_full_path(model)

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

        agent1 = ACAgent(self.model1, self.encoder)
        agent2 = ACAgent(self.model2, self.encoder)

        wins = 0
        losses = 0
        color1 = Player.black
        for i in range(self.num_games):
            print(f'Simulating game{i + 1}/{self.num_games}...')
            if color1 == Player.black:
                black_player, white_player = agent1, agent2
            else:
                white_player, black_player = agent1, agent2
            game_record = self.simulate_game(black_player, white_player)
            if game_record.winner == color1:
                wins += 1
            else:
                losses += 1
            color1 = color1.other
        print(f'Agent 1 record: {wins}/{wins + losses}')

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
