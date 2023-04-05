import argparse
import os
import logging
from collections import namedtuple

import h5py
import tensorflow as tf
from keras.models import load_model

from dlgo import rl
from dlgo import scoring
from dlgo.agent.pg import PolicyAgent
from dlgo.agent.pg import load_policy_agent
from dlgo.encoders.simple import SimpleEncoder
from dlgo.goboard_fast import GameState, Player, Point
from dlgo.tools.file_finder import FileFinder

COLS = 'ABCDEFGHJKLMNOPQRST'
STONE_TO_CHAR = {
    None: '.',
    Player.black: 'x',
    Player.white: 'o',
}


def avg(items):
    if not items:
        return 0.0
    return sum(items) / float(len(items))


def print_board(board):
    for row in range(BOARD_SIZE, 0, -1):
        line = []
        for col in range(1, BOARD_SIZE + 1):
            stone = board.get(Point(row=row, col=col))
            line.append(STONE_TO_CHAR[stone])
        print('%2d %s' % (row, ''.join(line)))
    print('   ' + COLS[:BOARD_SIZE])


class GameRecord(namedtuple('GameRecord', 'moves winner margin')):
    pass


def name(player):
    if player == Player.black:
        return 'B'
    return 'W'


def simulate_game(black_player, white_player):
    moves = []
    game = GameState.new_game(BOARD_SIZE)
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
    logging.info(game_result)

    return GameRecord(
        moves=moves,
        winner=game_result.winner,
        margin=game_result.winning_margin,
    )


def main():
    logging.basicConfig(filename='self_play.log', format='%(levelname)s:%(message)s', level=logging.INFO)
    logging.info('Started')
    player = SelfPlayer()
    player.play()
    logging.info('Finished')


class SelfPlayer:
    def __init__(self):
        self.rows, self.cols = 19, 19
        self.encoder = SimpleEncoder((self.rows, self.cols))
        self.model_name = 'model_simple_small_1000_20_epoch12_10proc.h5'
        # SelfPlayer creates two copies of existing model, one for each agent,
        # but it uses the same name and path for both copies
        self.model_copy_path = self.get_model_copy_path()
        self.exp_name = self.get_exp_name()
        self.exp_path = self.get_exp_path()

    def get_model_copy_path(self):
        finder = FileFinder()
        copy_name = finder.get_new_prefix_name_from_model(self.model_name, 'copy_')
        return finder.get_model_full_path(copy_name)

    def get_exp_name(self):
        finder = FileFinder()
        return finder.get_new_prefix_name_from_model(self.model_name, 'exp_')

    def get_exp_path(self):
        finder = FileFinder()
        return finder.get_exp_full_path(self.exp_name)

    def play(self):
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
        logging.info(f'>>>Starting self play of {self.model_name}')
        parser = argparse.ArgumentParser()
        parser.add_argument('--num-games', '-n', type=int, default=10)
        # parser.add_argument('--experience-out', '-e', required=True)

        logging.info(f'Listing GPU devices:')
        logging.info(tf.config.list_physical_devices('GPU'))

        args = parser.parse_args()
        num_games = args.num_games
        # experience_filename = args.experience_out

        global BOARD_SIZE
        BOARD_SIZE = 19

        agent1 = self.create_bot(1)
        agent2 = self.create_bot(2)
        collector1 = rl.EpExperienceCollector(self.exp_path)
        collector2 = rl.EpExperienceCollector(self.exp_path)
        agent1.set_collector(collector1)
        agent2.set_collector(collector2)

        for i in range(num_games):
            logging.info('Simulating game %d/%d...' % (i + 1, num_games))
            collector1.begin_episode()
            collector2.begin_episode()
            game_record = simulate_game(agent1, agent2)
            logging.info(f'>>>Completing episodes...')
            if game_record.winner == Player.black:
                collector1.complete_episode(reward=1)
                collector2.complete_episode(reward=-1)
            else:
                collector2.complete_episode(reward=1)
                collector1.complete_episode(reward=-1)
        logging.info(f'>>> Done')

    def create_bot(self, number):
        logging.info(f'>>>Creating bot {number}...')
        model = self.get_model()
        # print(model.summary())
        bot = PolicyAgent(model, self.encoder)
        with h5py.File(self.model_copy_path, "w") as model_file:
            bot.serialize(model_file)
        with h5py.File(self.model_copy_path, "r") as model_file:
            bot_from_file = load_policy_agent(model_file)
            return bot_from_file

    def get_model(self):
        model_copy = self.get_model_copy()
        model_file = None
        try:
            model_file = open(model_copy, 'r')
        finally:
            model_file.close()
        with h5py.File(model_copy, "r") as model_file:
            model = load_model(model_file)
        return model

    def get_model_copy(self):
        finder = FileFinder()
        return finder.copy_model_and_get_path(self.model_name)


if __name__ == '__main__':
    main()
