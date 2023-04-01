import argparse
import datetime
from collections import namedtuple
import os
from pathlib import Path
import shutil

import h5py
import tensorflow as tf

from dlgo import agent
from keras.models import load_model
from dlgo import scoring
from dlgo import rl
from dlgo.goboard_fast import GameState, Player, Point
from dlgo.agent.pg import PolicyAgent
from dlgo.encoders.simple import SimpleEncoder
from dlgo.agent.pg import load_policy_agent

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
    print(game_result)

    return GameRecord(
        moves=moves,
        winner=game_result.winner,
        margin=game_result.winning_margin,
    )


class SelfPlayer:
    def __init__(self):
        self.rows, self.cols = 19, 19
        self.encoder = SimpleEncoder((self.rows, self.cols))
        self.model_dir = 'checkpoints'
        self.model_name = 'simple_small_model_epoch_12.h5'
        self.model_copy_name = 'copy_' + self.model_name
        self.model_path = self.get_model_path()

    def get_model_path(self):
        path = Path(__file__)
        project_lvl_path = path.parent
        model_dir_full_path = project_lvl_path.joinpath(self.model_dir)
        model_path = str(model_dir_full_path.joinpath(self.model_name))
        model_copy_path = str(model_dir_full_path.joinpath(self.model_copy_name))
        if not os.path.exists(model_path):
            raise FileNotFoundError
        shutil.copy(model_path, model_copy_path)
        model_path = model_copy_path
        return model_path

    def get_model(self):
        model_path = self.get_model_path()
        model_file = None
        try:
            model_file = open(model_path, 'r')
        finally:
            model_file.close()
        with h5py.File(model_path, "r") as model_file:
            model = load_model(model_file)
        return model

    def create_bot(self):
        model = self.get_model()
        # print(model.summary())
        bot = PolicyAgent(model, self.encoder)
        with h5py.File(self.model_path, "w") as model_file:
            bot.serialize(model_file)
        with h5py.File(self.model_path, "r") as model_file:
            bot_from_file = load_policy_agent(model_file)
            return bot_from_file

    def play(self):
        parser = argparse.ArgumentParser()
        parser.add_argument('--num-games', '-n', type=int, default=10)
        # parser.add_argument('--experience-out', required=True)

        print(f'Listing GPU devices:')
        print(tf.config.list_physical_devices('GPU'))

        args = parser.parse_args()
        # experience_filename = args.experience_out
        num_games = args.num_games
        global BOARD_SIZE
        BOARD_SIZE = 19

        experience_filename = f'exp{num_games}.h5'
        agent1 = self.create_bot()
        agent2 = self.create_bot()
        collector1 = rl.ExperienceCollector()
        collector2 = rl.ExperienceCollector()
        agent1.set_collector(collector1)
        agent2.set_collector(collector2)

        for i in range(num_games):
            print('Simulating game %d/%d...' % (i + 1, num_games))
            collector1.begin_episode()
            collector2.begin_episode()

            game_record = simulate_game(agent1, agent2)
            if game_record.winner == Player.black:
                collector1.complete_episode(reward=1)
                collector2.complete_episode(reward=-1)
            else:
                collector2.complete_episode(reward=1)
                collector1.complete_episode(reward=-1)

        experience = rl.combine_experience([collector1, collector2])
        with h5py.File(experience_filename, "w", rdcc_nbytes=5242880) as experience_outf:
        # with h5py.File(experience_filename, 'w') as experience_outf:
            experience.serialize(experience_outf)
            propfaid = h5py.h5p.create(h5py.h5p.FILE_ACCESS)
            settings = list(propfaid.get_cache())
            print(f'propfaid settings: {settings}')


if __name__ == '__main__':
    player = SelfPlayer()
    player.play()

