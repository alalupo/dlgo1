# This scripts demonstrates all the steps to create and train an
# AGZ-style bot.
# For practical purposes, you would separate this script into multiple
# parts (for initializing, generating self-play games, and training).
# You'll also need to run for many more rounds.

import argparse
import os
import sys
from pathlib import Path

import h5py
import tensorflow as tf

keras = tf.keras
from keras.models import Model, save_model, load_model

this_directory = os.path.dirname(__file__)
project_directory = os.path.dirname(this_directory)
sys.path.append(project_directory)
sys.path.append(this_directory)

from dlgo import scoring
from dlgo import zero
from dlgo.goboard_fast import GameState, Player
from dlgo.networks.network_architectures import Network
from dlgo.utils import print_board


def simulate_game(board_size, black_agent, black_collector, white_agent, white_collector):
    game = GameState.new_game(board_size)
    agents = {
        Player.black: black_agent,
        Player.white: white_agent,
    }
    black_collector.begin_episode()
    white_collector.begin_episode()
    while not game.is_over():
        next_move = agents[game.next_player].select_move(game)
        game = game.apply_move(next_move)

    game_result = scoring.compute_game_result(game)
    print_board(game.board)
    # print(game_result)
    # Give the reward to the right agent.
    if game_result.winner == Player.black:
        black_collector.complete_episode(1)
        white_collector.complete_episode(-1)
    else:
        black_collector.complete_episode(-1)
        white_collector.complete_episode(1)


def save_trained_model(model, board_size, batches):
    path = str(Path(project_directory) / 'models' / f'model_zero_{board_size}_{batches}.h5')
    with h5py.File(path, 'w') as f:
        save_model(model=model, filepath=f, save_format='h5')
    print(f'>>> Zero style model has been saved.')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num-batches', '-b', type=int, default=2)

    args = parser.parse_args()
    num_batches = args.num_batches

    # Initialize a zero agent
    board_size = 19
    # Optimally, a few hundred rounds per move
    rounds = 10
    num_games = 2
    encoder = zero.ZeroEncoder(board_size)
    network = Network(board_size)

    model = Model(
        inputs=[network.board_input],
        outputs=[network.policy_output, network.value_output])

    black_agent = zero.ZeroAgent(model, encoder, rounds_per_move=rounds, c=2.0)
    white_agent = zero.ZeroAgent(model, encoder, rounds_per_move=rounds, c=2.0)

    for i in range(num_batches):
        print(f'>>> The batch {i + 1}/{num_batches}')
        if i != 0:
            path = str(Path(project_directory) / 'models' / f'model_zero_{board_size}_{num_batches}.h5')
            trained_model = load_model(path)
            black_agent = zero.ZeroAgent(trained_model, encoder, rounds_per_move=rounds, c=2.0)
            white_agent = zero.ZeroAgent(trained_model, encoder, rounds_per_move=rounds, c=2.0)
        c1 = zero.ZeroExperienceCollector()
        c2 = zero.ZeroExperienceCollector()
        black_agent.set_collector(c1)
        white_agent.set_collector(c2)

        # Optimally, thousands of games for each training batch.
        for j in range(num_games):
            print(f'>>> Starting the game {j + 1}/{num_games}')
            simulate_game(board_size, black_agent, c1, white_agent, c2)

        exp = zero.combine_experience([c1, c2])
        black_agent.train(exp, 0.01, 2048)
        save_trained_model(black_agent.model, board_size, num_batches)


if __name__ == '__main__':
    main()
