import argparse
import h5py
from collections import namedtuple

from keras.models import load_model

from dlgo.rl.ac import ACAgent
from dlgo import scoring
from dlgo.encoders.base import get_encoder_by_name
from dlgo.goboard_fast import GameState, Player

BOARD_SIZE = 19


class GameRecord(namedtuple('GameRecord', 'moves winner')):
    pass


def get_model(model_path):
    try:
        model_file = open(model_path, 'r')
    finally:
        model_file.close()
    with h5py.File(model_path, "r") as model_file:
        model = load_model(model_file)
    return model


def simulate_game(black_player, white_player):
    moves = []
    game = GameState.new_game(BOARD_SIZE)
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model1', required=True)
    parser.add_argument('--model2', required=True)
    parser.add_argument('--num-games', '-n', type=int, default=10)

    args = parser.parse_args()
    model1 = get_model(args.model1)
    model2 = get_model(args.model2)
    num_games = args.num_games

    encoder = get_encoder_by_name('simple', 9)
    agent1 = ACAgent(model1, encoder)
    agent2 = ACAgent(model2, encoder)

    wins = 0
    losses = 0
    color1 = Player.black
    for i in range(num_games):
        print('Simulating game %d/%d...' % (i + 1, num_games))
        if color1 == Player.black:
            black_player, white_player = agent1, agent2
        else:
            white_player, black_player = agent1, agent2
        game_record = simulate_game(black_player, white_player)
        if game_record.winner == color1:
            wins += 1
        else:
            losses += 1
        color1 = color1.other
    print('Agent 1 record: %d/%d' % (wins, wins + losses))


if __name__ == '__main__':
    main()
