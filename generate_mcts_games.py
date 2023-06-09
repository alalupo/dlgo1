import argparse
import numpy as np

from dlgo.encoders.base import get_encoder_by_name
from dlgo import goboard_fast as goboard
from dlgo import mcts
from dlgo.utils import print_board, print_move


def data_tmp():
    board_size = 9
    max_moves = 5
    boards, moves = [], []
    encoder = get_encoder_by_name('oneplane', board_size)
    game = goboard.GameState.new_game(board_size)
    bot = mcts.MCTSAgent(num_rounds=1, temperature=0.8)
    num_moves = 0
    print(f'encoder.num_points()={encoder.num_points()}')
    while not game.is_over():
        print_board(game.board)
        move = bot.select_move(game)
        if move.is_play:
            boards.append(encoder.encode(game))
            print(f'boards={boards}')
            move_one_hot = np.zeros(encoder.num_points())
            move_one_hot[encoder.encode_point(move.point)] = 1
            print(f'move one hot={move_one_hot}')
            moves.append(move_one_hot)
            print(f'moves={moves}')
        print_move(game.next_player, move)
        game = game.apply_move(move)
        num_moves += 1
        if num_moves > max_moves:
            break
    print(f'boards.len={len(boards)}')
    print(f'moves.len={len(moves)}')
    print(f'moves[0].len={len(moves[0])}')
    nda_boards = np.array(boards)
    nda_moves = np.array(moves)
    print(f'boards ndim={nda_boards.ndim}')
    print(f'boards shape={nda_boards.shape}')
    print(f'moves ndim={nda_moves.ndim}')
    print(f'moves shape={nda_moves.shape}')
    # print(f'boards[0]={boards[0]}')
    # print(f'boards[1]={boards[1]}')
    # print(f'boards[2]={boards[2]}')
    # print(f'boards[3]={boards[3]}')
    print(f'boards[3][0]={boards[3][0]}')


def generate_game(board_size, rounds, max_moves, temperature):
    boards, moves = [], []
    encoder = get_encoder_by_name('oneplane', board_size)
    game = goboard.GameState.new_game(board_size)
    bot = mcts.MCTSAgent(rounds, temperature)
    num_moves = 0
    while not game.is_over():
        print_board(game.board)
        move = bot.select_move(game)
        if move.is_play:
            boards.append(encoder.encode(game))
            move_one_hot = np.zeros(encoder.num_points())
            move_one_hot[encoder.encode_point(move.point)] = 1
            moves.append(move_one_hot)
        print_move(game.next_player, move)
        game = game.apply_move(move)
        num_moves += 1
        if num_moves > max_moves:
            break
    return np.array(boards), np.array(moves)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--board-size', '-b', type=int, default=9)
    parser.add_argument('--rounds', '-r', type=int, default=1000)
    parser.add_argument('--temperature', '-t', type=float, default=0.8)
    parser.add_argument('--max-moves', '-m', type=int, default=60, help='Max moves per game.')
    parser.add_argument('--num-games', '-n', type=int, default=10)
    parser.add_argument('--board-out')
    parser.add_argument('--move-out')

    args = parser.parse_args()
    xs = []
    ys = []

    for i in range(args.num_games):
        print(f'Generating game {i+1}/{args.num_games}...')
        x, y = generate_game(args.board_size, args.rounds, args.max_moves, args.temperature)
        xs.append(x)
        ys.append(y)

    x = np.concatenate(xs)
    y = np.concatenate(ys)
    np.save(args.board_out, x)
    np.save(args.move_out, y)


if __name__ == '__main__':
    main()
