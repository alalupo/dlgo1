import unittest
import numpy as np
from pathlib import Path

from dlgo.gotypes import Player, Point
from dlgo.goboard_fast import GameState, Board, Move
from dlgo.tools.board_decoder import BoardDecoder
from dlgo.encoders.base import get_encoder_by_name


class LoadingTest(unittest.TestCase):
    def setUp(self):
        self.board_size = 19
        self.encoder = get_encoder_by_name('simple', self.board_size)

    def test_validation(self):
        game_state = GameState.new_game(self.board_size)
        point = Point(16, 16)
        move = Move.play(point)
        game_state = game_state.apply_move(move)
        board_tensor = self.encoder.encode(game_state)
        print(f'')
        decoder = BoardDecoder(board_tensor)
        decoder.print()
        print(f'')
        legal_moves = game_state.legal_moves()
        all_moves = [Move.play(self.encoder.decode_point_index(i)) for i in range(361)]
        illegal_moves = [x for x in all_moves if x not in legal_moves]
        print(f'Illegal moves: ')
        for move in illegal_moves:
            print(move)
        self.assertEqual(1, len(illegal_moves))

