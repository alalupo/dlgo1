import numpy as np

from dlgo.goboard_fast import Board
from dlgo.gotypes import Player, Point
from dlgo.utils import print_board


class BoardDecoder:
    def __init__(self, board_tensor):
        if board_tensor.shape == (19, 19, 11):
            self.board_tensor = np.transpose(board_tensor, (2, 1, 0))
        else:
            self.board_tensor = np.asarray(board_tensor)
        self.board_size = self.board_tensor.shape[1]
        self.map = np.zeros((self.board_size, self.board_size))
        self.set_map()
        self.board = Board(self.board_size, self.board_size)
        self.place_stones()

    def set_map(self):
        for i in range(4):
            for c in range(self.board_size):
                for r in range(self.board_size):
                    if self.board_tensor[i][c][r] == 1:
                        self.map[c][r] = 1
        for i in range(4, 8):
            for c in range(self.board_size):
                for r in range(self.board_size):
                    if self.board_tensor[i][r][c] == 1:
                        self.map[r][c] = -1

    def place_stones(self):
        for r in range(self.board_size):
            for c in range(self.board_size):
                if self.map[c][r] == 1:
                    self.board.place_stone(Player.black, Point(r + 1, c + 1))
                if self.map[c][r] == -1:
                    self.board.place_stone(Player.white, Point(r + 1, c + 1))

    def print(self):
        print_board(self.board)
