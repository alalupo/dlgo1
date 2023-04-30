import numpy as np

from dlgo.encoders.base import Encoder
from dlgo.goboard_fast import Move, Board, GoString
from dlgo.gotypes import Player, Point
from dlgo.networks import network_types


class SimpleEncoder(Encoder):
    def __init__(self, board_size=(19, 19)):
        self.board_width, self.board_height = board_size[0], board_size[1]
        # The function of all 11 planes:
        #   0 - 3. black stones with 1, 2, 3, 4+ liberties
        #   4 - 7. white stones with 1, 2, 3, 4+ liberties
        #   8. black plays next
        #   9. white plays next
        #   10. move would be illegal due to ko
        self.num_planes = 11
        self.channels_sequence = network_types.channels()

    def name(self):
        return 'simple'

    def encode(self, game_state):
        board_tensor = np.zeros(self.shape())
        if game_state.next_player == Player.black:
            board_tensor[8] = 1
        else:
            board_tensor[9] = 1
        for r in range(self.board_height):
            for c in range(self.board_width):
                p = Point(row=r + 1, col=c + 1)
                go_string = game_state.board.get_go_string(p)

                if go_string is None:
                    if game_state.does_move_violate_ko(game_state.next_player,
                                                       Move.play(p)):
                        board_tensor[10][r][c] = 1
                else:
                    liberty_plane = min(4, go_string.num_liberties) - 1
                    if go_string.color == Player.white:
                        liberty_plane += 4
                    board_tensor[liberty_plane][r][c] = 1

        # transposing to adjust to channels_last tensorflow format
        # return np.transpose(board_tensor, (1, 2, 0))
        return board_tensor

    def encode_point(self, point):
        """Turn a board point into an integer index."""
        # Points are 1-indexed
        return self.board_width * (point.row - 1) + (point.col - 1)

    def decode_point_index(self, index):
        """Turn an integer index into a board point."""
        row = index // self.board_width
        col = index % self.board_width
        return Point(row=row + 1, col=col + 1)

    def num_points(self):
        return self.board_width * self.board_height

    def shape(self):
        return self.num_planes, self.board_width, self.board_height

    def shape_for_keras(self):
        # to adjust to channels_last tensorflow format
        return self.board_width, self.board_height, self.num_planes

    def encode_tensor(self, tensor, player):
        if not isinstance(tensor, np.ndarray) or not tensor.shape == (1, self.board_width * self.board_height):
            ValueError(f'The tensor should be a numpy array whose the expected shape is: (1, board-size * board-size)')
        if not isinstance(player, Player):
            ValueError(f'The player should be a Player object')
        reshaped_tensor = np.reshape(tensor, (self.board_width, self.board_height))
        board_tensor = np.zeros(self.shape())
        if player == Player.black:
            board_tensor[8] = 1
        else:
            board_tensor[9] = 1
        board = Board(self.board_width, self.board_height)
        for r in range(self.board_height):
            for c in range(self.board_width):
                r1 = self.turn_row_upside_down(r)
                if reshaped_tensor[r][c] == 1:
                    p = Point(row=r1 + 1, col=c + 1)
                    board.place_stone(Player.black, p)
                if reshaped_tensor[r][c] == -1:
                    p = Point(row=r1 + 1, col=c + 1)
                    board.place_stone(Player.white, p)
        for r in range(self.board_height):
            for c in range(self.board_width):
                r1 = self.turn_row_upside_down(r)
                if not reshaped_tensor[r][c] == 0:
                    p = Point(row=r1 + 1, col=c + 1)
                    go_string = board.get_go_string(p)
                    liberty_plane = min(4, go_string.num_liberties) - 1
                    if go_string.color == Player.white:
                        liberty_plane += 4
                    board_tensor[liberty_plane][r1][c] = 1
        return board_tensor

    def turn_row_upside_down(self, row):
        return self.board_width - 1 - row

def create(board_size):
    return SimpleEncoder(board_size)
