import numpy as np

import tensorflow as tf
keras = tf.keras
from keras.metrics import Precision, Recall

from dlgo.agent.base import Agent
from dlgo.agent.helpers import is_point_an_eye
from dlgo.encoders import base
from dlgo import goboard_fast


class DeepLearningAgent(Agent):
    def __init__(self, model, encoder):
        Agent.__init__(self)
        self.model = model
        self.model.summary()
        self.encoder = encoder
        self.last_state_value = 0

    def predict(self, game_state):
        board_tensor = self.encoder.encode(game_state)
        board_tensor = np.transpose(board_tensor, (1, 2, 0))
        X = np.array([board_tensor])

        # Create a TensorFlow function to get the output of the last layer
        # last_layer_output = self.model.layers[-1].output
        # get_output = tf.keras.backend.function([self.model.input], [last_layer_output])
        # Call the function to get the output tensor
        # move_probs = get_output([X])[0][0]

        move_probs = self.model([X])[0].numpy()
        return move_probs

    def select_move(self, game_state):
        num_moves = self.encoder.board_width * self.encoder.board_height
        move_probs = self.predict(game_state)
        # num_moves = self.encoder.board_width * self.encoder.board_height
        # board_tensor = self.encoder.encode(game_state)
        # board_tensor = np.transpose(board_tensor, (1, 2, 0))
        # X = np.array([board_tensor])
        # move_probs = self.model.predict(X)[0]
        move_probs = move_probs ** 3
        eps = 1e-6
        move_probs = np.clip(move_probs, eps, 1 - eps)
        move_probs = move_probs / np.sum(move_probs)
        candidates = np.arange(num_moves)
        ranked_moves = np.random.choice(candidates, num_moves, replace=False, p=move_probs)
        for point_idx in ranked_moves:
            point = self.encoder.decode_point_index(point_idx)
            if not game_state.is_valid_move(goboard_fast.Move.play(point)):
                # print(f'is {point} valid? => {game_state.is_valid_move(goboard_fast.Move.play(point))}')
                continue
            if not is_point_an_eye(game_state.board, point, game_state.next_player):
                return goboard_fast.Move.play(point)
            # print(f'is {point} an eye? => {is_point_an_eye(game_state.board, point, game_state.next_player)}')
            return goboard_fast.Move.pass_turn()
        return goboard_fast.Move.pass_turn()

    def diagnostics(self):
        return {'value': self.last_state_value}


