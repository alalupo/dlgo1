import numpy as np
import tensorflow as tf
keras = tf.keras
from keras.models import load_model
from keras.optimizers import SGD

from dlgo.goboard_fast import Move
from dlgo.agent.base import Agent
from dlgo.agent.helpers import is_point_an_eye
# from dlgo.tools.board_decoder import BoardDecoder

__all__ = [
    'PolicyAgent'
]


def normalize(x):
    total = np.sum(x)
    return x / total


def prepare_experience_data(experience, board_width, board_height):
    experience_size = experience.actions.shape[0]
    target_vectors = np.zeros((experience_size, board_width * board_height))
    for i in range(experience_size):
        action = experience.actions[i]
        reward = experience.rewards[i]
        target_vectors[i][action] = reward
    return target_vectors


class PolicyAgent(Agent):
    """Policy gradient learning agent."""

    def __init__(self, model, encoder):
        super().__init__()
        self.model = model
        self._encoder = encoder
        self._collector = None
        self._temperature = 0.0
        self.last_state_value = 0

    def diagnostics(self):
        return {'value': self.last_state_value}

    def set_temperature(self, temperature):
        self._temperature = temperature

    def set_collector(self, collector):
        self._collector = collector

    def predict_move(self, game_state, board_tensor):
        X = np.array([board_tensor])
        return self.model.predict([X], verbose=0)[0][0]

    def select_move(self, game_state):
        num_moves = self._encoder.board_width * self._encoder.board_height
        board_tensor = self._encoder.encode(game_state)
        board_tensor = np.transpose(board_tensor, (1, 2, 0))
        if np.random.random() < self._temperature:
            # Explore random moves.
            move_probs = np.ones(num_moves) / num_moves
        else:
            move_probs = self.predict_move(game_state, board_tensor)
        # Prevent move probs from getting stuck at 0 or 1.
        eps = 1e-5
        move_probs = np.clip(move_probs, eps, 1 - eps)
        # Re-normalize to get another probability distribution.
        move_probs = move_probs / np.sum(move_probs)

        # Turn the probabilities into a ranked list of moves.
        candidates = np.arange(num_moves)
        ranked_moves = np.random.choice(
            candidates, num_moves, replace=False, p=move_probs)
        for point_idx in ranked_moves:
            point = self._encoder.decode_point_index(point_idx)
            if not game_state.is_valid_move(Move.play(point)):
                continue
            if not is_point_an_eye(game_state.board, point, game_state.next_player):
                if self._collector is not None:
                    self._collector.record_decision(state=board_tensor, action=point_idx)
                    # decoder = BoardDecoder(board_tensor)
                    # decoder.print()
                    # print(f'')
                    # print(f'point_idx: {self._encoder.decode_point_index(point_idx)}')
                    return Move.play(point)
        # No legal, non-self-destructive moves less.
        return Move.pass_turn()

    def train(self, experience, lr, clipnorm, batch_size):
        self.model.compile(
            loss='categorical_crossentropy',
            optimizer=SGD(lr=lr, clipnorm=clipnorm))

        target_vectors = prepare_experience_data(
            experience,
            self._encoder.board_width,
            self._encoder.board_height)

        self.model.fit(
            experience.states, target_vectors,
            batch_size=batch_size,
            epochs=1)

