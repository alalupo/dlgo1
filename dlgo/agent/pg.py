"""Policy gradient learning."""
import numpy as np
import gc
import tensorflow as tf
from keras import backend as K
from keras.optimizers import SGD

from dlgo.agent.base import Agent
from dlgo.agent.helpers import is_point_an_eye
from dlgo import encoders
from dlgo.encoders.base import get_encoder_by_name
from dlgo import goboard
from dlgo import kerasutil

__all__ = [
    'PolicyAgent',
    'load_policy_agent',
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
    """An agent that uses a deep policy network to select moves."""

    def __init__(self, model, encoder):
        super().__init__()
        self._model = model
        self._encoder = encoder
        self._collector = None
        self._temperature = 0.0
        self.last_state_value = 0

    def diagnostics(self):
        return {'value': self.last_state_value}

    def collector_size(self):
        self._collector.show_size()

    def set_temperature(self, temperature):
        self._temperature = temperature

    def set_collector(self, collector):
        self._collector = collector

    def select_move(self, game_state):
        num_moves = self._encoder.board_width * self._encoder.board_height
        board_tensor = self._encoder.encode(game_state)
        X = np.array([board_tensor])
        # X shape for SimpleEncoder = (1, 11, 19, 19)
        # the conversion below due to: https://github.com/tensorflow/tensorflow/issues/44711#issuecomment-724439274
        X = tf.convert_to_tensor(X)
        if np.random.random() < self._temperature:
            # Explore random moves.
            move_probs = np.ones(num_moves) / num_moves
        else:
            # Follow our current policy.
            # move_probs = self._model.predict(X, verbose=0)[0]
            # the line above was changed due to:
            # https://github.com/tensorflow/tensorflow/issues/44711#issuecomment-1280844213
            move_probs = self._model(X, training=False).numpy()[0]
            # memory leak counter:
            gc.collect()
            # move_probs shape = (361, )
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
            if not game_state.is_valid_move(goboard.Move.play(point)):
                continue
            if not is_point_an_eye(game_state.board, point, game_state.next_player):
                if self._collector is not None:
                    self._collector.record_decision(state=board_tensor, action=point_idx)
                    return goboard.Move.play(point)
        # No legal, non-self-destructive moves less.
        return goboard.Move.pass_turn()

    def serialize(self, h5file):
        h5file.create_group('encoder')
        h5file['encoder'].attrs['name'] = self._encoder.name()
        h5file['encoder'].attrs['board_width'] = self._encoder.board_width
        h5file['encoder'].attrs['board_height'] = self._encoder.board_height
        h5file.create_group('model')
        kerasutil.save_model_to_hdf5_group(self._model, h5file['model'])

    def train(self, experience, lr, clipnorm, batch_size):
        self._model.compile(
            loss='categorical_crossentropy',
            optimizer=SGD(lr=lr, clipnorm=clipnorm))

        target_vectors = prepare_experience_data(
            experience,
            self._encoder.board_width,
            self._encoder.board_height)

        self._model.fit(
            experience.states, target_vectors,
            batch_size=batch_size,
            epochs=1)


def load_policy_agent(h5file):
    model = kerasutil.load_model_from_hdf5_group(h5file['model'])
    encoder_name = h5file['encoder'].attrs['name']
    if not isinstance(encoder_name, str):
        encoder_name = encoder_name.decode('ascii')
    board_width = h5file['encoder'].attrs['board_width']
    board_height = h5file['encoder'].attrs['board_height']
    encoder = get_encoder_by_name(
        encoder_name,
        (board_width, board_height))
    return PolicyAgent(model, encoder)
