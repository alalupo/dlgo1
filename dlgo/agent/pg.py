import logging.config

import numpy as np
import tensorflow as tf

keras = tf.keras
from keras.optimizers import SGD
from keras.models import Model

from dlgo.goboard_fast import Move
from dlgo.agent.base import Agent
from dlgo.agent.helpers import is_point_an_eye
from dlgo.encoders.simple import SimpleEncoder

# from dlgo.tools.board_decoder import BoardDecoder

__all__ = [
    'PolicyAgent'
]

logger = logging.getLogger('acTrainingLogger')


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

    def __init__(self, model: Model, encoder: SimpleEncoder):
        super().__init__()
        self.model = model
        self.encoder = encoder
        self.collector = None
        self.temperature = 0.0
        self.last_state_value = 0

    def diagnostics(self):
        return {'value': self.last_state_value}

    def set_temperature(self, temperature):
        self.temperature = temperature

    def set_collector(self, collector):
        self.collector = collector

    def predict(self, game_state):
        board_tensor = self.encoder.encode(game_state)
        board_tensor = np.transpose(board_tensor, (1, 2, 0))
        X = np.array([board_tensor])
        # return self.model.predict(input_tensor)[0]
        return self.model(X)[0].numpy()

    def select_move(self, game_state):
        num_moves = self.encoder.board_width * self.encoder.board_height
        board_tensor = self.encoder.encode(game_state)
        board_tensor = np.transpose(board_tensor, (1, 2, 0))
        X = np.array([board_tensor])
        if np.random.random() < self.temperature:
            # Explore random moves.
            move_probs = np.ones(num_moves) / num_moves
        else:
            # Follow our current policy.
            # move_probs = self.model.predict(X, verbose=0)[0]
            move_probs = self.model(X)[0]
            move_probs = move_probs.numpy()

        # Prevent move probs from getting stuck at 0 or 1.
        eps = 3e-5
        move_probs = np.clip(move_probs, eps, 1 - eps)
        # Re-normalize to get another probability distribution.
        move_probs = move_probs / np.sum(move_probs)

        # Turn the probabilities into a ranked list of moves.
        candidates = np.arange(num_moves)
        ranked_moves = np.random.choice(candidates, num_moves, replace=False, p=move_probs)
        for point_idx in ranked_moves:
            point = self.encoder.decode_point_index(point_idx)
            move = Move.play(point)
            move_is_valid = game_state.is_valid_move(move)
            fills_own_eye = is_point_an_eye(game_state.board, point, game_state.next_player)
            if not move_is_valid:
                continue
            if fills_own_eye:
                continue
            if self.collector is not None:
                self.collector.record_decision(state=board_tensor, action=point_idx)
                # decoder = BoardDecoder(board_tensor)
                # decoder.print()
                # print(f'')
                # print(f'point_idx: {self._encoder.decode_point_index(point_idx)}')
            return Move.play(point)
        # No legal, non-self-destructive moves less.
        return Move.pass_turn()

    def train(self, gen, lr, clipnorm, batch_size):

        opt = tf.keras.optimizers.Adam(learning_rate=lr, clipnorm=clipnorm)
        self.model.compile(optimizer=opt, loss='categorical_crossentropy')

        print(f'STEPS PER EPOCH: {len(gen)}')
        print(f'BATCH SIZE: {batch_size}')

        history = self.model.fit(
            gen.generate(),
            steps_per_epoch=len(gen),
            batch_size=batch_size,
            verbose=1,
            epochs=1,
            shuffle=False,
        )

        logger.info(f'Model name: {self.model.name}')
        logger.info(f'Model inputs: {self.model.inputs}')
        logger.info(f'Model outputs: {self.model.outputs}')
        # logger.info(f'{self.model.summary()}')
