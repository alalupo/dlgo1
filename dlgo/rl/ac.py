import logging.config

import numpy as np
import tensorflow as tf

keras = tf.keras
from keras.optimizers import SGD

from dlgo.goboard_fast import Move
from dlgo.agent.base import Agent
from dlgo.agent.helpers import is_point_an_eye

__all__ = [
    'ACAgent'
]

logger = logging.getLogger('acTrainingLogger')


class ACAgent(Agent):
    def __init__(self, model, encoder):
        super().__init__()
        self.model = model
        self.encoder = encoder
        self.collector = None
        self.temperature = 1.05
        self.last_state_value = 0

    def set_temperature(self, temperature):
        self.temperature = temperature

    def set_collector(self, collector):
        self.collector = collector

    def select_move(self, game_state):
        num_moves = self.encoder.board_width * self.encoder.board_height

        board_tensor = self.encoder.encode(game_state)
        board_tensor = np.transpose(board_tensor, (1, 2, 0))
        X = np.array([board_tensor])

        # actions, values = self.model.predict(X, verbose=0)
        actions, values = self.model(X)
        actions = actions.numpy()

        move_probs = actions[0]
        estimated_value = values[0][0]
        estimated_value = estimated_value.numpy()
        self.last_state_value = float(estimated_value)

        eps = 1e-5
        move_probs = (1 - eps) * move_probs + eps / num_moves
        move_probs = np.clip(move_probs, eps, 1 - eps)
        move_probs = move_probs / np.sum(move_probs)

        # Turn the probabilities into a ranked list of moves.
        candidates = np.arange(num_moves)
        ranked_moves = np.random.choice(
            candidates, num_moves, replace=False, p=move_probs)
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
                self.collector.record_decision(
                    state=board_tensor,
                    action=point_idx,
                    estimated_value=estimated_value
                )
            return move
        # No legal, non-self-destructive moves less.
        return Move.pass_turn()

    def train(self, experience, lr, batch_size):
        opt = SGD(learning_rate=lr, clipnorm=1)
        self.model.compile(
            optimizer=opt,
            loss=['categorical_crossentropy', 'huber_loss'],
            loss_weights=[1, 0.6])

        history = self.model.fit(
            experience.generate(),
            steps_per_epoch=len(experience),
            batch_size=batch_size,
            verbose=1,
            epochs=1,
            shuffle=False,
        )

        logger.info(f'Model name: {self.model.name}')
        logger.info(f'Model inputs: {self.model.inputs}')
        logger.info(f'Model outputs: {self.model.outputs}')
        # logger.info(f'{self.model.summary()}')

    def diagnostics(self):
        return {'value': self.last_state_value}
