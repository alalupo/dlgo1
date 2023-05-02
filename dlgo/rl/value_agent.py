import numpy as np
import logging.config
import tensorflow as tf

keras = tf.keras
from keras.optimizers import SGD

from dlgo.goboard_fast import Move
from dlgo.agent import Agent
from dlgo.agent.helpers import is_point_an_eye

__all__ = [
    'ValueAgent',
]

logger = logging.getLogger('acTrainingLogger')


class ValueAgent(Agent):
    def __init__(self, model, encoder, policy='eps-greedy'):
        super().__init__()
        self.model = model
        self.encoder = encoder
        self.collector = None
        self.temperature = 0.0
        self.policy = policy

        self.last_move_value = 0

    def set_temperature(self, temperature):
        self.temperature = temperature

    def set_collector(self, collector):
        self.collector = collector

    def set_policy(self, policy):
        if policy not in ('eps-greedy', 'weighted'):
            raise ValueError(policy)
        self.policy = policy

    def select_move(self, game_state):
        # Loop over all legal moves.
        moves = []
        board_tensors = []
        board_tensor = None
        for move in game_state.legal_moves():
            if not move.is_play:
                continue
            next_state = game_state.apply_move(move)
            board_tensor = self.encoder.encode(next_state)
            board_tensor = np.transpose(board_tensor, (1, 2, 0))
            moves.append(move)
            board_tensors.append(board_tensor)
        if not moves:
            return Move.pass_turn()

        board_tensors = np.array(board_tensors)
        # Values of the next state from opponent's view.
        opp_values = self.model.predict(board_tensors)
        # opp_values = self.model(board_tensors)
        # opp_values = opp_values.numpy()
        opp_values = opp_values.reshape(len(moves))

        # Values from our point of view.
        values = 1 - opp_values

        ranked_moves = np.array([])
        if self.policy == 'eps-greedy':
            ranked_moves = np.concatenate([ranked_moves, self.rank_moves_eps_greedy(values)])
        elif self.policy == 'weighted':
            ranked_moves = np.concatenate([ranked_moves, self.rank_moves_weighted(values)])

        for move_idx in ranked_moves:
            move = moves[move_idx]
            fills_own_eye = is_point_an_eye(game_state.board, move.point, game_state.next_player)
            if fills_own_eye:
                continue
            if self.collector is not None and board_tensor is not None:
                self.collector.record_decision(
                    state=board_tensor,
                    action=self.encoder.encode_point(move.point),
                )
            self.last_move_value = float(values[move_idx])
            return move
        # No legal, non-self-destructive moves less.
        return Move.pass_turn()

    def rank_moves_eps_greedy(self, values):
        if np.random.random() < self.temperature:
            values = np.random.random(values.shape)
        # This ranks the moves from worst to best.
        ranked_moves = np.argsort(values)
        # Return them in best-to-worst order.
        return ranked_moves[::-1]

    def rank_moves_weighted(self, values):
        p = values / np.sum(values)
        p = np.power(p, 1.0 / self.temperature)
        p = p / np.sum(p)
        return np.random.choice(
            np.arange(0, len(values)),
            size=len(values),
            p=p,
            replace=False)

    def train(self, experience, lr, batch_size=128):
        opt = SGD(learning_rate=lr, clipnorm=1)
        self.model.compile(optimizer=opt, loss='huber_loss')

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
        return {'value': self.last_move_value}
