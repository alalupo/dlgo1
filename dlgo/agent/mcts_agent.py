import operator
import numpy as np

from dlgo.agent.base import Agent
from dlgo.goboard_fast import Move
from dlgo.encoders.base import get_encoder_by_name
from dlgo.agent.predict import DeepLearningAgent
from dlgo.agent.pg import PolicyAgent
from dlgo.rl.value_agent import ValueAgent

__all__ = [
    'MCTSNode',
    'MCTSAgent'
]


class MCTSNode:
    def __init__(self, parent=None, probability=1.0):
        self.parent = parent
        self.children = {}
        self.visit_count = 0
        self.q_value = 0
        self.prior_value = probability
        self.u_value = probability

    def select_child(self):
        return max(self.children.items(),
                   key=lambda child: child[1].q_value + child[1].u_value)

    def expand_children(self, moves, probabilities):
        for move, prob in zip(moves, probabilities):
            if move not in self.children:
                self.children[move] = MCTSNode(probability=prob)

    def update_values(self, leaf_value):
        if self.parent is not None:
            self.parent.update_values(leaf_value)

        self.visit_count += 1

        self.q_value += leaf_value / self.visit_count

        if self.parent is not None:
            c_u = 5
            self.u_value = c_u * np.sqrt(self.parent.visit_count) \
                           * self.prior_value / (1 + self.visit_count)


class MCTSAgent(Agent):
    def __init__(self, strong_policy_model, fast_policy_model, value_model, lambda_value=0.5, num_simulations=25,
                 depth=5, rollout_limit=3):
        super().__init__()
        self.encoder = get_encoder_by_name('simple', 19)
        self.policy = PolicyAgent(strong_policy_model, self.encoder)
        self.rollout_policy = DeepLearningAgent(fast_policy_model, self.encoder)
        self.value = ValueAgent(value_model, self.encoder)
        self.lambda_value = lambda_value
        self.num_simulations = num_simulations
        self.depth = depth
        self.rollout_limit = rollout_limit
        self.root = MCTSNode()
        self.last_state_value = 0

    def select_move(self, game_state):
        for simulation in range(self.num_simulations):
            current_state = game_state
            node = self.root
            for depth in range(self.depth):
                if not node.children:
                    if current_state.is_over():
                        break
                    moves, probabilities = self.policy_probabilities(current_state)
                    node.expand_children(moves, probabilities)

                move, node = node.select_child()
                current_state = current_state.apply_move(move)
            value = self.value.predict(current_state)
            rollout = self.policy_rollout(current_state)

            weighted_value = (1 - self.lambda_value) * value + \
                             self.lambda_value * rollout

            node.update_values(weighted_value)

        move = max(self.root.children,
                   key=lambda move: self.root.children.get(move).visit_count)

        self.root = MCTSNode()
        if move in self.root.children:
            self.root = self.root.children[move]
            self.root.parent = None

        return move

    def policy_probabilities(self, game_state):
        encoder = self.policy.encoder
        outputs = self.policy.predict(game_state)
        legal_moves = game_state.legal_moves()
        if not legal_moves:
            return [], []
        encoded_points = [encoder.encode_point(move.point) for move in legal_moves if move.point]
        legal_outputs = outputs[encoded_points]
        normalized_outputs = legal_outputs / np.sum(legal_outputs)
        return legal_moves, normalized_outputs

    def policy_rollout(self, game_state):
        for step in range(self.rollout_limit):
            if game_state.is_over():
                break
            move_probabilities = self.rollout_policy.predict(game_state)
            encoder = self.rollout_policy.encoder

            legal_moves = set(game_state.legal_moves())
            valid_moves = [m for idx, m in enumerate(move_probabilities)
                            if Move(encoder.decode_point_index(idx)) in legal_moves]

            # valid_moves = [m for idx, m in enumerate(move_probabilities)
            #                 if Move(encoder.decode_point_index(idx)) in game_state.legal_moves()]

            max_index, max_value = max(enumerate(valid_moves), key=operator.itemgetter(1))
            max_point = encoder.decode_point_index(max_index)
            greedy_move = Move(max_point)
            if greedy_move in game_state.legal_moves():
                game_state = game_state.apply_move(greedy_move)

        next_player = game_state.next_player
        winner = game_state.winner()
        if winner is not None:
            return 1 if winner == next_player else -1
        else:
            return 0

    def diagnostics(self):
        return {'value': self.last_state_value}

    def serialize(self, h5file):
        raise IOError("MCTS agent can\'t be serialized")
