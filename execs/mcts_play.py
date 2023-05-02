import h5py
import tensorflow as tf
keras = tf.keras
from keras.models import load_model

from dlgo.agent.mcts_agent import MCTSAgent

fast_policy = load_model(h5py.File('alphago_sl_policy.h5', 'r'))
strong_policy = load_model(h5py.File('alphago_rl_policy.h5', 'r'))
value = load_model(h5py.File('alphago_value.h5', 'r'))

agent = MCTSAgent(strong_policy, fast_policy, value)

# TODO: register in frontend
