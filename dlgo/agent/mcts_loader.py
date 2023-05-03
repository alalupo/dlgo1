import os
import sys
from pathlib import Path

import h5py

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf

keras = tf.keras
from keras.models import load_model

project_path = Path(__file__).resolve().parent.parent.parent
os.chdir(project_path)
sys.path.append(str(project_path))
sys.path.append(str(Path.cwd() / 'dlgo'))

from dlgo.agent.mcts_agent import MCTSAgent


class MCTSLoader:
    def __init__(self):
        self.model_dir = Path.cwd() / 'models'
        self.fast_policy_name = 'model_sl_fast_small_1000_10_epoch10.h5'
        self.strong_policy_name = 'model_rl_strong_improved_100_1_epoch1.h5'
        self.value_name = 'model_value_rl_value_improved.h5'
        self.fast_policy_path = str(self.model_dir / self.fast_policy_name)
        self.strong_policy_path = str(self.model_dir / self.strong_policy_name)
        self.value_path = str(self.model_dir / self.value_name)
        self.fast_model = load_model(h5py.File(self.fast_policy_path, 'r'))
        self.strong_model = load_model(h5py.File(self.strong_policy_path, 'r'))
        self.value_model = load_model(h5py.File(self.value_path, 'r'))
        self.agent = MCTSAgent(self.strong_model, self.fast_model, self.value_model)

    def get_agent(self):
        return self.agent
