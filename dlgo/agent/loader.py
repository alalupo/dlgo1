import os
import sys
from pathlib import Path

import tensorflow as tf

keras = tf.keras
from keras.models import load_model

project_path = Path(__file__).resolve().parent.parent.parent
os.chdir(project_path)
sys.path.append(str(project_path))
sys.path.append(str(project_path / 'dlgo'))

from dlgo.zero.agent import ZeroAgent
from dlgo.agent.mcts_loader import MCTSLoader
from dlgo.agent.pg import PolicyAgent
from dlgo.agent.predict import DeepLearningAgent
from dlgo.rl.ac import ACAgent
from dlgo.rl.q import QAgent
from dlgo.zero.encoder import ZeroEncoder
from dlgo.encoders.base import get_encoder_by_name
import h5py


class Loader:
    def __init__(self, name):
        self.agent_name = name
        self.encoder = get_encoder_by_name('simple', 19)
        self.zero_model_name = 'model_sl_zero_v1.h5'
        self.pg_model_name = 'model_sl_strong_improved3_10000_1_epoch1_24proc.h5'
        self.predict_model_name = 'model_sl_strong_improved3_10000_1_epoch1_24proc.h5'
        self.q_model_name = 'model_sl_strong_improved3_10000_1_epoch1_24proc.h5'
        self.ac_model_name = 'model_sl_strong_improved3_10000_1_epoch1_24proc.h5'
        self.model_dir = Path.cwd() / 'models'
        self.model = self.get_model()

    def get_model(self):
        path = None
        if self.agent_name == 'mcts':
            return None
        if self.agent_name == 'zero':
            path = str(self.model_dir / self.zero_model_name)
        elif self.agent_name == 'pg':
            path = str(self.model_dir / self.pg_model_name)
        elif self.agent_name == 'predict':
            path = str(self.model_dir / self.predict_model_name)
        elif self.agent_name == 'q':
            path = str(self.model_dir / self.q_model_name)
        elif self.agent_name == 'ac':
            path = str(self.model_dir / self.ac_model_name)
        else:
            raise ValueError('Unknown agent type')
        model_file = None
        try:
            model_file = open(path, 'r')
        finally:
            model_file.close()
        with h5py.File(path, "r") as model_file:
            model = load_model(model_file)
        return model

    def create_bot(self):
        if self.agent_name == 'mcts':
            player = MCTSLoader()
            return player.get_agent()
        elif self.agent_name == 'zero':
            encoder = ZeroEncoder(19)
            return ZeroAgent(self.model, encoder, rounds_per_move=30)
        elif self.agent_name == 'pg':
            return PolicyAgent(self.model, self.encoder)
        elif self.agent_name == 'predict':
            return DeepLearningAgent(self.model, self.encoder)
        elif self.agent_name == 'q':
            q_agent = QAgent(self.model, self.encoder)
            q_agent.set_temperature(0.01)
            return q_agent
        elif self.agent_name == 'ac':
            ac_agent = ACAgent(self.model, self.encoder)
            ac_agent.set_temperature(0.05)
            return ac_agent
        else:
            raise ValueError('Unknown agent type')
