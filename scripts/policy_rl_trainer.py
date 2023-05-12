import argparse
import logging.config
import os
import sys
from pathlib import Path

import tensorflow as tf

keras = tf.keras
from keras.models import load_model, save_model

project_path = Path(__file__).resolve().parent.parent
os.chdir(project_path)
sys.path.append(str(project_path))
sys.path.append(str(project_path / 'dlgo'))

from dlgo.agent.pg import PolicyAgent
from dlgo.encoders.base import get_encoder_by_name
from dlgo.exp.exp_reader import ExpReader
import h5py

logger = logging.getLogger('trainingLogger')


def main():
    logger.info('RL TRAINER: STARTED')
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', '-m', required=True)
    parser.add_argument('experience', nargs='+')

    args = parser.parse_args()
    model_name = args.model
    experience_files = args.experience

    model_path = str(Path.cwd() / 'models' / model_name)
    exp_paths = []
    if isinstance(experience_files, (list, tuple)):
        for exp_file in experience_files:
            exp_path = str(Path.cwd() / 'exp' / exp_file)
            exp_paths.append(exp_path)
    else:
        exp_path = str(Path.cwd() / 'exp' / experience_files)
        exp_paths.append(exp_path)

    trainer = RLTrainer(model_path, exp_paths)
    trainer.train()
    logger.info('RL TRAINER: FINISHED')


class RLTrainer:
    def __init__(self, model_sl_path: str, exp_paths: list[str]):
        self.board_size = 19
        self.batch_size = 1024
        self.encoder = get_encoder_by_name('simple', self.board_size)
        self.model_sl_path = model_sl_path
        self.model_rl_path = Path(str(self.model_sl_path).replace('_sl_', '_rl_'))
        self.learning_rate = 0.007
        self.exp_paths = []
        if isinstance(exp_paths, (list, tuple)):
            for exp_file in exp_paths:
                self.exp_paths.append(exp_file)
        else:
            self.exp_paths.append(exp_paths)

    def train(self):
        print(f'')
        print(f'>>> LOADING RL AGENT...')
        rl_agent = self.create_bot()

        for exp_filename in self.exp_paths:
            print(f'')
            print(f'>>> LOADING EXPERIENCE: {exp_filename}...')
            generator = ExpReader(exp_file=exp_filename,
                                  batch_size=self.batch_size,
                                  num_planes=self.encoder.num_planes,
                                  board_size=self.board_size,
                                  seed=1234,
                                  client='pg')
            print(f'>>> TRAINING RL MODEL...')
            rl_agent.train(
                generator,
                lr=self.learning_rate,
                clipnorm=1,
                batch_size=self.batch_size)

        self.save_rl_model(rl_agent.model)

    def create_bot(self):
        model = self.get_sl_model()
        print(f'>>> Bot for {self.model_sl_path} has been created.')
        return PolicyAgent(model, self.encoder)

    def get_sl_model(self):
        model_file = None
        try:
            model_file = open(self.model_sl_path, 'r')
        finally:
            model_file.close()
        with h5py.File(self.model_sl_path, "r") as model_file:
            model = load_model(model_file)
        return model

    def save_rl_model(self, model):
        with h5py.File(self.model_rl_path, 'w') as f:
            save_model(model=model, filepath=f, save_format='h5')
        print(f'>>> RL model has been saved at {self.model_rl_path}.')


if __name__ == '__main__':
    main()
