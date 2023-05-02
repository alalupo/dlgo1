import os
from pathlib import Path

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import argparse
import logging.config

import h5py
import tensorflow as tf

keras = tf.keras
from keras.models import Model, load_model, save_model

from dlgo.encoders.base import get_encoder_by_name
from dlgo.networks.network_architectures import ValueNetwork
from dlgo.rl.value_agent import ValueAgent
from dlgo.exp.exp_reader import ExpGenerator

tf.get_logger().setLevel('WARNING')
logger = logging.getLogger('trainingLogger')


def main():
    logger.info('VALUE TRAINER: STARTED')

    parser = argparse.ArgumentParser()
    parser.add_argument('experience', nargs='+')

    args = parser.parse_args()
    experience_files = args.experience

    trainer = ValueTrainer(experience_files)
    trainer.train()

    logger.info('VALUE TRAINER: FINISHED')


class ValueTrainer:
    def __init__(self, exp_files):
        self.board_size = 19
        self.num_classes = self.board_size * self.board_size
        self.batch_size = 128
        self.encoder = get_encoder_by_name('simple', board_size=self.board_size)
        self.network = ValueNetwork(encoder=self.encoder)
        self.optimizer = tf.keras.optimizers.Adagrad()
        self.loss_function = 'categorical_crossentropy'
        self.model_dir = str(Path.cwd() / 'models')
        self.model = self.build_model()
        self.model_name = f'model_value_rl_{self.network.name}.h5'
        self.model_path = Path.cwd() / 'models' / self.model_name
        self.learning_rate = 0.007
        self.exp_files = exp_files
        self.exp_paths = []
        if isinstance(exp_files, (list, tuple)):
            for exp_file in exp_files:
                exp_path = str(Path.cwd() / 'exp' / exp_file)
                self.exp_paths.append(exp_path)
        else:
            exp_path = str(Path.cwd() / 'exp' / exp_files)
            self.exp_paths.append(exp_path)
        logger.info(f'BOARD SIZE: {self.board_size}')
        logger.info(f'ENCODER: {self.encoder.name()}')
        logger.info(f'NETWORK: {self.network.name}')
        logger.info(f'ENCODER\'S ORIGINAL SHAPE: {self.encoder.shape}')
        logger.info(f'OPTIMIZER: {self.optimizer}')
        logger.info(f'LOSS FUNCTION: {self.loss_function}')
        logger.info(f'BATCH SIZE: {self.batch_size}')

    def build_model(self):
        model = Model(inputs=self.network.board_input,
                      outputs=self.network.output)
        print(f'*' * 80)
        print(f'Model summary:')
        model.summary()
        print(f'*' * 80)
        return model

    def train(self):
        print(f'')
        print(f'>>>LOADING AGENT')
        value_agent = self.create_bot(self.model_path)

        for exp_filename in self.exp_paths:
            print(f'')
            print(f'>>>LOADING EXPERIENCE: {exp_filename}')
            generator = ExpGenerator(exp_file=exp_filename,
                                     batch_size=self.batch_size,
                                     num_planes=self.encoder.num_planes,
                                     board_size=self.board_size,
                                     seed=1234)
            print(f'>>>MODEL TRAINING')
            value_agent.train(
                generator,
                lr=self.learning_rate,
                batch_size=self.batch_size)
        print(f'>>>New model is getting saved.')
        with h5py.File(self.model_path, 'w') as model_outf:
            save_model(model=value_agent.model, filepath=model_outf, save_format='h5')

    def create_bot(self, model_path):
        print(f'>>>Creating bot {model_path}...')
        model = self.get_model(model_path)
        return ValueAgent(model, self.encoder)

    def get_model(self, model_path):
        model_file = None
        try:
            model_file = open(model_path, 'r')
        finally:
            model_file.close()
        with h5py.File(model_path, "r") as model_file:
            model = load_model(model_file)
        return model


if __name__ == '__main__':
    main()
