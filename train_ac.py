import argparse
import logging.config
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import h5py
import tensorflow as tf

# tf.compat.v1.reset_default_graph()
# tf.keras.backend.clear_session()
# tf.debugging.set_log_device_placement(True)
keras = tf.keras
from keras.models import load_model, save_model

from dlgo.tools.file_finder import FileFinder
from dlgo.rl.ac import ACAgent
from dlgo.encoders.base import get_encoder_by_name
from dlgo.exp.exp_reader import ExpGenerator

logger = logging.getLogger('acTrainingLogger')


def main():
    logger.info('TRAINER AC: STARTED')
    parser = argparse.ArgumentParser()
    parser.add_argument('--board-size', '-size', type=int, default=19, required=False)
    parser.add_argument('--learning-model', '-in', required=True)
    parser.add_argument('--model-out', '-out', required=True)
    parser.add_argument('--learning-rate', '-lr', type=float, default=0.0001)
    parser.add_argument('--batch-size', '-batch', type=int, default=128)
    parser.add_argument('experience', nargs='+')

    args = parser.parse_args()

    board_size = args.board_size
    learning_model_filename = args.learning_model
    updated_model_filename = args.model_out
    learning_rate = args.learning_rate
    batch_size = args.batch_size
    experience_files = args.experience

    logger.info(f'Learning agent filename: {learning_model_filename}')
    logger.info(f'Updated agent filename: {updated_model_filename}')
    logger.info(f'Experience files: {experience_files}')

    trainer = ACTrainer(board_size, learning_model_filename, updated_model_filename, learning_rate, batch_size,
                        experience_files)
    trainer.train()

    logger.info('TRAINER AC: FINISHED')


class ACTrainer:
    def __init__(self, board_size, model_in, model_out, learning_rate, batch_size, exp_files):
        self.board_size = board_size
        self.rows, self.cols = self.board_size, self.board_size
        self.encoder = get_encoder_by_name('simple', self.board_size)
        self.model_in_path = self.get_model_path(model_in)
        self.model_out_path = self.get_model_path(model_out)
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.exp_files = exp_files
        self.exp_paths = []
        if isinstance(exp_files, (list, tuple)):
            for exp_file in exp_files:
                exp_path = self.get_exp_path(exp_file)
                self.exp_paths.append(exp_path)
        else:
            exp_path = self.get_exp_path(exp_files)
            self.exp_paths.append(exp_path)
        logger.info(f'=== NEW ACTrainer OBJECT CREATED ===')
        logger.info(f'ENCODER: {self.encoder.name()}')

    @staticmethod
    def get_model_path(model):
        finder = FileFinder()
        return finder.get_model_full_path(model)

    def get_exp_path(self, name):
        finder = FileFinder()
        return finder.get_exp_full_path(name)

    def train(self):
        print(f'')
        print(f'>>>LOADING AGENT')
        learning_agent = self.create_bot(self.model_in_path)

        for exp_filename in self.exp_paths:
            print(f'')
            print(f'>>>LOADING EXPERIENCE: {exp_filename}')
            generator = ExpGenerator(exp_file=exp_filename,
                                     batch_size=self.batch_size,
                                     num_planes=self.encoder.num_planes,
                                     board_size=self.board_size,
                                     seed=1234)
            print(f'>>>MODEL TRAINING')
            learning_agent.train(
                generator,
                lr=self.learning_rate,
                batch_size=self.batch_size)
        print(f'>>>New model is getting saved.')
        # save_model(model=learning_agent.model, filepath=self.model_out_path, save_format='h5')
        with h5py.File(self.model_out_path, 'w') as model_outf:
            save_model(model=learning_agent.model, filepath=model_outf, save_format='h5')
        # learning_agent.model.save(self.model_out_path, save_format='h5')

    def create_bot(self, model_path):
        print(f'>>>Creating bot {model_path}...')
        model = self.get_model(model_path)
        return ACAgent(model, self.encoder)

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
