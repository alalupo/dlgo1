import argparse
import logging.config

import h5py
import tensorflow as tf
keras = tf.keras
from keras.models import load_model, save_model
from keras.optimizers import SGD

from dlgo.tools.file_finder import FileFinder
from dlgo import rl
from dlgo.encoders.base import get_encoder_by_name
from dlgo.exp.exp_reader import ExpGenerator

logging.config.fileConfig('log_confs/ac_train_logging.conf')
logger = logging.getLogger('acTrainingLogger')


def main():
    logger.info('TRAINER AC: STARTED')
    parser = argparse.ArgumentParser()
    parser.add_argument('--board-size', '-size', type=int, default=19, required=False)
    parser.add_argument('--learning-model', '-model', required=True)
    parser.add_argument('--model-out', '-out', required=True)
    parser.add_argument('--learning-rate', '-lr', type=float, default=0.0001)
    parser.add_argument('--batch-size', '-batch', type=int, default=512)
    parser.add_argument('experience', nargs='+')

    args = parser.parse_args()
    board_size = args.board_size
    learning_model_filename = args.learning_model
    experience_files = args.experience
    updated_model_filename = args.model_out
    learning_rate = args.lr
    batch_size = args.bs

    logger.info(f'Learning agent filename: {learning_model_filename}')
    logger.info(f'Updated agent filename: {updated_model_filename}')
    logger.info(f'Experience files: {experience_files}')

    trainer = ACTrainer(board_size, learning_model_filename, updated_model_filename, learning_rate, batch_size,
                        experience_files)
    trainer.train()

    # encoder = get_encoder_by_name('simple', board_size=board_size)
    # print(f'>>>LOADING AGENT')
    # learning_agent = create_bot(learning_model_filename, encoder)
    #
    # for exp_filename in experience_files:
    #     print(f'>>>LOADING EXPERIENCE: {exp_filename}')
    #     generator = ExpGenerator(exp_file=exp_filename,
    #                              batch_size=batch_size,
    #                              num_planes=encoder.num_planes,
    #                              board_size=board_size)
    #     print(f'>>>AGENT TRAINING')
    #     learning_agent.train(
    #         generator,
    #         lr=learning_rate,
    #         batch_size=batch_size)
    # print(f'>>>Updated agent is getting serialized.')
    # with h5py.File(updated_model_filename, 'w') as updated_agent_outf:
    #     save_model(model=learning_agent.model, filepath=updated_agent_outf, save_format='h5')

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
        logger.info(f'=== NEW ACTrainer OBJECT CREATED ===')
        logger.info(f'ENCODER: {self.encoder.name()}')

    @staticmethod
    def get_model_path(model):
        finder = FileFinder()
        return finder.get_model_full_path(model)

    def train(self):
        print(f'>>>LOADING AGENT')
        learning_agent = self.create_bot(self.model_in_path)

        for exp_filename in self.exp_files:
            print(f'>>>LOADING EXPERIENCE: {exp_filename}')
            generator = ExpGenerator(exp_file=exp_filename,
                                     batch_size=self.batch_size,
                                     num_planes=self.encoder.num_planes,
                                     board_size=self.board_size)
            print(f'>>>MODEL TRAINING')
            learning_agent.train(
                generator,
                lr=self.learning_rate,
                batch_size=self.batch_size)
        print(f'>>>New model is getting saved.')
        with h5py.File(self.model_out_path, 'w') as model_outf:
            save_model(model=learning_agent.model, filepath=model_outf, save_format='h5')

    def create_bot(self, model_path):
        print(f'>>>Creating bot {model_path}...')
        model = self.get_model(model_path)
        return rl.ACAgent(model, self.encoder)

    def get_model(self, model_path):
        model_file = None
        try:
            model_file = open(model_path, 'r')
        finally:
            model_file.close()
        with h5py.File(model_path, "r") as model_file:
            model = load_model(model_file)
            model.compile(
                loss='categorical_crossentropy',
                optimizer=SGD(lr=self.learning_rate, clipnorm=1.0))
        return model


if __name__ == '__main__':
    main()
