import argparse
import logging.config

import h5py
from keras.models import load_model, save_model

from dlgo import rl
from dlgo.encoders.base import get_encoder_by_name
from dlgo.exp.exp_reader import ExpGenerator

logging.config.fileConfig('log_confs/train_logging.conf')
logger = logging.getLogger('trainingLogger')


def get_model(model_path):
    model_file = None
    # make sure the model file is closed
    try:
        model_file = open(model_path, 'r')
    finally:
        model_file.close()
    # load the model
    with h5py.File(model_path, "r") as model_file:
        model = load_model(model_file)
    return model


def create_bot(model_path, encoder):
    model = get_model(model_path)
    return rl.ACAgent(model, encoder)


def main():
    logger.info('TRAINER AC: STARTED')
    parser = argparse.ArgumentParser()
    parser.add_argument('--board-size', '-size', type=int, default=19, required=False)
    parser.add_argument('--learning-model', '-model', required=True)
    parser.add_argument('--model-out', '-out', required=True)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--bs', type=int, default=512)
    parser.add_argument('experience', nargs='+')

    args = parser.parse_args()
    board_size = args.board_size
    learning_model_filename = args.learning_model
    experience_files = args.experience
    updated_model_filename = args.model_out
    learning_rate = args.lr
    batch_size = args.bs

    logger.info(f'Learning agent filename: {learning_model_filename}')
    logger.info(f'Experience files: {experience_files}')
    logger.info(f'Updated agent filename: {updated_model_filename}')

    encoder = get_encoder_by_name('simple', board_size=board_size)
    print(f'>>>LOADING AGENT')
    learning_agent = create_bot(learning_model_filename, encoder)

    for exp_filename in experience_files:
        print(f'>>>LOADING EXPERIENCE: {exp_filename}')
        generator = ExpGenerator(exp_file=exp_filename,
                                 batch_size=batch_size,
                                 num_planes=encoder.num_planes,
                                 board_size=board_size)
        print(f'>>>AGENT TRAINING')
        learning_agent.train(
            generator,
            lr=learning_rate,
            batch_size=batch_size)
    print(f'>>>Updated agent is getting serialized.')
    with h5py.File(updated_model_filename, 'w') as updated_agent_outf:
        save_model(model=learning_agent.model, filepath=updated_agent_outf, save_format='h5')

    logger.info('TRAINER AC: FINISHED')


if __name__ == '__main__':
    main()
