import argparse
import h5py
import logging.config

from keras.models import load_model

from dlgo.encoders.simple import SimpleEncoder
from dlgo import agent
from dlgo import rl

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
    parser.add_argument('--learning-model', '-model', required=True)
    parser.add_argument('--model-out', '-out', required=True)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--bs', type=int, default=512)
    parser.add_argument('experience', nargs='+')

    args = parser.parse_args()
    learning_model_filename = args.learning_model
    experience_files = args.experience
    updated_model_filename = args.model_out
    learning_rate = args.lr
    batch_size = args.bs

    logger.info(f'Learning agent filename: {learning_model_filename}')
    logger.info(f'Experience files: {experience_files}')
    logger.info(f'Updated agent filename: {updated_model_filename}')

    encoder = SimpleEncoder((19, 19))
    print(f'>>>LOADING AGENT')
    learning_agent = create_bot(learning_model_filename, encoder)

    for exp_filename in experience_files:
        print(f'>>>LOADING EXPERIENCE: {exp_filename}')
        exp_buffer = rl.load_experience(h5py.File(exp_filename))
        print(f'>>>AGENT TRAINING')
        learning_agent.train(
            exp_buffer,
            lr=learning_rate,
            batch_size=batch_size)
    print(f'>>>Updated agent is getting serialized.')
    with h5py.File(updated_model_filename, 'w') as updated_agent_outf:
        learning_agent.serialize(updated_agent_outf)

    logger.info('TRAINER AC: FINISHED')


if __name__ == '__main__':
    main()
