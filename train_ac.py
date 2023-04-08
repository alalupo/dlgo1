import argparse
import h5py
import logging.config

from dlgo import agent
from dlgo import rl

logging.config.fileConfig('train_logging.conf')
logger = logging.getLogger('trainingLogger')


def main():
    logger.info('TRAINER AC: STARTED')
    parser = argparse.ArgumentParser()
    parser.add_argument('--learning-agent', required=True)
    parser.add_argument('--agent-out', required=True)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--bs', type=int, default=512)
    parser.add_argument('experience', nargs='+')

    args = parser.parse_args()
    learning_agent_filename = args.learning_agent
    experience_files = args.experience
    updated_agent_filename = args.agent_out
    learning_rate = args.lr
    batch_size = args.bs

    logger.info(f'Learning agent filename: {learning_agent_filename}')
    logger.info(f'Experience files: {experience_files}')
    logger.info(f'Updated agent filename: {updated_agent_filename}')

    print(f'>>>LOADING AGENT')
    learning_agent = rl.load_ac_agent(h5py.File(learning_agent_filename))

    for exp_filename in experience_files:
        print(f'>>>LOADING EXPERIENCE: {exp_filename}')
        exp_buffer = rl.load_experience(h5py.File(exp_filename))
        print(f'>>>AGENT TRAINING')
        learning_agent.train(
            exp_buffer,
            lr=learning_rate,
            batch_size=batch_size)
    print(f'>>>Updated agent is getting serialized.')
    with h5py.File(updated_agent_filename, 'w') as updated_agent_outf:
        learning_agent.serialize(updated_agent_outf)

    logger.info('TRAINER AC: FINISHED')


if __name__ == '__main__':
    main()
