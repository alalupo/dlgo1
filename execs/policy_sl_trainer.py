import os
from pathlib import Path
import sys

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import argparse
import logging.config

import matplotlib.pyplot as plt
import tensorflow as tf

keras = tf.keras
from keras.models import Model
from keras.callbacks import ModelCheckpoint

project_path = Path(__file__).resolve().parent.parent
os.chdir(project_path)
sys.path.append(str(project_path))
sys.path.append(str(Path.cwd() / 'dlgo'))

from dlgo.data.data_processor import GoDataProcessor
from dlgo.encoders.base import get_encoder_by_name
from dlgo.networks.network_architectures import FastPolicyNetwork, StrongPolicyNetwork

tf.get_logger().setLevel('WARNING')
logger = logging.getLogger('trainingLogger')


def cleaning():
    data_dir = Path.cwd() / 'data'
    samples_file = Path('test_samples.py')
    if Path(samples_file).is_file():
        Path.unlink(samples_file)
    for path in data_dir.glob("*.npy"):
        if Path(path).is_file():
            Path.unlink(path)


def main():
    logger.info('POLICY TRAINER: STARTED')
    cleaning()

    parser = argparse.ArgumentParser()
    parser.add_argument('--strong', default=True, action=argparse.BooleanOptionalAction)  # --no-strong, jeśli fast
    parser.add_argument('--num-games', '-n', type=int, default=100, required=True)
    parser.add_argument('--epochs', '-e', type=int, default=5, required=True)

    args = parser.parse_args()
    strong_policy = args.strong
    num_games = args.num_games
    epochs = args.epochs

    if strong_policy:
        logger.info(f'>>> Strong policy model getting trained...')
    else:
        logger.info(f'>>> Fast policy model getting trained...')
    logger.info(f'GAMES: {num_games}')
    logger.info(f'EPOCHS: {epochs}')

    trainer = SLTrainer(strong_policy, num_games, epochs)
    train_generator, test_generator = trainer.get_datasets()
    trainer.train_model(train_generator, test_generator)

    cleaning()
    logger.info('POLICY TRAINER: FINISHED')


class SLTrainer:
    """
    Supervised Learning policy trainer class.
    """
    def __init__(self, strong, num_games, num_epochs):
        self.num_games = num_games
        self.epochs = num_epochs
        self.board_size = 19
        self.num_classes = self.board_size * self.board_size
        self.batch_size = 128
        self.encoder = get_encoder_by_name('simple', board_size=self.board_size)
        if strong:
            self.network = StrongPolicyNetwork(encoder=self.encoder)
        else:
            self.network = FastPolicyNetwork(encoder=self.encoder)
        self.optimizer = tf.keras.optimizers.Adagrad()
        self.loss_function = 'categorical_crossentropy'
        self.model_dir = str(Path.cwd() / 'models')
        self.model = self.build_model()
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

    def get_datasets(self):
        processor = GoDataProcessor(encoder=self.encoder.name(), board_size=self.board_size)
        train_generator = processor.load_go_data(num_samples=self.num_games, data_type='train')
        print(f'>>>Train generator loaded')
        test_generator = processor.load_go_data(num_samples=self.num_games, data_type='test')
        print(f'>>>Test generator loaded')
        return train_generator, test_generator

    def train_model(self, train_generator, test_generator):
        encoder_name = self.encoder.name()
        network_name = self.network.name
        print(f'>>>Model compiling...')
        self.model.compile(optimizer=self.optimizer,
                           loss=self.loss_function,
                           metrics=['accuracy'])
        print(f'>>>Model fitting...')
        callback = ModelCheckpoint(self.model_dir +
                                   '/model_sl_' +
                                   network_name + '_' +
                                   str(self.num_games) + '_' +
                                   str(self.epochs) + '_' +
                                   'epoch{epoch}.h5',
                                   save_weights_only=False,
                                   save_best_only=True)
        train_steps = train_generator.get_num_samples(batch_size=self.batch_size) // self.batch_size
        test_steps = test_generator.get_num_samples(batch_size=self.batch_size) // self.batch_size
        logger.info(f'Train steps = {train_steps}')
        logger.info(f'Test steps = {test_steps}')
        history = self.model.fit(
            train_generator.generate(self.batch_size),
            shuffle=False,
            epochs=self.epochs,
            steps_per_epoch=train_steps,
            validation_data=test_generator.generate(self.batch_size),
            validation_steps=test_steps,
            callbacks=[callback])
        print(f'>>>Model evaluating...')
        score = self.model.evaluate(
            test_generator.generate(self.batch_size),
            steps=test_steps)
        print(f'*' * 80)
        logger.info(f'Test loss: {score[0]}')
        logger.info(f'Test accuracy: {score[1]}')
        self.save_plots(history, self.model_dir, network_name)

    def save_plots(self, history, model_dir, network_name):
        loss = history.history['loss']
        val_loss = history.history['val_loss']
        epochs = range(1, len(loss) + 1)
        plt.plot(epochs, loss, 'bo', label='Train loss')
        plt.plot(epochs, val_loss, 'b', label='Validation loss')
        plt.title(f'Train and validation loss (games: {self.num_games}, epochs: {self.epochs})')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig(f'{model_dir}/graph_{network_name}_{self.num_games}_{self.epochs}_loss.png')
        plt.clf()
        acc = history.history['accuracy']
        val_acc = history.history['val_accuracy']
        plt.plot(epochs, acc, 'bo', label='Train accuracy')
        plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
        plt.title(f'Train and validation accuracy (games: {self.num_games}, epochs: {self.epochs})')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.savefig(f'{model_dir}/graph_{network_name}_{self.num_games}_{self.epochs}_accuracy.png')


if __name__ == '__main__':
    main()
