import argparse
import logging.config
import os
from pathlib import Path

import keras.backend as K
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.callbacks import ModelCheckpoint
from keras.models import Model, Sequential

from dlgo.data.parallel_processor import GoDataProcessor
from dlgo.encoders.base import get_encoder_by_name
from dlgo.networks.func_networks import show_data_format, TrainerNetwork

logging.config.fileConfig('log_confs/train_logging.conf')
logger = logging.getLogger('trainingLogger')


def show_intro():
    print(f'*' * 80)
    print(f'Don\'t forget to clean up data directory of npy files before you start training a new model.')
    print(f'Put into the terminal the following command:')
    print(f'    find ./data/ -name \*.npy -delete')
    show_data_format()
    print(f'*' * 80)
    print(f'Usage example:')
    print(f'python3 trainer.py --board-size 19 --num-games 1000 --epochs 10')
    print(f'*' * 80)


def first_training(trainer, batch_size):
    logger.info(f'FIRST TRAINING')
    trainer.train_model(batch_size)


def next_training(trainer, batch_size):
    logger.info(f'NEXT TRAINING')
    filename = 'model_simple_small_5000_3_epoch3_8proc.h5'
    logger.info(f'MODEL NAME: {filename}')
    trainer.continue_training(batch_size, filename)


def main():
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    logger.info('TRAINER: STARTED')
    parser = argparse.ArgumentParser()
    parser.add_argument('--board-size', '-size', type=int, default=19, required=False)
    parser.add_argument('--num-games', '-n', type=int, default=100, required=False)
    parser.add_argument('--epochs', '-e', type=int, default=5, required=False)

    args = parser.parse_args()
    board_size = args.board_size
    num_games = args.num_games
    epochs = args.epochs

    encoder = get_encoder_by_name('simple', board_size=board_size)
    input_shape = encoder.shape_for_others()
    network = TrainerNetwork(encoder=encoder)

    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    loss_function = 'categorical_crossentropy'
    batch_size = 128

    show_intro()
    logger.info(f'GAMES: {num_games}')
    logger.info(f'EPOCHS: {epochs}')
    logger.info(f'BOARD SIZE: {board_size}')
    logger.info(f'ENCODER: {encoder.name()}')
    logger.info(f'NETWORK: {network.name}')
    logger.info(f'INPUT SHAPE: {input_shape}')
    logger.info(f'OPTIMIZER: {optimizer}')
    logger.info(f'LOSS FUNCTION: {loss_function}')
    logger.info(f'BATCH SIZE: {batch_size}')

    trainer = Trainer(network, encoder, num_games, epochs, optimizer, loss_function, board_size)
    first_training(trainer, batch_size)
    logger.info('TRAINER: FINISHED')


class Trainer:
    def __init__(self, network, encoder, num_games, num_epochs, optimizer,
                 loss='categorical_crossentropy', board_size=19):
        self.network = network
        self.encoder = encoder
        self.num_games = num_games
        self.epochs = num_epochs
        self.optimizer = optimizer
        self.loss = loss
        self.board_size = board_size
        self.go_board_rows, self.go_board_cols = self.board_size, self.board_size
        self.num_classes = self.go_board_rows * self.go_board_cols
        self.model_dir = self.get_model_directory()
        self.model = self.build_model()

    @staticmethod
    def get_model_directory():
        path = Path(__file__)
        project_lvl_path = path.parent
        model_directory_name = 'models'
        model_directory = project_lvl_path.joinpath(model_directory_name)
        if not os.path.exists(model_directory):
            raise FileNotFoundError
        return str(model_directory)

    def build_model(self):
        model = Model(inputs=self.network.board_input,
                      outputs=self.network.output)
        print(f'*' * 80)
        print(f'Model summary:')
        model.summary()
        print(f'*' * 80)
        return model

    def train_model(self, batch_size=128):
        K.clear_session()
        train_generator, test_generator = self.get_datasets()
        encoder_name = self.encoder.name()
        network_name = self.network.name
        print(f'>>>Model compiling...')
        self.model.compile(optimizer=self.optimizer,
                           loss=self.loss,
                           metrics=['accuracy'])
        print(f'>>>Model fitting...')
        callback = ModelCheckpoint(self.model_dir + '/model_' + encoder_name + '_' + network_name + '_epoch_{epoch}.h5',
                                   save_weights_only=False,
                                   save_best_only=True)
        history = self.model.fit(
            train_generator.generate(batch_size),
            epochs=self.epochs,
            steps_per_epoch=train_generator.get_num_samples(batch_size=batch_size) / batch_size,
            validation_data=test_generator.generate(batch_size),
            validation_steps=test_generator.get_num_samples(batch_size=batch_size) / batch_size,
            callbacks=[callback])
        print(f'>>>Model evaluating...')
        score = self.model.evaluate(
            test_generator.generate(batch_size),
            steps=test_generator.get_num_samples(batch_size=batch_size) / batch_size)
        print(f'*' * 80)
        logger.info(f'Test loss: {score[0]}')
        logger.info(f'Test accuracy: {score[1]}')
        self.save_plots(history, self.model_dir, encoder_name, network_name)

    def get_datasets(self):
        processor = GoDataProcessor(encoder=self.encoder.name(), board_size=self.board_size)
        train_generator = processor.load_go_data(num_samples=self.num_games, data_type='train')
        print(f'>>>Train generator loaded')
        test_generator = processor.load_go_data(num_samples=self.num_games, data_type='test')
        print(f'>>>Test generator loaded')
        return train_generator, test_generator

    def save_plots(self, history, model_dir, encoder_name, network_name):
        loss = history.history['loss']
        val_loss = history.history['val_loss']
        epochs = range(1, len(loss) + 1)
        plt.plot(epochs, loss, 'bo', label='Train loss')
        plt.plot(epochs, val_loss, 'b', label='Validation loss')
        plt.title(f'Train and validation loss (games: {self.num_games}, epochs: {self.epochs})')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig(f'{model_dir}/graph_{encoder_name}_{network_name}_{self.num_games}_{self.epochs}_loss.png')

        plt.clf()

        acc = history.history['accuracy']
        val_acc = history.history['val_accuracy']
        plt.plot(epochs, acc, 'bo', label='Train accuracy')
        plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
        plt.title(f'Train and validation accuracy (games: {self.num_games}, epochs: {self.epochs})')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.savefig(f'{model_dir}/graph_{encoder_name}_{network_name}_{self.num_games}_{self.epochs}_accuracy.png')

    def continue_training(self, batch_size, filename):
        K.clear_session()
        # To continue training from the last saved checkpoint,
        # create a new model with the same architecture as the previous model
        new_model = Sequential()
        # Define the ModelCheckpoint callback to save the best model during training
        new_filename = 'final_' + filename
        model_path = self.model_dir + filename
        encoder_name = self.encoder.name()
        network_name = self.network.name
        train_generator, test_generator = self.get_datasets()
        checkpoint_callback = ModelCheckpoint(filepath=model_path,
                                              save_weights_only=False,
                                              save_best_only=True)
        # Load the weights from the saved checkpoint
        new_model.load_weights(model_path)

        # Compile the new model with the same optimizer, loss, and metrics as the previous model
        new_model.compile(optimizer=self.optimizer, loss=self.loss, metrics=['accuracy'])

        # Train the new model with additional epochs
        history = new_model.fit(train_generator.generate(batch_size),
                                epochs=self.epochs,
                                steps_per_epoch=train_generator.get_num_samples(batch_size=batch_size) / batch_size,
                                validation_data=test_generator.generate(batch_size),
                                validation_steps=test_generator.get_num_samples(batch_size=batch_size) / batch_size,
                                callbacks=[checkpoint_callback])

        # Save the final model
        new_model.save(new_filename)
        print(f'>>>Model evaluating...')
        score = new_model.evaluate(
            test_generator.generate(batch_size),
            steps=test_generator.get_num_samples(batch_size=batch_size) / batch_size)
        print(f'*' * 80)
        logger.info(f'Test loss: {score[0]}')
        logger.info(f'Test accuracy: {score[1]}')
        self.save_plots(history, self.model_dir, encoder_name, network_name)


if __name__ == '__main__':
    main()
