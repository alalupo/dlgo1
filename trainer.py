import logging.config
import os
from pathlib import Path

import matplotlib.pyplot as plt
import tensorflow as tf
from keras.callbacks import ModelCheckpoint
from keras.layers.core import Dense
from keras.models import Sequential
import keras.backend as K

from dlgo.data.parallel_processor import GoDataProcessor
from dlgo.encoders.simple import SimpleEncoder
from dlgo.networks import network_types

logging.config.fileConfig('train_logging.conf')
logger = logging.getLogger('trainingLogger')


def locate_directory():
    path = Path(__file__)
    project_lvl_path = path.parent
    model_directory_name = 'checkpoints'
    data_directory = project_lvl_path.joinpath(model_directory_name)
    if not os.path.exists(data_directory):
        raise FileNotFoundError
    return str(data_directory)


def show_intro():
    print(f'*' * 80)
    print(f'Don\'t forget to clean up data directory of npy files before you start training a new model.')
    print(f'Put into the terminal the following command:')
    print(f'    find ./data/ -name \*.npy -delete')
    network_types.show_data_format()
    print(f'*' * 80)


def main():
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    logger.info('TRAINER: STARTED')
    show_intro()
    rows, cols = 19, 19
    encoder = SimpleEncoder((rows, cols))
    input_shape = (rows, cols, encoder.num_planes)
    network = network_types.SmallNetwork(input_shape)
    num_games = 20000
    epochs = 20
    logger.info(f'GAMES: {num_games}')
    logger.info(f'EPOCHS: {epochs}')
    K.clear_session()
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    batch_size = 128
    trainer = Trainer(network, encoder, optimizer, num_games, epochs, rows, cols)
    trainer.train_model(batch_size)
    logger.info('TRAINER: FINISHED')


class Trainer:
    def __init__(self, network, encoder, optimizer, num_games=100, num_epochs=5, rows=19, cols=19):
        self.network = network
        self.encoder = encoder
        self.optimizer = optimizer
        self.num_games = num_games
        self.epochs = num_epochs
        self.go_board_rows, self.go_board_cols = rows, cols
        self.num_classes = self.go_board_rows * self.go_board_cols
        self.model = self.build_model()

    def build_model(self):
        network_layers = self.network.layers()
        model = Sequential()
        for layer in network_layers:
            model.add(layer)
        model.add(Dense(self.num_classes, activation='softmax'))
        print(f'*' * 80)
        print(f'Model summary:')
        logger.info(model.summary())
        print(f'*' * 80)
        return model

    def train_model(self, batch_size=128):
        K.clear_session()
        processor = GoDataProcessor(encoder=self.encoder.name())
        generator = processor.load_go_data('train', num_samples=self.num_games, use_generator=True)
        print(f'>>>Train generator loaded')
        test_generator = processor.load_go_data('test', num_samples=self.num_games, use_generator=True)
        print(f'>>>Test generator loaded')
        checkpoint_dir = locate_directory()
        encoder_name = self.encoder.name()
        network_name = self.network.name
        print(f'>>>Model compiling...')
        self.model.compile(optimizer=self.optimizer,
                           loss='categorical_crossentropy',
                           metrics=['accuracy'])
        print(f'>>>Model fitting...')
        history = self.model.fit(
            generator.generate(batch_size, self.num_classes),
            epochs=self.epochs,
            steps_per_epoch=generator.get_num_samples() / batch_size,
            validation_data=test_generator.generate(batch_size, self.num_classes),
            validation_steps=test_generator.get_num_samples() / batch_size,
            callbacks=[
                ModelCheckpoint(checkpoint_dir + '/model_' + encoder_name + '_' + network_name + '_epoch_{epoch}.h5',
                                save_weights_only=False,
                                save_best_only=True
                                )
            ])
        print(f'>>>Model evaluating...')
        score = self.model.evaluate(
            test_generator.generate(batch_size, self.num_classes),
            steps=test_generator.get_num_samples() / batch_size)
        print(f'*' * 80)
        logger.info(f'Test loss: {score[0]}')
        logger.info(f'Test accuracy: {score[1]}')
        self.save_plots(history, checkpoint_dir, encoder_name, network_name)

    def save_plots(self, history, checkpoint_dir, encoder_name, network_name):
        loss = history.history['loss']
        val_loss = history.history['val_loss']
        epochs = range(1, len(loss) + 1)
        plt.plot(epochs, loss, 'bo', label='Train loss')
        plt.plot(epochs, val_loss, 'b', label='Validation loss')
        plt.title(f'Train and validation loss (games: {self.num_games}, epochs: {self.epochs})')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig(f'{checkpoint_dir}/graph_{encoder_name}_{network_name}_{self.num_games}_{self.epochs}_loss.png')

        plt.clf()

        acc = history.history['accuracy']
        val_acc = history.history['val_accuracy']
        plt.plot(epochs, acc, 'bo', label='Train accuracy')
        plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
        plt.title(f'Train and validation accuracy (games: {self.num_games}, epochs: {self.epochs})')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.savefig(f'{checkpoint_dir}/graph_{encoder_name}_{network_name}_{self.num_games}_{self.epochs}_accuracy.png')


if __name__ == '__main__':
    main()
