from pathlib import Path
import glob
import numpy as np
import tensorflow as tf

keras = tf.keras
from keras.utils import Sequence


class DataGenerator(Sequence):
    """Generates data for Keras
    """

    def __init__(self, data_directory, samples, data_type, batch_size=128, dim=(19, 19), n_channels=11,
                 n_classes=361, shuffle=True):
        """Initialization"""
        self.data_directory = data_directory
        self.samples = samples
        self.files = set(file_name for file_name, index in samples)
        self.data_type = data_type
        self.dim = dim
        self.batch_size = batch_size
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.indexes = None
        self.on_epoch_end()

    def __len__(self):
        """Denotes the number of batches per epoch"""
        return int(np.floor(len(self.samples) / self.batch_size))

    def __getitem__(self, index):
        """Generate one batch of data"""
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]

        # Find list of IDs
        samples_batch = [self.samples[k] for k in indexes]


        # Generate data
        X, y = self.__data_generation(samples_batch)
        X = X.astype('float32')
        return X, y

    def on_epoch_end(self):
        """Updates indexes after each epoch"""
        self.indexes = np.arange(len(self.samples))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __data_generation(self, samples_batch):
        """Generates data containing batch_size samples"""  # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, *self.dim, self.n_channels))
        y = np.empty((self.batch_size), dtype=int)

        # Generate data
        for i, sample in enumerate(samples_batch):
            file = sample[0]
            index = sample[1]
            file_name = file.replace('.tar.gz', '') + self.data_type
            file_paths = glob.glob(str(self.data_directory / f'{file_name}_features_{index}.npy'))
            # base = self.data_directory.joinpath(file_name + '_features_*.npy')

            # for feature_file in glob.glob(str(base)):
            #     label_file = feature_file.replace('features', 'labels')
            #     x = np.load(feature_file)
            #     x = np.transpose(x, (0, 2, 3, 1))  # channels last
            #     y = np.load(label_file)
            #     x = x.astype('float32')
            #
            # file_paths = glob.glob(str(self.data_directory / f'*_{self.data_type}_feature_{ID}.npy'))

            # iterate over the matched file paths
            for file_path in file_paths:
                X[i, ] = np.load(file_path)
                label_file = file_path.replace('features', 'labels')
                y[i] = np.load(label_file)

        return X, keras.utils.to_categorical(y, num_classes=self.n_classes)

