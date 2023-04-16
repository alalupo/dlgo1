import glob
import numpy as np
from keras.utils import to_categorical


class DataGenerator:
    def __init__(self, data_directory, samples, board_size, data_type):
        self.data_directory = data_directory
        self.samples = samples
        self.board_size = board_size
        self.num_classes = self.board_size * self.board_size
        self.data_type = data_type
        self.files = set(file_name for file_name, index in samples)
        self.num_samples = None

    def get_num_samples(self, batch_size=128):
        if self.num_samples is not None:
            return self.num_samples
        else:
            self.num_samples = 0
            for X, y in self._generate(batch_size=batch_size):
                self.num_samples += X.shape[0]
            return self.num_samples

    def count_files(self):
        print(f'{len(self.files)} ZIP FILES IN GENERATOR')
        samples = 0
        for zip_file_name in self.files:
            file_name = zip_file_name.replace('.tar.gz', '') + self.data_type
            base = self.data_directory.joinpath(file_name + '_features_*.npy')
            for feature_file in glob.glob(str(base)):
                samples += 1
        print(f'{samples} SAMPLES IN GENERATOR')
        return samples

    def _generate(self, batch_size):
        for zip_file_name in self.files:
            file_name = zip_file_name.replace('.tar.gz', '') + self.data_type
            base = self.data_directory.joinpath(file_name + '_features_*.npy')

            # WITH NPY MERGING:
            # arr_x = np.empty(shape=(0, 19, 19, 11))
            # arr_y = np.empty(shape=(0,))
            # for feature_file in glob.glob(str(base)):
            #     x = np.load(feature_file)
            #     arr_x = np.concatenate((arr_x, x))
            #     label_file = feature_file.replace('features', 'labels')
            #     y = np.load(label_file)
            #     arr_y = np.concatenate((arr_y, y))
            # arr_x = arr_x.astype('float32')
            # arr_y = to_categorical(arr_y.astype(int), self.num_classes)
            # while arr_x.shape[0] >= batch_size:
            #     x_batch, arr_x = arr_x[:batch_size], arr_x[batch_size:]
            #     y_batch, arr_y = arr_y[:batch_size], arr_y[batch_size:]
            #     yield x_batch, y_batch

            for feature_file in glob.glob(str(base)):
                label_file = feature_file.replace('features', 'labels')
                x = np.load(feature_file)
                y = np.load(label_file)
                x = x.astype('float32')
                y = to_categorical(y.astype(int), self.num_classes)
                while x.shape[0] >= batch_size:
                    x_batch, x = x[:batch_size], x[batch_size:]
                    y_batch, y = y[:batch_size], y[batch_size:]
                    yield x_batch, y_batch

    def generate(self, batch_size=128):
        while True:
            for item in self._generate(batch_size):
                yield item
