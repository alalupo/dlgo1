import glob
import numpy as np
from keras.utils import to_categorical


class DataGenerator:
    def __init__(self, data_directory, samples):
        self.data_directory = data_directory
        self.samples = samples
        self.files = set(file_name for file_name, index in samples)
        self.num_samples = None

    def get_num_samples(self, batch_size=128, num_classes=19*19):
        if self.num_samples is not None:
            return self.num_samples
        else:
            self.num_samples = 0
            for X, y in self._generate(batch_size=batch_size, num_classes=num_classes):
                self.num_samples += X.shape[0]
            return self.num_samples

    def _generate(self, batch_size, num_classes, data_type='train'):
        for zip_file_name in self.files:
            file_name = zip_file_name.replace('.tar.gz', '') + data_type
            base = self.data_directory.joinpath(file_name + '_features_*.npy')
            for feature_file in glob.glob(str(base)):
                label_file = feature_file.replace('features', 'labels')
                x = np.load(feature_file)
                y = np.load(label_file)
                x = x.astype('float32')
                y = to_categorical(y.astype(int), num_classes)
                while x.shape[0] >= batch_size:
                    x_batch, x = x[:batch_size], x[batch_size:]
                    y_batch, y = y[:batch_size], y[batch_size:]
                    yield x_batch, y_batch

    def generate(self, batch_size=128, num_classes=19*19, data_type='train'):
        while True:
            for item in self._generate(batch_size, num_classes, data_type):
                yield item
