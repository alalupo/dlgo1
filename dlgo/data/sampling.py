# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at http://mozilla.org/MPL/2.0/.
import os
import random
import logging
from dlgo.data.index_processor import KGSIndex
from dlgo.tools.file_finder import FileFinder
from six.moves import range

logger = logging.getLogger('trainingLogger')


class Sampler:
    """Sample training and test data from zipped sgf files such that test data is kept stable."""

    def __init__(self, num_test_games=100, cap_year=2014, seed=1337):
        self.finder = FileFinder()
        self.data_dir = self.finder.data_dir
        self.num_test_games = num_test_games
        self.test_games = []
        self.train_games = []
        self.test_folder = self.finder.project_path.joinpath('test_samples.py')
        self.cap_year = cap_year
        random.seed(seed)
        # self.print_available_games()
        self.compute_test_samples()

    def draw_data(self, data_type, num_samples):
        if data_type == 'test':
            self.print_test_games()
            return self.test_games
        elif data_type == 'train' and num_samples is not None:
            return self.draw_training_samples(num_samples)
        elif data_type == 'train' and num_samples is None:
            return self.draw_all_training()
        else:
            raise ValueError(data_type + " is not a valid data type, choose from 'train' or 'test'")

    def draw_test_samples(self, num_sample_games):
        """Draw num_sample_games many training games from index."""
        available_games = []
        index = KGSIndex(data_directory=self.data_dir)

        for fileinfo in index.file_info:
            filename = fileinfo['filename']
            year = int(filename.split('-')[1].split('_')[0])
            if year <= self.cap_year:
                continue
            num_games = fileinfo['num_games']
            for i in range(num_games):
                available_games.append((filename, i))
        logger.debug(f'Total number of available test games: {len(available_games)}')

        sample_set = set()
        while len(sample_set) < num_sample_games:
            sample = random.choice(available_games)
            if sample not in sample_set:
                sample_set.add(sample)
        logger.debug(f'Drawn {num_sample_games} test samples')
        return list(sample_set)

    def draw_training_games(self):
        """Get list of all non-test games, that are no later than dec 2014
        Ignore games after cap_year to keep training data stable
        """
        index = KGSIndex(data_directory=self.data_dir)
        for file_info in index.file_info:
            filename = file_info['filename']
            year = int(filename.split('-')[1].split('_')[0])
            if year > self.cap_year:
                continue
            num_games = file_info['num_games']
            for i in range(num_games):
                sample = (filename, i)
                if sample not in self.test_games:
                    self.train_games.append(sample)
        logger.debug(f'Total num training games: {len(self.train_games)}')

    def compute_test_samples(self):
        """If not already existing, create local file to store fixed set of test samples"""
        if not os.path.isfile(self.test_folder):
            test_games = self.draw_test_samples(self.num_test_games)
            test_sample_file = open(self.test_folder, 'w')
            for sample in test_games:
                test_sample_file.write(str(sample) + "\n")
            test_sample_file.close()

        test_sample_file = open(self.test_folder, 'r')
        sample_contents = test_sample_file.read()
        test_sample_file.close()
        for line in sample_contents.split('\n'):
            if line != "":
                (filename, index) = eval(line)
                self.test_games.append((filename, index))

    def draw_training_samples(self, num_sample_games):
        """Draw training games, not overlapping with any of the test games."""
        available_games = []
        index = KGSIndex(data_directory=self.data_dir)
        for fileinfo in index.file_info:
            filename = fileinfo['filename']
            year = int(filename.split('-')[1].split('_')[0])
            if year > self.cap_year:
                continue
            num_games = fileinfo['num_games']
            for i in range(num_games):
                available_games.append((filename, i))
        logger.debug(f'Total number of available training games: {len(available_games)}')

        sample_set = set()
        while len(sample_set) < num_sample_games:
            sample = random.choice(available_games)
            if sample not in self.test_games:
                sample_set.add(sample)
        logger.debug(f'Drawn {num_sample_games} training samples.')
        return list(sample_set)

    def draw_all_training(self):
        """Draw all available training games."""
        available_games = []
        index = KGSIndex(data_directory=self.data_dir)

        for fileinfo in index.file_info:
            filename = fileinfo['filename']
            year = int(filename.split('-')[1].split('_')[0])
            if year > self.cap_year:
                continue
            if 'num_games' in fileinfo.keys():
                num_games = fileinfo['num_games']
            else:
                continue
            for i in range(num_games):
                available_games.append((filename, i))
        logger.debug(f'Total num games: {len(available_games)}')

        sample_set = set()
        for sample in available_games:
            if sample not in self.test_games:
                sample_set.add(sample)
        logger.debug(f'Drawn all samples, ie : {len(sample_set)} samples.')
        return list(sample_set)

    def print_test_games(self):
        latest_year = 2000
        earliest_year = 2020
        for test_game in self.test_games:
            filename = test_game[0]
            year = int(filename.split('-')[1].split('_')[0])
            if year > latest_year:
                latest_year = year
            if year < earliest_year:
                earliest_year = year
        logger.debug(f'The latest year in the test set: {latest_year}')
        logger.debug(f'The earliest year in the test set: {earliest_year}')

    def print_available_games(self):
        """Draw num_sample_games many training games from index."""
        available_games = 0
        index = KGSIndex(data_directory=self.data_dir)

        for fileinfo in index.file_info:
            num_games = fileinfo['num_games']
            available_games += num_games
        logger.info(f'>>>Total number of KGS games: {available_games}')

