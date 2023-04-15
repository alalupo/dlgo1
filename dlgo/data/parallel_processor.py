import glob
import gzip
import logging
import multiprocessing
import os
import os.path
import shutil
import sys
import tarfile
from multiprocessing import get_context
from pathlib import Path

import numpy as np
from keras.utils import to_categorical

from dlgo.data.generator import DataGenerator
from dlgo.data.index_processor import KGSIndex
from dlgo.data.sampling import Sampler
from dlgo.data.mmap_np import NpArrayMapper
from dlgo.encoders.base import get_encoder_by_name
from dlgo.goboard_fast import Board, GameState, Move
from dlgo.gosgf import Sgf_game
from dlgo.gotypes import Player, Point
from dlgo.tools.file_finder import FileFinder

logger = logging.getLogger('trainingLogger')


def worker(jobinfo):
    try:
        clazz, encoder, zip_file, data_file_name, game_list, board_size = jobinfo
        clazz(encoder=encoder, board_size=board_size).process_zip(zip_file, data_file_name, game_list)
    except (KeyboardInterrupt, SystemExit):
        raise Exception('>>> Exiting child process.')


# def locate_data_directory():
#     path = Path(__file__)
#     project_lvl_path = path.parents[2]
#     data_directory_name = 'data'
#     data_directory = project_lvl_path.joinpath(data_directory_name)
#     return str(data_directory)


class GoDataProcessor:
    def __init__(self, encoder='simple', board_size=19):
        self.encoder_string = encoder
        self.board_size = board_size
        self.encoder = get_encoder_by_name(encoder, self.board_size)
        finder = FileFinder()
        self.data_dir = finder.data_dir
        self.test_ratio = 0.2

    def load_go_data(self, data_type='train', num_samples=1000):
        index = KGSIndex(data_directory=self.data_dir)
        index.download_files()
        sampler = Sampler(num_test_games=np.floor(num_samples * self.test_ratio))
        data = sampler.draw_data(data_type, num_samples)
        self.map_to_workers(data_type, data)
        print(f'load data: generator')
        return DataGenerator(self.data_dir, data)

    def unzip_data(self, zip_file_name):
        this_gz = gzip.open(self.data_dir.joinpath(zip_file_name))
        tar_file = zip_file_name[0:-3]
        this_tar = open(self.data_dir.joinpath(tar_file), 'wb')
        shutil.copyfileobj(this_gz, this_tar)
        this_tar.close()
        return tar_file

    def process_zip(self, zip_file_name, data_file_name, game_list):
        size_limit = 50000
        size_limit_crossed = False
        tar_file = self.unzip_data(zip_file_name)
        zip_file = tarfile.open(self.data_dir.joinpath(tar_file))
        name_list = zip_file.getnames()
        total_examples = self.num_total_examples(zip_file, game_list, name_list)
        shape = self.encoder.shape_for_others()
        feature_shape = np.insert(shape, 0, np.asarray([total_examples]))
        feature_shape = tuple(feature_shape)
        print(f'FEATURE SHAPE = {feature_shape}')
        if total_examples > size_limit:
            size_limit_crossed = True
            feature_mapper = NpArrayMapper(
                self.data_dir.joinpath('tmp_' + zip_file_name + '_features.npy'), feature_shape, np.float64)
            feature_mapper.create_map()
            label_mapper = NpArrayMapper(
                self.data_dir.joinpath('tmp_' + zip_file_name + '_labels.npy'), (total_examples,), np.float64)
            label_mapper.create_map()
        else:
            features = np.zeros(feature_shape)
            labels = np.zeros((total_examples,))
            print(f'FEATURES DTYPE = {features.dtype}')
            print(f'LABELS DTYPE = {labels.dtype}')

        counter = 0
        for index in game_list:
            name = name_list[index + 1]
            if not name.endswith('.sgf'):
                raise ValueError(name + ' is not a valid sgf')
            sgf_content = zip_file.extractfile(name).read()
            sgf = Sgf_game.from_string(sgf_content)

            game_state, first_move_done = self.get_handicap(sgf)

            for item in sgf.main_sequence_iter():
                color, move_tuple = item.get_move()
                point = None
                if color is not None:
                    if move_tuple is not None:
                        row, col = move_tuple
                        point = Point(row + 1, col + 1)
                        move = Move.play(point)
                    else:
                        move = Move.pass_turn()
                    if first_move_done and point is not None:
                        if size_limit_crossed:
                            feature_mapper.write_to_map(counter, self.encoder.encode(game_state))
                            label_mapper.write_to_map(counter, self.encoder.encode_point(point))
                        else:
                            features[counter] = self.encoder.encode(game_state)
                            labels[counter] = self.encoder.encode_point(point)

                        counter += 1
                    game_state = game_state.apply_move(move)
                    first_move_done = True

        feature_file_base = self.data_dir.joinpath(data_file_name + '_features_%d')
        label_file_base = self.data_dir.joinpath(data_file_name + '_labels_%d')

        chunk = 0  # Due to files with large content, split up after chunk-size
        if size_limit_crossed:
            logger.info(f'Features size: {feature_mapper.get_size()} MB')
            num_chunks = feature_mapper.num_chunks()
            for i in range(num_chunks):
                feature_file = str(feature_file_base) % chunk
                label_file = str(label_file_base) % chunk
                chunk += 1
                current_features = feature_mapper.get_chunk(i)
                current_labels = label_mapper.get_chunk(i)
                np.save(feature_file, current_features)
                np.save(label_file, current_labels)
            feature_mapper.clean_up()
            label_mapper.clean_up()
        else:
            logger.info(f'Features size: {round(features.nbytes / 1000000, 2)}')
            chunksize = 1024
            while features.shape[0] >= chunksize:
                feature_file = str(feature_file_base) % chunk
                label_file = str(label_file_base) % chunk
                chunk += 1
                current_features, features = features[:chunksize], features[chunksize:]
                current_labels, labels = labels[:chunksize], labels[chunksize:]
                np.save(feature_file, current_features)
                np.save(label_file, current_labels)
        print(f'=== Zip processing done. ===')

    # def consolidate_games(self, name, samples):
    #     files_needed = set(file_name for file_name, index in samples)
    #     file_names = []
    #     for zip_file_name in files_needed:
    #         file_name = zip_file_name.replace('.tar.gz', '') + name
    #         file_names.append(file_name)
    #
    #     feature_list = []
    #     label_list = []
    #     for file_name in file_names:
    #         file_prefix = file_name.replace('.tar.gz', '')
    #         base = self.data_dir.joinpath(file_prefix + '_features_*.npy')
    #         for feature_file in glob.glob(base):
    #             label_file = feature_file.replace('features', 'labels')
    #             x = np.load(feature_file)
    #             y = np.load(label_file)
    #             x = x.astype('float32')
    #             y = to_categorical(y.astype(int), 19 * 19)
    #             feature_list.append(x)
    #             label_list.append(y)
    #
    #     features = np.concatenate(feature_list, axis=0)
    #     labels = np.concatenate(label_list, axis=0)
    #
    #     feature_file = self.data_dir.joinpath(name)
    #     label_file = self.data_dir.joinpath(name)
    #
    #     np.save(feature_file, features)
    #     np.save(label_file, labels)
    #
    #     return features, labels

    @staticmethod
    def get_handicap(sgf):  # Get handicap stones
        go_board = Board(19, 19)
        first_move_done = False
        move = None
        game_state = GameState.new_game(19)
        if sgf.get_handicap() is not None and sgf.get_handicap() != 0:
            for setup in sgf.get_root().get_setup_stones():
                for move in setup:
                    row, col = move
                    go_board.place_stone(Player.black, Point(row + 1, col + 1))  # black gets handicap
            first_move_done = True
            game_state = GameState(go_board, Player.white, None, move)
        return game_state, first_move_done

    def map_to_workers(self, data_type, samples):
        zip_names = set()
        indices_by_zip_name = {}
        for filename, index in samples:
            zip_names.add(filename)
            if filename not in indices_by_zip_name:
                indices_by_zip_name[filename] = []
            indices_by_zip_name[filename].append(index)

        zips_to_process = []
        for zip_name in zip_names:
            base_name = zip_name.replace('.tar.gz', '')
            data_file_name = base_name + data_type
            if not os.path.isfile(self.data_dir.joinpath(data_file_name)):
                zips_to_process.append((self.__class__, self.encoder_string, zip_name,
                                        data_file_name, indices_by_zip_name[zip_name], self.board_size))

        cores = multiprocessing.cpu_count()  # Determine number of CPU cores and split work load among them
        pnum = 1  # By default pnum = cores but can be set to 1 if no multiprocessing needed
        print(f'The number of CPU: {cores}')
        print(f'The actual number of parallel processes: {pnum}')
        with get_context("spawn").Pool(processes=pnum, initializer=self.start_process) as pool:
            p = pool.map_async(worker, zips_to_process)
            try:
                _ = p.get()
            except KeyboardInterrupt:  # Caught keyboard interrupt, terminating workers
                print(f'Keyboard interrupt')
                pool.terminate()
                pool.join()
                sys.exit(-1)
        # pool = multiprocessing.Pool(processes=cores)
        # p = pool.map_async(worker, zips_to_process)
        # try:
        #     _ = p.get()
        # except KeyboardInterrupt:  # Caught keyboard interrupt, terminating workers
        #     pool.terminate()
        #     pool.join()
        #     sys.exit(-1)

    def num_total_examples(self, zip_file, game_list, name_list):
        total_examples = 0
        for index in game_list:
            name = name_list[index + 1]
            if name.endswith('.sgf'):
                sgf_content = zip_file.extractfile(name).read()
                sgf = Sgf_game.from_string(sgf_content)
                game_state, first_move_done = self.get_handicap(sgf)

                num_moves = 0
                for item in sgf.main_sequence_iter():
                    color, move = item.get_move()
                    if color is not None:
                        if first_move_done:
                            num_moves += 1
                        first_move_done = True
                total_examples = total_examples + num_moves
            else:
                raise ValueError(name + ' is not a valid sgf')
        return total_examples

    @staticmethod
    def start_process():
        print(f'Starting {multiprocessing.current_process().name}')
