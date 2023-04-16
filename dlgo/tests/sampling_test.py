import unittest
from pathlib import Path

import numpy as np

from dlgo.data.sampling import Sampler


class SamplingTest(unittest.TestCase):
    def setUp(self):
        self.board_size = 19
        self.samples_file = Path('test_samples.py')
        if Path(self.samples_file).is_file():
            Path.unlink(self.samples_file)
        for path in Path("./data").glob("*.npy"):
            if Path(path).is_file():
                Path.unlink(path)

    def tearDown(self):
        # Clean up the test file
        if Path(self.samples_file).is_file():
            Path.unlink(self.samples_file)
        for path in Path("./data").glob("*.npy"):
            if Path(path).is_file():
                Path.unlink(path)

    def test_sampling(self):
        num_games = 1001
        expected_num_test_games = 200
        ratio = 5
        sampler = Sampler(num_test_games=np.floor(num_games / ratio))
        train_data = sampler.draw_data('train', num_games)
        self.assertEqual(expected_num_test_games, sampler.num_test_games)
        self.assertEqual(expected_num_test_games, len(sampler.test_games))
        self.assertEqual(num_games, len(train_data))
