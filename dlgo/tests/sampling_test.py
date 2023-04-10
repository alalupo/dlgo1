import unittest

import numpy as np

from dlgo.data.sampling import Sampler


class SamplingTest(unittest.TestCase):
    def test_sampling(self):
        num_games = 1001
        expected_num_test_games = 200
        ratio = 5
        sampler = Sampler(num_test_games=np.floor(num_games / ratio))
        train_data = sampler.draw_data('train', num_games)
        self.assertEqual(expected_num_test_games, sampler.num_test_games)
        self.assertEqual(expected_num_test_games, len(sampler.test_games))
        self.assertEqual(num_games, len(train_data))
