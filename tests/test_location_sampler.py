import unittest

import numpy as np

from source.data_processing.location_sampler import *


class TestSampler(unittest.TestCase):
    def test_uniform(self):
        sampler = LocationSampler(6, 6)

        Y = sampler.sample_uniform(10)
        self.assertEqual(Y.shape, (10, 2))

    def test_gaussian(self):
        sampler = LocationSampler(6, 6)
        centroids = np.array([(2, 2), (4, 2)])
        std_dev = 0.35

        Y = sampler.sample_gaussians(centroids, std_dev, 10)
        self.assertEqual(Y.shape, (10, 2))

    def test_gaussian_clipping(self):
        sampler = LocationSampler(6, 6)
        centroids = np.array([(-100, -100)])
        std_dev = 0.035

        Y = sampler.sample_gaussians(centroids, std_dev, 10)
        self.assertTrue(np.all(Y == np.zeros_like(Y)))


if __name__ == '__main__':
    unittest.main()
