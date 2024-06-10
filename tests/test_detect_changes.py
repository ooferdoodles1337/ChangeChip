import sys
import os
import unittest
import cv2
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from changechip import *


class TestFindVectorSet(unittest.TestCase):

    def test_find_vector_set(self):
        descriptors = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        jump_size = 2
        shape = (3, 3)
        expected_vector_set = np.array([[1, 2, 3], [7, 8, 9]])
        expected_mean_vec = np.array([4, 5, 6])

        vector_set, mean_vec = find_vector_set(descriptors, jump_size, shape)

        self.assertTrue(np.array_equal(vector_set, expected_vector_set))
        self.assertTrue(np.array_equal(mean_vec, expected_mean_vec))


if __name__ == "__main__":
    unittest.main()
