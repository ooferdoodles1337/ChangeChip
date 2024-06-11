import sys
import os
import unittest
import cv2
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from changechip import *


class TestPipeline(unittest.TestCase):

    def test_pipeline(self):
        images = (
            cv2.imread(os.path.join("tests", "test_data", "input.jpg")),
            cv2.imread(os.path.join("tests", "test_data", "reference.jpg")),
        )
        output = pipeline(images, "example_output", resize_factor=0.5, debug=True)
        self.assertIsNotNone(output)
        self.assertIsInstance(output, np.ndarray)


if __name__ == "__main__":
    unittest.main()
