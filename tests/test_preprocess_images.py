import sys
import os
import unittest
import cv2
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from changechip import *


class TestResizeImages(unittest.TestCase):

    def setUp(self):
        # Create mock images for testing
        self.input_image = np.ones((200, 300, 3), dtype=np.uint8) * 255  # White image
        self.reference_image = (
            np.ones((400, 600, 3), dtype=np.uint8) * 128
        )  # Gray image

    def test_resize_images(self):
        resize_factor = 0.5
        resized_input_image, resized_reference_image = resize_images(
            (self.input_image, self.reference_image), resize_factor
        )

        # Compute expected dimensions
        average_width = (
            self.input_image.shape[1] + self.reference_image.shape[1]
        ) * 0.5
        average_height = (
            self.input_image.shape[0] + self.reference_image.shape[0]
        ) * 0.5
        expected_shape = (
            int(resize_factor * average_width),
            int(resize_factor * average_height),
        )

        # Verify the shape of the resized images
        self.assertEqual(resized_reference_image.shape[1], expected_shape[0])
        self.assertEqual(resized_reference_image.shape[0], expected_shape[1])
        self.assertEqual(resized_input_image.shape[1], expected_shape[0])
        self.assertEqual(resized_input_image.shape[0], expected_shape[1])


class TestHomography(unittest.TestCase):

    def setUp(self):
        # Paths for homography test
        self.test_data_dir = os.path.join("tests", "test_data")
        self.output_directory = os.path.join(self.test_data_dir, "test_outputs")
        os.makedirs(self.output_directory, exist_ok=True)

        self.input_image_path = os.path.join(self.test_data_dir, "input.jpg")
        self.reference_image_path = os.path.join(self.test_data_dir, "reference.jpg")
        self.input_image = cv2.imread(self.input_image_path)
        self.reference_image = cv2.imread(self.reference_image_path)

    def test_homography(self):
        # Perform homography with debug mode enabled
        input_image, reference_image_registered = homography(
            (self.input_image, self.reference_image),
            debug=True,
            output_directory=self.output_directory,
        )

        # Verify the shape of the registered image
        self.assertEqual(reference_image_registered.shape, self.reference_image.shape)

        # Check if the registered image is not completely blank (all zeros)
        self.assertFalse(np.all(reference_image_registered == 0))

        # Verify that debug images are saved
        self.assertTrue(
            os.path.exists(os.path.join(self.output_directory, "matching.png"))
        )
        self.assertTrue(
            os.path.exists(os.path.join(self.output_directory, "aligned.png"))
        )


class TestHistogramMatching(unittest.TestCase):

    def setUp(self):
        self.test_data_dir = os.path.join("tests", "test_data")
        self.output_directory = os.path.join(self.test_data_dir, "test_outputs")

        os.makedirs(self.output_directory, exist_ok=True)

        self.input_image_path = os.path.join(self.test_data_dir, "input.jpg")
        self.reference_image_path = os.path.join(self.test_data_dir, "reference.jpg")
        self.input_image = cv2.imread(self.input_image_path)
        self.reference_image = cv2.imread(self.reference_image_path)

    def test_histogram_matching(self):

        input_image, reference_image_matched = histogram_matching(
            (self.input_image, self.reference_image),
            debug=True,
            output_directory=self.output_directory,
        )

        # Verify the shapes remain unchanged
        self.assertEqual(input_image.shape, self.input_image.shape)
        self.assertEqual(reference_image_matched.shape, self.reference_image.shape)

        # Check if the matched image is not completely blank (all zeros)
        self.assertFalse(np.all(reference_image_matched == 0))

        # Verify that debug images are saved
        self.assertTrue(
            os.path.exists(os.path.join(self.output_directory, "histogram_matched.jpg"))
        )


class TestPreprocessImages(unittest.TestCase):

    def setUp(self):
        self.test_data_dir = os.path.join("tests", "test_data")
        self.output_directory = os.path.join(self.test_data_dir, "test_outputs")

        os.makedirs(self.output_directory, exist_ok=True)

        self.input_image_path = os.path.join(self.test_data_dir, "input.jpg")
        self.reference_image_path = os.path.join(self.test_data_dir, "reference.jpg")
        self.input_image = cv2.imread(self.input_image_path)
        self.reference_image = cv2.imread(self.reference_image_path)

    def test_preprocess_images(self):
        # Prepare test data
        images = (self.input_image, self.reference_image)

        # Test preprocess_images function
        processed_images = preprocess_images(images, resize_factor=0.5)

        # Check if processed_images is not None
        self.assertIsNotNone(processed_images)


if __name__ == "__main__":
    unittest.main()
