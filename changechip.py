import cv2
import numpy as np
import os
from skimage.exposure import match_histograms

from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA

import time


def resize_images(images, resize_factor=1.0):
    """
    Resizes the input and reference images based on the average dimensions of the two images and a resize factor.

    Parameters:
    images (tuple): A tuple containing two images (input_image, reference_image). Both images should be numpy arrays.
    resize_factor (float): A factor by which to resize the images. Default is 1.0, which means the images will be resized to the average dimensions of the two images.

    Returns:
    tuple: A tuple containing the resized input and reference images.

    Example:
    >>> input_image = cv2.imread('input.jpg')
    >>> reference_image = cv2.imread('reference.jpg')
    >>> resized_images = resize_images((input_image, reference_image), resize_factor=0.5)
    """
    input_image, reference_image = images
    average_width = (input_image.shape[1] + reference_image.shape[1]) * 0.5
    average_height = (input_image.shape[0] + reference_image.shape[0]) * 0.5
    new_shape = (
        int(resize_factor * average_width),
        int(resize_factor * average_height),
    )
    print(new_shape)

    input_image = cv2.resize(input_image, new_shape, interpolation=cv2.INTER_AREA)
    reference_image = cv2.resize(
        reference_image, new_shape, interpolation=cv2.INTER_AREA
    )

    return input_image, reference_image


def homography(images, debug=False, output_directory=None):
    """
    Apply homography transformation to align two images.

    Args:
        images (tuple): A tuple containing two images, where the first image is the input image and the second image is the reference image.
        debug (bool, optional): If True, debug images will be generated. Defaults to False.
        output_directory (str, optional): The directory to save the debug images. Defaults to None.

    Returns:
        tuple: A tuple containing the aligned input image and the reference image.
    """
    input_image, reference_image = images
    # Initiate SIFT detector
    sift = cv2.SIFT_create()
    # find the keypoints and descriptors with SIFT

    input_keypoints, input_descriptors = sift.detectAndCompute(input_image, None)
    reference_keypoints, reference_descriptors = sift.detectAndCompute(
        reference_image, None
    )
    # BFMatcher with default params
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(reference_descriptors, input_descriptors, k=2)

    # Apply ratio test
    good_draw = []
    good_without_list = []
    for m, n in matches:
        if m.distance < 0.8 * n.distance:  # 0.8 = a value suggested by David G. Lowe.
            good_draw.append([m])
            good_without_list.append(m)

    # cv.drawMatchesKnn expects list of lists as matches.
    if debug:
        temp_image = cv2.drawMatchesKnn(
            reference_image,
            reference_keypoints,
            input_image,
            input_keypoints,
            good_draw,
            None,
            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
        )
        if output_directory:
            os.makedirs(output_directory, exist_ok=True)
            debug_image_path = os.path.join(output_directory, "matching.png")
            cv2.imwrite(debug_image_path, temp_image)

    # Extract location of good matches
    reference_points = np.zeros((len(good_without_list), 2), dtype=np.float32)
    input_points = reference_points.copy()

    for i, match in enumerate(good_without_list):
        input_points[i, :] = reference_keypoints[match.queryIdx].pt
        reference_points[i, :] = input_keypoints[match.trainIdx].pt

    # Find homography
    h, _ = cv2.findHomography(input_points, reference_points, cv2.RANSAC)

    # Use homography
    height, width = reference_image.shape[:2]
    white_reference_image = 255 - np.zeros(shape=reference_image.shape, dtype=np.uint8)
    white_reg = cv2.warpPerspective(white_reference_image, h, (width, height))
    blank_pixels_mask = np.any(white_reg != [255, 255, 255], axis=-1)
    reference_image_registered = cv2.warpPerspective(
        reference_image, h, (width, height)
    )
    if debug:
        cv2.imwrite(
            os.path.join(output_directory, "aligned.png"), reference_image_registered
        )

    input_image[blank_pixels_mask] = [0, 0, 0]
    reference_image_registered[blank_pixels_mask] = [0, 0, 0]

    return input_image, reference_image_registered


def histogram_matching(images, debug=False, output_directory=None):
    """
    Perform histogram matching between an input image and a reference image.

    Args:
        images (tuple): A tuple containing the input image and the reference image.
        debug (bool, optional): If True, save the histogram-matched image to the output directory. Defaults to False.
        output_directory (str, optional): The directory to save the histogram-matched image. Defaults to None.

    Returns:
        tuple: A tuple containing the input image and the histogram-matched reference image.
    """

    input_image, reference_image = images

    reference_image_matched = match_histograms(
        reference_image, input_image, channel_axis=-1
    )
    if debug:
        cv2.imwrite(
            os.path.join(output_directory, "histogram_matched.jpg"),
            reference_image_matched,
        )
    reference_image_matched = np.asarray(reference_image_matched, dtype=np.uint8)
    return input_image, reference_image_matched


def preprocess_images(images, resize_factor=1.0):
    """
    Preprocesses a list of images by performing the following steps:
    1. Resizes the images based on the given resize factor.
    2. Applies homography to align the resized images.
    3. Performs histogram matching on the aligned images.

    Args:
        images (tuple): A tuple containing the input image and the reference image.
        resize_factor (float, optional): The factor by which to resize the images. Defaults to 1.0.

    Returns:
        tuple: The preprocessed images.
    """
    start_time = time.time()
    resized_images = resize_images(images, resize_factor)
    aligned_images = homography(resized_images)
    matched_images = histogram_matching(aligned_images)
    print("--- Preprocessing time - %s seconds ---" % (time.time() - start_time))
    return matched_images


images = (
    cv2.imread("tests/test_data/input.jpg"),
    cv2.imread("tests/test_data/reference.jpg"),
)
input_image, _ = preprocess_images(images, resize_factor=0.5)
print(input_image.shape[::-1][1:])


# The returned vector_set goes later to the PCA algorithm which derives the EVS (Eigen Vector Space).
# Therefore, there is a mean normalization of the data
# jump_size is for iterating non-overlapping windows. This parameter should be eqaul to the window_size of the system
def find_vector_set(descriptors, jump_size, shape):
    """
    Find the vector set from the given descriptors.

    Parameters:
    - descriptors: numpy.ndarray
        The input descriptors array.
    - jump_size: int
        The jump size for sampling the descriptors.
    - shape: tuple
        The shape of the descriptors array.

    Returns:
    - vector_set: numpy.ndarray
        The vector set obtained from the descriptors.
    - mean_vec: numpy.ndarray
        The mean vector of the vector set.

    """
    size_0, size_1 = shape
    descriptors_2d = descriptors.reshape((size_0, size_1, descriptors.shape[1]))
    vector_set = descriptors_2d[::jump_size, ::jump_size]
    vector_set = vector_set.reshape(
        (vector_set.shape[0] * vector_set.shape[1], vector_set.shape[2])
    )
    mean_vec = np.mean(vector_set, axis=0)
    vector_set = vector_set - mean_vec  # mean normalization
    return vector_set, mean_vec


# returns the FSV (Feature Vector Space) which then goes directly to clustering (with Kmeans)
# Multiply the data with the EVS to get the entire data in the PCA target space
def find_FVS(descriptors, EVS, mean_vec):
    """
    Calculate the feature vector space (FVS) by performing dot product of descriptors and EVS,
    and subtracting the mean vector from the result.

    Args:
        descriptors (numpy.ndarray): Array of descriptors.
        EVS (numpy.ndarray): Eigenvalue matrix.
        mean_vec (numpy.ndarray): Mean vector.

    Returns:
        numpy.ndarray: The calculated feature vector space (FVS).

    """
    FVS = np.dot(descriptors, EVS)
    FVS = FVS - mean_vec
    print("\nfeature vector space size", FVS.shape)
    return FVS


# assumes descriptors is already flattened
# returns descriptors after moving them into the PCA vector space
def descriptors_to_pca(descriptors, pca_target_dim, window_size, shape):
    """
    Applies Principal Component Analysis (PCA) to a set of descriptors.

    Args:
        descriptors (list): List of descriptors.
        pca_target_dim (int): Target dimensionality for PCA.
        window_size (int): Size of the sliding window.
        shape (tuple): Shape of the descriptors.

    Returns:
        list: Feature vector set after applying PCA.
    """
    vector_set, mean_vec = find_vector_set(descriptors, window_size, shape)
    pca = PCA(pca_target_dim)
    pca.fit(vector_set)
    EVS = pca.components_
    mean_vec = np.dot(mean_vec, EVS.transpose())
    FVS = find_FVS(descriptors, EVS.transpose(), mean_vec)
    return FVS


def get_descriptors(
    input_image,
    reference_image,
    window_size,
    pca_dim_gray,
    pca_dim_rgb,
    debug=False,
    output_directory=None,
):
    descriptors = np.zeros(
        (input_image.shape[0], input_image.shape[1], window_size * window_size)
    )
    diff_image = cv2.absdiff(input_image, reference_image)
    diff_image = cv2.cvtColor(diff_image, cv2.COLOR_BGR2GRAY)
    cv2.imwrite(os.path.join(output_directory, "diff.jpg"), diff_image)
    diff_image = np.pad(
        diff_image,
        ((window_size // 2, window_size // 2), (window_size // 2, window_size // 2)),
        "constant",
    )  # default is 0
    for i in range(input_image.shape[0]):
        for j in range(input_image.shape[1]):
            descriptors[i, j, :] = diff_image[
                i : i + window_size, j : j + window_size
            ].ravel()
    descriptors_gray_diff = descriptors.reshape(
        (descriptors.shape[0] * descriptors.shape[1], descriptors.shape[2])
    )

    #################################################   3-channels-diff (abs)

    descriptors = np.zeros(
        (input_image.shape[0], input_image.shape[1], window_size * window_size * 3)
    )
    diff_image_r = cv2.absdiff(input_image[:, :, 0], reference_image[:, :, 0])
    diff_image_g = cv2.absdiff(input_image[:, :, 1], reference_image[:, :, 1])
    diff_image_b = cv2.absdiff(input_image[:, :, 2], reference_image[:, :, 2])

    if debug:
        cv2.imwrite(
            os.path.join(output_directory, "final_diff.jpg"),
            cv2.absdiff(input_image, reference_image),
        )
        cv2.imwrite(os.path.join(output_directory, "final_diff_r.jpg"), diff_image_r)
        cv2.imwrite(os.path.join(output_directory, "final_diff_g.jpg"), diff_image_g)
        cv2.imwrite(os.path.join(output_directory, "final_diff_b.jpg"), diff_image_b)

    diff_image_r = np.pad(
        diff_image_r,
        ((window_size // 2, window_size // 2), (window_size // 2, window_size // 2)),
        "constant",
    )  # default is 0
    diff_image_g = np.pad(
        diff_image_g,
        ((window_size // 2, window_size // 2), (window_size // 2, window_size // 2)),
        "constant",
    )  # default is 0
    diff_image_b = np.pad(
        diff_image_b,
        ((window_size // 2, window_size // 2), (window_size // 2, window_size // 2)),
        "constant",
    )  # default is 0

    for i in range(input_image.shape[0]):
        for j in range(input_image.shape[1]):
            feature_r = diff_image_r[i : i + window_size, j : j + window_size].ravel()
            feature_g = diff_image_g[i : i + window_size, j : j + window_size].ravel()
            feature_b = diff_image_b[i : i + window_size, j : j + window_size].ravel()
            descriptors[i, j, :] = np.concatenate((feature_r, feature_g, feature_b))
    descriptors_rgb_diff = descriptors.reshape(
        (descriptors.shape[0] * descriptors.shape[1], descriptors.shape[2])
    )

    #################################################   concatination

    shape = input_image.shape[::-1][1:]  # I have no idea why its flipped like this
    descriptors_gray_diff = descriptors_to_pca(
        descriptors_gray_diff, pca_dim_gray, window_size, shape
    )
    descriptors_colored_diff = descriptors_to_pca(
        descriptors_rgb_diff, pca_dim_rgb, window_size, shape
    )

    descriptors = np.concatenate(
        (descriptors_gray_diff, descriptors_colored_diff), axis=1
    )

    return descriptors
    pass


def compute_change_map(
    input_image,
    reference_image,
    window_size,
    clusters,
    pca_dim_gray,
    pca_dim_rgb,
    debug=False,
    output_directory=None,
):

    descriptors = get_descriptors(
        input_image,
        reference_image,
        window_size,
        pca_dim_gray,
        pca_dim_rgb,
        debug=False,
        output_directory=None,
    )
    pass


def detect_changes(
    images,
    save_directory,
    window_size=5,
    clusters=16,
    pca_dim_gray=3,
    pca_dim_rgb=9,
    debug=False,
    output_directory=None,
):
    start_time = time.time()
    input_image, reference_image_registered = images
    pass
