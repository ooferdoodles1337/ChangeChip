import cv2
import numpy as np
import os
from skimage.exposure import match_histograms

from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA

import matplotlib.pyplot as plt
import seaborn as sns

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
        assert output_directory is not None, "Output directory must be provided"
        temp_image = cv2.drawMatchesKnn(
            reference_image,
            reference_keypoints,
            input_image,
            input_keypoints,
            good_draw,
            None,
            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
        )
        os.makedirs(output_directory, exist_ok=True)
        cv2.imwrite(os.path.join(output_directory, "matching.png"), temp_image)

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
        assert output_directory is not None, "Output directory must be provided"
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
        assert output_directory is not None, "Output directory must be provided"
        cv2.imwrite(
            os.path.join(output_directory, "histogram_matched.jpg"),
            reference_image_matched,
        )
    reference_image_matched = np.asarray(reference_image_matched, dtype=np.uint8)
    return input_image, reference_image_matched


def preprocess_images(images, resize_factor=1.0, debug=False, output_directory=None):
    """
    Preprocesses a list of images by performing the following steps:
    1. Resizes the images based on the given resize factor.
    2. Applies homography to align the resized images.
    3. Performs histogram matching on the aligned images.

    Args:
        images (tuple): A tuple containing the input image and the reference image.
        resize_factor (float, optional): The factor by which to resize the images. Defaults to 1.0.
        debug (bool, optional): Whether to enable debug mode. Defaults to False.
        output_directory (str, optional): The directory to save the output images. Defaults to None.

    Returns:
        tuple: The preprocessed images.

    Example:
        >>> images = (input_image, reference_image)
        >>> preprocess_images(images, resize_factor=0.5, debug=True, output_directory='output/')
    """
    start_time = time.time()
    resized_images = resize_images(images, resize_factor)
    aligned_images = homography(
        resized_images, debug=debug, output_directory=output_directory
    )
    matched_images = histogram_matching(
        aligned_images, debug=debug, output_directory=output_directory
    )
    print("--- Preprocessing time - %s seconds ---" % (time.time() - start_time))
    return matched_images


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
    # print("\nfeature vector space size", FVS.shape)
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
    images,
    window_size,
    pca_dim_gray,
    pca_dim_rgb,
    debug=False,
    output_directory=None,
):
    """
    Compute descriptors for change detection between input_image and reference_image.

    Args:
        images (tuple): A tuple containing the input image and reference image as numpy arrays.
        window_size (int): The size of the sliding window.
        pca_dim_gray (int): The number of principal components to keep for grayscale difference.
        pca_dim_rgb (int): The number of principal components to keep for RGB difference.
        debug (bool, optional): Whether to save debug images. Defaults to False.
        output_directory (str, optional): The directory to save debug images. Defaults to None.

    Returns:
        numpy.ndarray: The computed descriptors.

    Raises:
        AssertionError: If debug is True but output_directory is not provided.

    """
    input_image, reference_image = images
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
        assert output_directory is not None, "Output directory must be provided"
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


def k_means_clustering(FVS, components, image_shape):
    """
    Perform K-means clustering on the given feature vectors.

    Args:
        FVS (array-like): The feature vectors to be clustered.
        components (int): The number of clusters (components) to create.
        image_shape (tuple): The size of the images used to reshape the change map.

    Returns:
        array-like: The change map obtained from the K-means clustering.

    """
    kmeans = KMeans(components, verbose=0)
    kmeans.fit(FVS)
    flatten_change_map = kmeans.predict(FVS)
    change_map = np.reshape(flatten_change_map, (image_shape[0], image_shape[1]))
    return change_map


# calculates the mse value for each cluster of change_map
def clustering_to_mse_values(change_map, input_image, reference_image, n):
    """
    Calculates the Mean Squared Error (MSE) values for each cluster in the change map.

    Args:
        change_map (numpy.ndarray): The change map indicating the cluster labels for each pixel.
        input_image (numpy.ndarray): The input image.
        reference_image (numpy.ndarray): The reference image.
        n (int): The number of clusters.

    Returns:
        tuple: A tuple containing two lists:
            - A list of MSE values for each cluster normalized by the maximum possible MSE (255^2).
            - A list of the number of pixels in each cluster.
    """
    mse = [0.0 for i in range(0, n)]
    size = [0 for i in range(0, n)]
    input_image = input_image.astype(int)
    reference_image = reference_image.astype(int)
    for i in range(change_map.shape[0]):
        for j in range(change_map.shape[1]):
            mse[change_map[i, j]] += np.mean(
                (input_image[i, j] - reference_image[i, j]) ** 2
            )
            size[change_map[i, j]] += 1
    return [(mse[k] / (255**2)) / size[k] for k in range(0, n)], size


def compute_change_map(
    images,
    output_directory,
    window_size,
    clusters,
    pca_dim_gray,
    pca_dim_rgb,
    debug=False,
):
    """
    Computes the change map for a pair of input images.

    Args:
        images (tuple): A tuple containing the input image and the reference image.
        output_directory (str): The directory where the output files will be saved.
        window_size (int): The size of the sliding window used for feature extraction.
        clusters (int): The number of clusters for k-means clustering.
        pca_dim_gray (int): The number of dimensions to reduce the gray channel to using PCA.
        pca_dim_rgb (int): The number of dimensions to reduce the RGB channels to using PCA.
        debug (bool, optional): Whether to enable debug mode. Defaults to False.

    Returns:
        tuple: A tuple containing the change map, the mean squared error (MSE) array, and the size array.
    """
    start_time = time.time()
    input_image, reference_image = images
    descriptors = get_descriptors(
        images,
        window_size,
        pca_dim_gray,
        pca_dim_rgb,
        debug=debug,
        output_directory=output_directory,
    )
    print("--- Feature extraction time - %s seconds ---" % (time.time() - start_time))
    # Now we are ready for clustering!
    change_map = k_means_clustering(descriptors, clusters, input_image.shape)
    print("--- K-means clustering time - %s seconds ---" % (time.time() - start_time))
    mse_array, size_array = clustering_to_mse_values(
        change_map, input_image, reference_image, clusters
    )
    print("--- MSE calculation time - %s seconds ---" % (time.time() - start_time))
    sorted_indexes = np.argsort(mse_array)
    colors_array = [
        plt.cm.jet(
            float(np.argwhere(sorted_indexes == class_).flatten()[0]) / (clusters - 1)
        )
        for class_ in range(clusters)
    ]
    colored_change_map = np.zeros(
        (change_map.shape[0], change_map.shape[1], 3), np.uint8
    )
    palette_colored_change_map = np.zeros(
        (change_map.shape[0], change_map.shape[1], 3), np.uint8
    )
    palette = sns.color_palette("Paired", clusters)
    for i in range(change_map.shape[0]):
        for j in range(change_map.shape[1]):
            colored_change_map[i, j] = (
                255 * colors_array[change_map[i, j]][0],
                255 * colors_array[change_map[i, j]][1],
                255 * colors_array[change_map[i, j]][2],
            )
            palette_colored_change_map[i, j] = [
                255 * palette[change_map[i, j]][0],
                255 * palette[change_map[i, j]][1],
                255 * palette[change_map[i, j]][2],
            ]

    if debug:
        assert output_directory is not None, "Output directory must be provided"
        cv2.imwrite(
            os.path.join(
                output_directory,
                f"window_size_{window_size}_pca_dim_gray{pca_dim_gray}_pca_dim_rgb{pca_dim_rgb}_clusters_{clusters}.jpg",
            ),
            colored_change_map,
        )
        cv2.imwrite(
            os.path.join(
                output_directory,
                f"PALETTE_window_size_{window_size}_pca_dim_gray{pca_dim_gray}_pca_dim_rgb{pca_dim_rgb}_clusters_{clusters}.jpg",
            ),
            palette_colored_change_map,
        )

    # Saving Output for later evaluation
    np.savetxt(
        os.path.join(output_directory, "clustering_data.csv"),
        change_map,
        delimiter=",",
    )
    print("--- Function End - %s seconds ---" % (time.time() - start_time))
    return change_map, mse_array, size_array


# selects the classes to be shown to the user as 'changes'.
# this selection is done by an MSE heuristic using DBSCAN clustering, to seperate the highest mse-valued classes from the others.
# the eps density parameter of DBSCAN might differ from system to system
def find_group_of_accepted_classes_DBSCAN(MSE_array, output_directory):
    """
    Finds the group of accepted classes using the DBSCAN algorithm.

    Parameters:
    - MSE_array (list): A list of mean squared error values.
    - output_directory (str): The directory where the output files will be saved.

    Returns:
    - accepted_classes (list): A list of indices of the accepted classes.

    Raises:
    - None

    """

    clustering = DBSCAN(eps=0.02, min_samples=1).fit(np.array(MSE_array).reshape(-1, 1))
    number_of_clusters = len(set(clustering.labels_))
    if number_of_clusters == 1:
        print("No significant changes are detected.")
        exit(0)
    # print(clustering.labels_)
    classes = [[] for i in range(number_of_clusters)]
    centers = [0 for i in range(number_of_clusters)]
    for i in range(len(MSE_array)):
        centers[clustering.labels_[i]] += MSE_array[i]
        classes[clustering.labels_[i]].append(i)

    centers = [centers[i] / len(classes[i]) for i in range(number_of_clusters)]
    min_class = centers.index(min(centers))
    accepted_classes = []
    for i in range(len(MSE_array)):
        if clustering.labels_[i] != min_class:
            accepted_classes.append(i)
    plt.figure()
    plt.xlabel("Index")
    plt.ylabel("MSE")
    plt.scatter(range(len(MSE_array)), MSE_array, c="red")
    # print(accepted_classes)
    # print(np.array(MSE_array)[np.array(accepted_classes)])
    plt.scatter(
        accepted_classes[:], np.array(MSE_array)[np.array(accepted_classes)], c="blue"
    )
    plt.title("K Mean Classification")
    plt.savefig(os.path.join(output_directory, "mse.png"))

    # save output for later evaluation
    np.savetxt(
        os.path.join(output_directory, "accepted_classes.csv"),
        accepted_classes,
        delimiter=",",
    )
    return [accepted_classes]


def draw_combination_on_transparent_input_image(
    classes_mse, clustering, combination, transparent_input_image
):
    """
    Draws a combination of classes on a transparent input image based on their mean squared error (MSE) order.

    Args:
        classes_mse (numpy.ndarray): Array of mean squared errors for each class.
        clustering (dict): Dictionary containing the clustering information for each class.
        combination (list): List of classes to be drawn on the image.
        transparent_input_image (numpy.ndarray): Transparent input image.

    Returns:
        numpy.ndarray: Transparent input image with the specified combination of classes drawn on it.
    """

    # HEAT MAP ACCORDING TO MSE ORDER
    sorted_indexes = np.argsort(classes_mse)
    for class_ in combination:
        index = np.argwhere(sorted_indexes == class_).flatten()[0]
        c = plt.cm.jet(float(index) / (len(classes_mse) - 1))
        for [i, j] in clustering[class_]:
            transparent_input_image[i, j] = (
                c[2] * 255,
                c[1] * 255,
                c[0] * 255,
                255,
            )  # BGR
    return transparent_input_image


def detect_changes(
    images,
    output_directory,
    output_alpha,
    window_size,
    clusters,
    pca_dim_gray,
    pca_dim_rgb,
    debug=False,
):
    """
    Detects changes between two images using PCA-Kmeans clustering and post-processing and outputs results to output_directory.

    Args:
        images (tuple): A tuple containing the input image and the reference image.
        output_directory (str): The directory where the output image will be saved.
        output_alpha (int): The alpha value for the output image transparency.
        window_size (int): The size of the sliding window used for computing change map.
        clusters (int): The number of clusters for PCA-Kmeans clustering.
        pca_dim_gray (int): The number of dimensions to reduce to for grayscale images.
        pca_dim_rgb (int): The number of dimensions to reduce to for RGB images.
        debug (bool, optional): Whether to enable debug mode. Defaults to False.

    Returns:
        None
    """
    start_time = time.time()
    input_image, reference_image_registered = images
    clustering_map, mse_array, size_array = compute_change_map(
        images,
        output_directory,
        window_size=window_size,
        clusters=clusters,
        pca_dim_gray=pca_dim_gray,
        pca_dim_rgb=pca_dim_rgb,
        debug=debug,
    )

    clustering = [[] for _ in range(clusters)]
    for i in range(clustering_map.shape[0]):
        for j in range(clustering_map.shape[1]):
            clustering[int(clustering_map[i, j])].append([i, j])
    input_image_copy = input_image.copy()
    b_channel, g_channel, r_channel = cv2.split(input_image_copy)
    alpha_channel = np.ones(b_channel.shape, dtype=b_channel.dtype) * 255
    alpha_channel[:, :] = output_alpha
    groups = find_group_of_accepted_classes_DBSCAN(mse_array, output_directory)

    for group in groups:
        transparent_input_image = cv2.merge(
            (b_channel, g_channel, r_channel, alpha_channel)
        )
        result = draw_combination_on_transparent_input_image(
            mse_array, clustering, group, transparent_input_image
        )
        cv2.imwrite(os.path.join(output_directory, "output.png"), result)
    print(
        "--- PCA-Kmeans + Post-processing time - %s seconds ---"
        % (time.time() - start_time)
    )


def pipeline(
    images,
    output_directory="output",
    resize_factor=1.0,
    output_alpha=50,
    window_size=5,
    clusters=16,
    pca_dim_gray=3,
    pca_dim_rgb=9,
    debug=False,
):
    """
    Process a series of images to detect changes and generate an output image.

    Args:
        images (list): A list of input images to process.
        output_directory (str, optional): The directory to save the output image. Defaults to "output".
        resize_factor (float, optional): The factor by which to resize the images. Defaults to 1.0.
        output_alpha (int, optional): The alpha value for the output image. Defaults to 50.
        window_size (int, optional): The size of the sliding window for change detection. Defaults to 5.
        clusters (int, optional): The number of clusters for color quantization. Defaults to 16.
        pca_dim_gray (int, optional): The number of dimensions to keep for grayscale PCA. Defaults to 3.
        pca_dim_rgb (int, optional): The number of dimensions to keep for RGB PCA. Defaults to 9.
        debug (bool, optional): Whether to enable debug mode. Defaults to False.

    Returns:
        numpy.ndarray: The output image.
    """
    os.makedirs(output_directory, exist_ok=True)
    preprocessed_images = preprocess_images(
        images,
        resize_factor=resize_factor,
        debug=debug,
        output_directory=output_directory,
    )
    detect_changes(
        preprocessed_images,
        output_directory=output_directory,
        output_alpha=output_alpha,
        window_size=window_size,
        clusters=clusters,
        pca_dim_gray=pca_dim_gray,
        pca_dim_rgb=pca_dim_rgb,
        debug=debug,
    )
    return cv2.imread(os.path.join(output_directory, "output.png"))
