import sys
import numpy as np

np.set_printoptions(threshold=sys.maxsize)
import cv2
import time
from PCA_Kmeans import (
    compute_change_map,
    find_group_of_accepted_classes_DBSCAN,
    draw_combination_on_transparent_input_image,
)
import global_variables
import os
import argparse


def main(
    output_dir,
    input_path,
    reference_path,
    n,
    window_size,
    pca_dim_gray,
    pca_dim_rgb,
    cut,
    lighting_fix,
    use_homography,
    resize_factor,
    save_extra_stuff,
):
    """

    :param output_dir: destination directory for the output
    :param input_path: path to the input image
    :param reference_path: path to the reference image
    :param n: number of classes for clustering the diff descriptors
    :param window_size: window size for the diff descriptors
    :param pca_dim_gray: pca target dimension for the gray diff descriptor
    :param pca_dim_rgb: pca target dimension for the rgb diff descriptor
    :param cut: true to enable DXTR cropping
    :param lighting_fix: true to enable histogram matching
    :param use_homography: true to enable SIFT homography (always recommended)
    :param resize_factor: scale the input images, usually with factor smaller than 1 for faster results
    :param save_extra_stuff: save diagnostics and extra results, usually for debugging
    :return: the results are saved in output_dir
    """
    global_variables.init(output_dir, save_extra_stuff)  # setting global variables

    if use_homography:
        from registration import homography
    if lighting_fix:
        from light_differences_elimination import light_diff_elimination

    # for time estimations
    start_time = time.time()

    # read the inputs
    input_image = cv2.imread(input_path, 1)
    reference_image = cv2.imread(reference_path, 1)

    # we need the images to be the same size. resize_factor is for increasing or decreasing further the images
    new_shape = (
        int(resize_factor * 0.5 * (input_image.shape[1] + reference_image.shape[1])),
        int(resize_factor * 0.5 * (input_image.shape[0] + reference_image.shape[0])),
    )
    input_image = cv2.resize(input_image, new_shape, interpolation=cv2.INTER_AREA)
    reference_image = cv2.resize(
        reference_image, new_shape, interpolation=cv2.INTER_AREA
    )
    global_variables.set_size(new_shape[0], new_shape[1])
    if cut:
        import crop

        input_image, mask = crop.crop_image(input_image)
        reference_image, _ = crop.crop_image(reference_image)

        if global_variables.save_extra_stuff:
            cv2.imwrite(
                global_variables.output_dir + "/cropped_input_image.jpg", input_image
            )
            cv2.imwrite(
                global_variables.output_dir + "/cropped_reference_image.jpg",
                reference_image,
            )

        if use_homography:
            reference_image_registered, mask_registered, blank_pixels = homography(
                cut, input_image, reference_image, mask
            )
        else:
            reference_image_registered = reference_image

        # min_width = min(input_image.shape[:2][0], reference_image.shape[:2][0])
        # min_height = min(input_image.shape[:2][1], reference_image.shape[:2][1])

        # for i in range(min_width):
        #     for j in range(min_height):
        #         if mask_registered[i][j][0]:
        #             reference_image_registered[i][j] = 0
        #             input_image[i][j] = 0
        # cv2.imwrite(global_variables.output_dir + "/blanked_input_image.jpg", input_image)
        # cv2.imwrite(
        #     global_variables.output_dir + "/blanked_reference_image.jpg", reference_image_registered
        # )
        # print("--- Cropping time - %s seconds ---" % (time.time() - start_time))
    else:
        if use_homography:
            reference_image_registered, mask_registered, blank_pixels = homography(
                cut, input_image, reference_image, None
            )
        else:
            reference_image_registered = reference_image

    if use_homography:
        input_image[blank_pixels] = [0, 0, 0]
        reference_image_registered[blank_pixels] = [0, 0, 0]

    if global_variables.save_extra_stuff:
        cv2.imwrite(global_variables.output_dir + "/resized_blanked_1.jpg", input_image)

    if lighting_fix:
        # Using the histogram matching, only reference_image_registered is changed
        reference_image_registered = light_diff_elimination(
            input_image, reference_image_registered
        )

    print("--- Preprocessing time - %s seconds ---" % (time.time() - start_time))

    start_time = time.time()
    clustering_map, mse_array, size_array = compute_change_map(
        input_image,
        reference_image_registered,
        window_size=window_size,
        clusters=n,
        pca_dim_gray=pca_dim_gray,
        pca_dim_rgb=pca_dim_rgb,
    )

    clustering = [[] for _ in range(n)]
    for i in range(clustering_map.shape[0]):
        for j in range(clustering_map.shape[1]):
            clustering[int(clustering_map[i, j])].append([i, j])

    input_image = cv2.imread(input_path)
    input_image = cv2.resize(input_image, new_shape, interpolation=cv2.INTER_AREA)
    b_channel, g_channel, r_channel = cv2.split(input_image)
    alpha_channel = np.ones(b_channel.shape, dtype=b_channel.dtype) * 255
    # control output opacity
    alpha_channel[:, :] = 50
    groups = find_group_of_accepted_classes_DBSCAN(mse_array)
    for group in groups:
        transparent_input_image = cv2.merge(
            (b_channel, g_channel, r_channel, alpha_channel)
        )
        result = draw_combination_on_transparent_input_image(
            mse_array, clustering, group, transparent_input_image
        )
        cv2.imwrite(global_variables.output_dir + "/ACCEPTED_CLASSES" + ".png", result)

    print(
        "--- PCA-Kmeans + Post-processing time - %s seconds ---"
        % (time.time() - start_time)
    )


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Parameters for Running")
    parser.add_argument(
        "-output_dir", dest="output_dir", help="destination directory for the output"
    )
    parser.add_argument(
        "-input_path", dest="input_path", help="path to the input image"
    )
    parser.add_argument(
        "-reference_path", dest="reference_path", help="path to the reference image"
    )
    parser.add_argument(
        "-n", dest="n", help="number of classes for clustering the diff descriptors"
    )
    parser.add_argument(
        "-window_size", dest="window_size", help="window size for the diff descriptors"
    )
    parser.add_argument(
        "-pca_dim_gray",
        dest="pca_dim_gray",
        help="pca target dimension for the gray diff descriptor",
    )
    parser.add_argument(
        "-pca_dim_rgb",
        dest="pca_dim_rgb",
        help="pca target dimension for the rgb diff descriptor",
    )
    parser.add_argument(
        "-pca_target_dim",
        dest="pca_target_dim",
        help="pca target dimension for final combination of the descriptors",
    )
    parser.add_argument(
        "-cut",
        dest="cut",
        help="true to enable DXTR cropping",
        default=False,
        action="store_true",
    )
    parser.add_argument(
        "-lighting_fix",
        dest="lighting_fix",
        help="true to enable histogram matching",
        default=False,
        action="store_true",
    )
    parser.add_argument(
        "-use_homography",
        dest="use_homography",
        help="true to enable SIFT homography (always recommended)",
        default=False,
        action="store_true",
    )
    parser.add_argument(
        "-resize_factor",
        dest="resize_factor",
        help="scale the input images, usually with factor smaller than 1 for faster results",
    )
    parser.add_argument(
        "-save_extra_stuff",
        dest="save_extra_stuff",
        help="save diagnostics and extra results, usually for debugging",
        default=False,
        action="store_true",
    )
    args = parser.parse_args()
    main(
        args.output_dir,
        args.input_path,
        args.reference_path,
        int(args.n),
        int(args.window_size),
        int(args.pca_dim_gray),
        int(args.pca_dim_rgb),
        bool(args.cut),
        bool(args.lighting_fix),
        bool(args.use_homography),
        float(args.resize_factor),
        bool(args.save_extra_stuff),
    )
