
import numpy as np
import cv2
import global_variables

def light_diff_elimination_NAIVE(image_1, image_2_registered):
    img_hsv1 = cv2.cvtColor(image_1, cv2.COLOR_RGB2HSV)
    img_cpy1 = np.copy(img_hsv1)
    img_hsv2 = cv2.cvtColor(image_2_registered, cv2.COLOR_RGB2HSV)
    img_cpy2 = np.copy(img_hsv2)
    for i in range(img_hsv1.shape[0]):
        for j in range(img_hsv1.shape[1]):
            if img_cpy1[i, j, 1] >= 50 or img_cpy1[i, j, 2] <= 205:
                img_cpy1[i, j, 1] = (img_hsv1[i, j, 1] + img_hsv2[i, j, 1])
    for i in range(img_hsv1.shape[0]):
        for j in range(img_hsv1.shape[1]):
            if img_cpy2[i, j, 1] >= 50 or img_cpy2[i, j, 2] <= 205:
                img_cpy2[i, j, 1] = (img_hsv1[i, j, 1] + img_hsv2[i, j, 1])
    # image_1[:, :, 1] = 120
    # image_2_registered[:, :, 1] = 120 vc
    image_1 = cv2.cvtColor(img_cpy1, cv2.COLOR_HSV2RGB)
    if (global_variables.save_extra_stuff):
        cv2.imwrite(global_variables.output_dir + '/img1_light_correction.jpg', image_1)
    image_2_registered = cv2.cvtColor(img_cpy2, cv2.COLOR_HSV2RGB)
    if (global_variables.save_extra_stuff):
        cv2.imwrite(global_variables.output_dir + '/img2_light_correction.jpg',
                image_2_registered)
    return image_1, image_2_registered

#rgb - are the images in rgb colors of just gray?
def light_diff_elimination(image_1, image_2_registered):
    from ExactHistogramMatching.histogram_matching import ExactHistogramMatcher
    reference_histogram = ExactHistogramMatcher.get_histogram(image_1)
    new_target_img = ExactHistogramMatcher.match_image_to_histogram(image_2_registered, reference_histogram)
    cv2.imwrite(global_variables.output_dir + '/image_2_registered_histogram_matched.jpg', new_target_img)
    new_target_img = np.asarray(new_target_img, dtype=np.uint8)
    return new_target_img