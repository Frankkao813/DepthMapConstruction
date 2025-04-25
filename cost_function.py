import numpy as np
import cv2

def compute_zncc(left_img, right_img, window_size, max_disparity):
    assert left_img.shape == right_img.shape
    H, W = left_img.shape
    pad = window_size // 2

    left = np.pad(left_img.astype(np.float32), pad, mode='reflect')
    right = np.pad(right_img.astype(np.float32), pad, mode='reflect')

    mean_kernel = np.ones((window_size, window_size), dtype=np.float32) / (window_size ** 2)
    sum_kernel = np.ones((window_size, window_size), dtype=np.float32)

    mean_L = cv2.filter2D(left, -1, mean_kernel)
    zero_mean_L = left - mean_L
    std_L = cv2.filter2D(zero_mean_L ** 2, -1, sum_kernel)[pad:-pad, pad:-pad]

    mean_R = cv2.filter2D(right, -1, mean_kernel)
    zero_mean_R = right - mean_R
    std_R = cv2.filter2D(zero_mean_R ** 2, -1, sum_kernel)

    zncc_volume = np.zeros((H, W, max_disparity), dtype=np.float32)

    for d in range(max_disparity):

        local_zero_mean_R = np.pad(zero_mean_R[:, d:], ((0, 0), (0, d)), mode='constant', constant_values=0)
        local_std_R = np.pad(std_R[:, d:], ((0, 0), (0, d)), mode='constant', constant_values=0)[pad:-pad, pad:-pad]

        numerator = cv2.filter2D(zero_mean_L * local_zero_mean_R, -1, sum_kernel)[pad:-pad, pad:-pad]

        denominator = np.sqrt(std_L * local_std_R)

        zncc = np.zeros_like(denominator)
        valid = denominator > 1e-6
        zncc[valid] = numerator[valid] / denominator[valid]

        zncc_volume[:, :, d] = zncc

    return zncc_volume

def compute_ncc(left_img, right_img, window_size, max_disparity):
    assert left_img.shape == right_img.shape
    H, W = left_img.shape
    window_area = window_size ** 2
    pad = window_size // 2

    # Convert to float32 and pad
    left = np.pad(left_img.astype(np.float32), pad, mode='constant')
    right = np.pad(right_img.astype(np.float32), pad, mode='constant')

    # Define convolution kernels
    sum_kernel = np.ones((window_size, window_size), dtype=np.float32)

    # Precompute left image statistics
    left_sq = left ** 2
    sum_L = cv2.filter2D(left, -1, sum_kernel)[pad:-pad, pad:-pad]
    sum_L2 = cv2.filter2D(left_sq, -1, sum_kernel)[pad:-pad, pad:-pad]

    # Allocate NCC volume
    ncc_volume = np.zeros((H, W, max_disparity), dtype=np.float32)

    # Define shift function to simulate disparity
    shift = lambda arr: np.pad(arr, ((0, 0), (0, 1)), mode='constant')[:, 1:]

    # Compute NCC for each disparity level
    for d in range(max_disparity):
        # Shift right image and compute statistics
        right_shifted = right
        right_sq = right_shifted ** 2
        sum_R = cv2.filter2D(right, -1, sum_kernel)[pad:-pad, pad:-pad]
        sum_R2 = cv2.filter2D(right_sq, -1, sum_kernel)[pad:-pad, pad:-pad]

        sum_LR = cv2.filter2D(left * right_shifted, -1, sum_kernel)[pad:-pad, pad:-pad]

        numerator = sum_LR
        denominator = np.sqrt(np.maximum(sum_L2 * sum_R2, 1e-8))
        ncc = numerator / denominator

        ncc_volume[:, :, d] = ncc

        # Shift right image one pixel left for next disparity
        right = shift(right)

    return ncc_volume

def compute_ssd(left_img, right_img, window_size, max_disparity):
    assert left_img.shape == right_img.shape
    H, W = left_img.shape
    pad = window_size // 2

    # Convert to float32 and pad
    left = np.pad(left_img.astype(np.float32), pad, mode='constant')
    right = np.pad(right_img.astype(np.float32), pad, mode='constant')

    kernel = np.ones((window_size, window_size), dtype=np.float32)

    ssd_volume = np.zeros((H, W, max_disparity), dtype=np.float32)

    shift = lambda arr: np.pad(arr, ((0, 0), (0, 1)), mode='constant')[:, 1:]

    for d in range(max_disparity):
        diff = (left - right) ** 2

        ssd = cv2.filter2D(diff, -1, kernel)[pad:-pad, pad:-pad]

        ssd_volume[:, :, d] = ssd

        right = shift(right)

    return ssd_volume