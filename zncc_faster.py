import numpy as np
import cv2  # Optional, only for loading and resizing images
import time
import matplotlib.pyplot as plt
from utils import show_images_side_by_side
import numpy as np
import cv2


def compute_zncc_fast(left_img, right_img, window_size, max_disparity):
    """
    Fast ZNCC computation using convolution (cv2.filter2D)
    """
    assert left_img.shape == right_img.shape
    H, W = left_img.shape
    window_area = window_size ** 2
    pad = window_size // 2

    # Convert to float32 for precision and padding
    left = np.pad(left_img.astype(np.float32), pad, mode='constant')
    right = np.pad(right_img.astype(np.float32), pad, mode='constant')

    sum = np.ones((window_size, window_size), dtype=np.float32)
    mean = np.ones((window_size, window_size), dtype=np.float32) / window_area
    zncc_volume = np.zeros((H, W, max_disparity), dtype=np.float32)
    
    # Precompute image statistics
    mean_L = cv2.filter2D(left, -1, mean)
    std_L = cv2.filter2D((left - mean_L)**2, -1, sum)[pad:-pad, pad:-pad]
    err_L = cv2.filter2D((left - mean_L), -1, sum)[pad:-pad, pad:-pad]
    mean_R = cv2.filter2D(right, -1, mean)
    # fixed: mean_L -> mean_R
    std_R = cv2.filter2D((right - mean_R)**2, -1, sum)[pad:-pad, pad:-pad]
    err_R = cv2.filter2D((right - mean_R), -1, sum)[pad:-pad, pad:-pad]

    shift = lambda arr: np.pad(arr, ((0, 0), (0, 1)), mode='constant')[:, 1:]

    for d in range(max_disparity):
        std_R = shift(std_R)
        err_R = shift(err_R)
    
        zncc = (err_L * err_R) / np.sqrt(std_L * std_R)

        zncc_volume[:, :, d] = zncc

    print(f'meanl shape: {mean_L.shape}')
    print(f'zncc shape: {zncc.shape}')
    return zncc_volume


img_idx = 2
left = cv2.imread(f'./source/{img_idx}_left.png', cv2.IMREAD_GRAYSCALE)
right = cv2.imread(f'./source/{img_idx}_right.png', cv2.IMREAD_GRAYSCALE)
show_images_side_by_side(left, right)


# show the effect of different window sizes
window_sizes = [3, 5, 9, 15]
disparity_maps = []
time_list = []

for w in window_sizes:
    start_time = time.time()
    zncc = compute_zncc_fast(left, right, window_size=w, max_disparity=64)
    disparity_map = np.argmax(zncc, axis=2)
    disparity_maps.append(disparity_map)
    end_time = time.time()
    time_list.append(end_time - start_time)
    print(f"ZNCC computation time for window size {w}: {end_time - start_time:.4f} seconds")
# create subplots
fig, axes = plt.subplots(1, len(window_sizes), figsize=(15, 5))
for i, (w, disp_map) in enumerate(zip(window_sizes, disparity_maps)):
    axes[i].imshow(disp_map, cmap='gray')
    axes[i].set_title(f"Window Size: {w}")
    axes[i].axis('off')
plt.tight_layout()
plt.savefig(f"./result/zncc_faster_window_{img_idx}.png")
plt.show()
