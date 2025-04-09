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

    kernel = np.ones((window_size, window_size), dtype=np.float32) / window_area
    zncc_volume = np.zeros((H, W, max_disparity), dtype=np.float32)

    # Precompute left image statistics
    mean_L = cv2.filter2D(left, -1, kernel)[pad:-pad, pad:-pad]
    mean_L_sq = cv2.filter2D(left ** 2, -1, kernel)[pad:-pad, pad:-pad]
    std_L = np.sqrt(np.maximum(mean_L_sq - mean_L ** 2, 1e-5))

    for d in range(max_disparity):
        # Shift right image to the left by d pixels
        shifted = np.zeros_like(right)
        if d > 0:
            shifted[:, d:] = right[:, :-d]
        else:
            shifted = right

        mean_R = cv2.filter2D(shifted, -1, kernel)[pad:-pad, pad:-pad]
        mean_R_sq = cv2.filter2D(shifted ** 2, -1, kernel)[pad:-pad, pad:-pad]
        std_R = np.sqrt(np.maximum(mean_R_sq - mean_R ** 2, 1e-5))

        prod = cv2.filter2D(left * shifted, -1, kernel)[pad:-pad, pad:-pad]
        zncc = (prod - mean_L * mean_R) / (std_L * std_R)
        zncc_volume[:, :, d] = zncc

    return zncc_volume



left = cv2.imread('./source/1_left.jpg', cv2.IMREAD_GRAYSCALE)
right = cv2.imread('./source/1_right.jpg', cv2.IMREAD_GRAYSCALE)
show_images_side_by_side(left, right)

# incoke the function
window_size = 5
max_disparity = 64
start_time = time.time()
zncc_volume = compute_zncc_fast(left, right, window_size, max_disparity)
disparity_map = np.argmax(zncc_volume, axis=2)
end_time = time.time()
print(f"ZNCC computation time: {end_time - start_time:.4f} seconds")
# save the image
cv2.imwrite("./result/zncc_faster.png", (disparity_map * 4).astype(np.uint8))

# show the effect of different window sizes
window_sizes = [3, 5, 9, 15]
disparity_maps = []
for w in window_sizes:
    zncc = compute_zncc_fast(left, right, window_size=w, max_disparity=64)
    disparity_map = np.argmax(zncc, axis=2)
    disparity_maps.append(disparity_map)
# create subplots
fig, axes = plt.subplots(1, len(window_sizes), figsize=(15, 5))
for i, (w, disp_map) in enumerate(zip(window_sizes, disparity_maps)):
    axes[i].imshow(disp_map, cmap='gray')
    axes[i].set_title(f"Window Size: {w}")
    axes[i].axis('off')
plt.tight_layout()
plt.savefig("./result/zncc_faster_window_sizes.png")
plt.show()
