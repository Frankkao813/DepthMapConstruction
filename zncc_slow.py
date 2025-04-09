import numpy as np
import cv2  # Optional, only for loading and resizing images
import time
import matplotlib.pyplot as plt
from utils import show_images_side_by_side

def compute_zncc_slow(left_img, right_img, window_size, max_disparity):
    # first check that if the image is of the same size
    assert left_img.shape == right_img.shape
    H, W = left_img.shape
    # since we are extracting K /times K patches, for the edge pixels we apply  padding
    padding = window_size // 2

    # image pad
    left_padded = np.pad(left_img, padding, mode="constant")
    right_padded = np.pad(right_img, padding, mode="constant")

    zncc_3d = np.zeros((H, W, max_disparity), dtype=np.float64)

    # for each disparity and [x, y] pixel, compute the zncc

    for d in range(max_disparity):
        for rows in range(padding, H):
            for cols in range(padding, W):
                # left patch
                patch_left = left_padded[rows - padding: rows + padding + 1,
                                        cols - padding: cols + padding + 1]
                mean_left = np.mean(patch_left)
                shifted_left = patch_left - mean_left

                # right patch
                shifted_col = cols - d
                if shifted_col < padding:
                    # the shifted col is out of bound
                    continue  

                patch_right = right_padded[rows - padding: rows + padding + 1,
                                        shifted_col - padding: shifted_col + padding + 1]
                mean_right = np.mean(patch_right)
                shifted_right = patch_right - mean_right

                numerator = np.sum(shifted_left * shifted_right)
                denominator = np.sqrt(np.sum(shifted_left**2) * np.sum(shifted_right**2)) + 1e-5
                zncc = numerator / denominator

                zncc_3d[rows, cols, d] = zncc
    return zncc_3d




# invoke the function
# import the left image and right image

# first read in the two images
left = cv2.imread('./source/1_left.jpg', cv2.IMREAD_GRAYSCALE)
right = cv2.imread('./source/1_right.jpg', cv2.IMREAD_GRAYSCALE)



# Call the function to display the images
show_images_side_by_side(left, right)

# calculate the zncc_slow - start the timer

time_start = time.time()
zncc_slow = compute_zncc_slow(left, right, window_size=5, max_disparity=64)
# Get disparity map by picking disparity with highest ZNCC score
disparity_map = np.argmax(zncc_slow, axis=2)

time_end = time.time()
print("time elapsed", time_end - time_start)

#  why to multiply by 4? -> to make the disparity map more visible
cv2.imshow("Disparity Map", (disparity_map * 4).astype(np.uint8))

cv2.imwrite("./result/disparity_map.png", (disparity_map * 4).astype(np.uint8))
cv2.waitKey(0)
cv2.destroyAllWindows()