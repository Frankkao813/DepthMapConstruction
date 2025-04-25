import numpy as np
import cv2  # Optional, only for loading and resizing images
import time
import matplotlib.pyplot as plt
from utils import show_images_side_by_side
from cost_function import compute_zncc, compute_ncc, compute_ssd

def get_fitting_result(volume, volume_type = "NCC"):
    if volume_type == "ssd" or volume_type == "SSD":
        return np.argmin(volume, axis=2)
    else:
        return np.argmax(volume, axis=2)
    

def display_disparity_map(disparity_map, volume_type):
    disp_norm = cv2.normalize(disparity_map, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    disp_norm = disp_norm.astype(np.uint8)
    plt.figure(figsize=(10, 5))
    plt.imshow(disp_norm, cmap='gray')
    plt.colorbar(label='Disparity')
    plt.title('Disparity Map (from '+ volume_type + ')')
    plt.axis('off')
    plt.show()

img_idx = 2
left = cv2.imread(f'./source/{img_idx}_left.png', cv2.IMREAD_GRAYSCALE)
right = cv2.imread(f'./source/{img_idx}_right.png', cv2.IMREAD_GRAYSCALE)
#left = cv2.imread('./dataset/Minoru3D/ActFigures/im0.png', cv2.IMREAD_GRAYSCALE)
#right = cv2.imread('./dataset/Minoru3D/ActFigures/im1.png', cv2.IMREAD_GRAYSCALE)
show_images_side_by_side(left, right)

window_size = 5
max_disparity = 20

# volume_type = "SSD"
# volume = compute_ssd(left, right, window_size, max_disparity)
# disparity_map = get_fitting_result(volume, volume_type)
# display_disparity_map(disparity_map, volume_type)

volume_type = "NCC"
volume = compute_ncc(left, right, window_size, max_disparity)
disparity_map = get_fitting_result(volume, volume_type)
display_disparity_map(disparity_map, volume_type)

# volume_type = "ZNCC"
# volume = compute_zncc(left, right, window_size, max_disparity)
# disparity_map = get_fitting_result(volume, volume_type)
# display_disparity_map(disparity_map, volume_type)