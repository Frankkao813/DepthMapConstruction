import numpy as np
import cv2  # Optional, only for loading and resizing images
import time
import matplotlib.pyplot as plt
#from utils import show_images_side_by_side
import numpy as np
import cv2
from numba import njit
import os
from numpy.lib.stride_tricks import sliding_window_view


DIRECTION_OFFSET = {
    "left-right": (0, -1),
    "right-left": (0, 1),
    "up-down": (-1, 0),
    "down-up": (1, 0),
    "top-right-bottom-left": (-1, 1),
    "bottom-left-top-right": (1, -1),
    "bottom-right-top-left": (1, 1),
    "top-left-bottom-right": (-1, -1)
}



def compute_zncc(left_img, right_img, window_size, max_disparity):
    """
    Compute ZNCC (Zero-mean Normalized Cross-Correlation) volume between two images.

    Args:
        left_img (np.ndarray): Left grayscale image (H, W).
        right_img (np.ndarray): Right grayscale image (H, W).
        window_size (int): Size of the local window (must be odd, e.g., 3, 5, 7).
        max_disparity (int): Maximum disparity (number of shifts to search).

    Returns:
        np.ndarray: ZNCC cost volume of shape (H, W, max_disparity).
    """
    assert left_img.shape == right_img.shape, "Input images must have the same shape"
    H, W = left_img.shape
    pad = window_size // 2

    # Convert to float32 and apply padding to handle window boundaries
    left = np.pad(left_img.astype(np.float32), pad, mode='reflect')
    right = np.pad(right_img.astype(np.float32), pad, mode='reflect')

    # Create sliding windows for the left image
    # Shape becomes (H, W, window_size, window_size)
    left_patches = sliding_window_view(left, (window_size, window_size))

    # zncc volume intialization
    zncc_volume = np.zeros((H, W, max_disparity), dtype=np.float32)

    # iterate over all disparity levels
    for d in range(max_disparity):
        # Shift the right image by disparity d (shift left by d pixels)
        # Pad zeros on the right to maintain original size
        right_shifted = np.pad(right[:, d:], ((0, 0), (0, d)), mode='constant', constant_values=0)

        # Create sliding windows for the shifted right image
        right_patches = sliding_window_view(right_shifted, (window_size, window_size))

        # Compute mean value for each window (broadcasted across patch dimensions)
        mean_L = np.mean(left_patches, axis=(-2, -1), keepdims=True)  # shape (H, W, 1, 1)
        mean_R = np.mean(right_patches, axis=(-2, -1), keepdims=True)

        # Subtract mean from each window to zero-center the patches
        left_zero_mean = left_patches - mean_L
        right_zero_mean = right_patches - mean_R

        # numerator: sum of element-wise product over each window
        numerator = np.sum(left_zero_mean * right_zero_mean, axis=(-2, -1))

        # denominator: product of L2 norms (square root of summed squares)
        denom_L = np.sum(left_zero_mean ** 2, axis=(-2, -1))
        denom_R = np.sum(right_zero_mean ** 2, axis=(-2, -1))
        denominator = np.sqrt(denom_L * denom_R)

        # division by zero safely
        zncc = np.zeros_like(numerator)
        valid = denominator > 1e-8 
        zncc[valid] = numerator[valid] / denominator[valid]

        # store the ZNCC score for disparity d
        zncc_volume[:, :, d] = zncc

    return zncc_volume




def left_right_order(size):
    """
    Compute the left-right order of the pixels in the image
    """
    H, W = size
    order = []
    for i in range(H):
        for j in range(W):
            order.append((i, j))
    order = np.array(order)
    #print("left-right order", order)
    return order

def right_left_order(size):
    """
    Compute the right-left order of the pixels in the image
    """
    H, W = size
    order = []
    for i in range(H):
        for j in range(W-1, -1, -1):
            order.append((i, j))
    order = np.array(order)
    #print("right-left order", order)
    return order

def up_down_order(size):
    """
    Compute the up-down order of the pixels in the image
    """
    H, W = size
    order = []
    for i in range(W):
        for j in range(H):
            order.append((j, i))
    order = np.array(order)
    #print("up-down order", order)
    return order

def down_up_order(size):
    """
    Compute the down-up order of the pixels in the image
    """
    H, W = size
    order = []
    for i in range(W):
        for j in range(H-1, -1, -1):
            order.append((j, i))
    order = np.array(order)
    #print("up-down order", order)
    return order

# deal with the slanted line order
def top_right_bottom_left_order(size):
    """
    Compute the top-right-bottom-left order of the pixels in the image
    """
    H, W = size
    order = []

    # Loop over diagonals from top-right to bottom-left
    for diag in range(-(W - 1), H):  # y - x = diag
        for y in range(H):
            x = y - diag
            if 0 <= x < W:
                order.append((y, x))

    order = np.array(order)
    #print("top-right-bottom-left order", order)
    return order

def bottom_left_top_right_order(size):
    """
    Compute the bottom-left-top-right order of the pixels in the image
    """
    H, W = size
    order = []

    # Loop over diagonals from bottom-left to top-right
    for diag in range(H + W - 1):  # y + x = diag
        for y in range(H):
            x = diag - y
            if 0 <= x < W:
                order.append((y, x))

    order = np.array(order)
    #print("bottom-left-top-right order", order)
    return order

def bottom_right_top_left_order(size):
    """
    Compute the bottom-right to top-left (↖) scan order of the pixels,
    starting from (H-1, W-1), based on diagonals where y + x is constant.
    """
    H, W = size
    order = []

    # Loop over diagonals in reverse order (H+W-2 down to 0)
    for diag in range(H + W - 2, -1, -1):  # y + x = diag
        for y in range(H - 1, -1, -1):     # from bottom to top
            x = diag - y
            if 0 <= x < W:
                order.append((y, x))

    order = np.array(order)
    #print("bottom-right-top-left order", order)
    return order

# the direciton of top left to bottom right
def top_left_bottom_right_order(size):
    """
    Compute the top-left to bottom-right (↘) scan order of the pixels,
    starting from (0, 0), based on diagonals where y + x is constant.
    """
    H, W = size
    order = []

    # Loop over diagonals from 0 to H+W-2
    for diag in range(H + W - 1):  # y + x = diag
        for y in range(H):
            x = diag - y
            if 0 <= x < W:
                order.append((y, x))

    order = np.array(order)
    #print("top-left-bottom-right order", order)
    return order

# https://medium.com/@weidagang/accelerated-python-numba-3e6a88335f83
@njit
def aggregate_costs_numba(zncc_volume, order, dy, dx, P1=1.0, P2=3.0):
    H, W, D = zncc_volume.shape
    L = np.zeros((H, W, D), dtype=np.float32)

    for n in range(order.shape[0]):
        row, col = order[n]
        prev_row = row + dy
        prev_col = col + dx

        if 0 <= prev_row < H and 0 <= prev_col < W:
            prev_vals = L[prev_row, prev_col, :]
            min_prev = np.min(prev_vals)

            for d in range(D):
                d0 = prev_vals[d]
                d1 = prev_vals[d - 1] + P1 if d > 0 else 1e9
                d2 = prev_vals[d + 1] + P1 if d < D - 1 else 1e9
                d3 = min_prev + P2
                L[row, col, d] = zncc_volume[row, col, d] + min(d0, d1, d2, d3) - min_prev
        else:
            for d in range(D):
                L[row, col, d] = zncc_volume[row, col, d]

    return L


def compute_sgm_fast(zncc_volume, direction, P1=1.0, P2=3.0):
    H, W, D = zncc_volume.shape

    if direction == "left-right":
        order = left_right_order((H, W))
    elif direction == "right-left":
        order = right_left_order((H, W))
    elif direction == "up-down":
        order = up_down_order((H, W))
    elif direction == "down-up":
        order = down_up_order((H, W))
    elif direction == "top-right-bottom-left":
        order = top_right_bottom_left_order((H, W))
    elif direction == "bottom-left-top-right":
        order = bottom_left_top_right_order((H, W))
    elif direction == "bottom-right-top-left":
        order = bottom_right_top_left_order((H, W))
    elif direction == "top-left-bottom-right":
        order = top_left_bottom_right_order((H, W))
    else:
        raise ValueError(f"Unknown direction: {direction}")

    # Convert order to array of shape (N, 2) for numba compatibility
    order_arr = np.array(order, dtype=np.int32)
    dy, dx = DIRECTION_OFFSET[direction]

    return aggregate_costs_numba(zncc_volume.astype(np.float32), order_arr, dy, dx, P1, P2)




def run_sgm_varying_P2(left, right, P2_vals, directions, window_size=5, max_disp=64):
    disparity_maps = []
    for P2 in P2_vals:
        zncc = compute_zncc(left, right, window_size=window_size, max_disparity=max_disp)
        agg = np.zeros_like(zncc)
        for d in directions:
            agg += compute_sgm_fast(zncc.copy(), d, P1=1, P2=P2)  # fixed P1
        cost = zncc + agg
        disparity_map = np.argmax(cost, axis=2)
        disparity_maps.append(disparity_map)
        print(f"P2={P2}: disparity min={disparity_map.min()}, max={disparity_map.max()}")
    return disparity_maps

def run_sgm_varying_P1(left, right, P1_vals, directions, window_size=5, max_disp=64):
    disparity_maps = []
    for P1 in P1_vals:
        zncc = compute_zncc(left, right, window_size=window_size, max_disparity=max_disp)
        agg = np.zeros_like(zncc)
        for d in directions:
            agg += compute_sgm_fast(zncc.copy(), d, P1=P1, P2=3)  # fixed P2
        cost = zncc + agg
        disparity_map = np.argmax(cost, axis=2)
        disparity_maps.append(disparity_map)
        print(f"P1={P1}: disparity min={disparity_map.min()}, max={disparity_map.max()}")
    return disparity_maps






def show_image_by_window(values, disparity_maps, idx="0", param_name="P2", title="Effect of different values"):
    """
    Display a row of disparity maps with a shared title and labeled subplots.

    Parameters:
    - values: list of P1 or P2 values (used in subplot titles)
    - disparity_maps: list of corresponding disparity maps
    - idx: string index to use in saved filename
    - param_name: either 'P1' or 'P2' (used in titles)
    - title: main figure title
    """
    fig, axes = plt.subplots(1, len(values), figsize=(4 * len(values), 4))

    if len(values) == 1:
        axes = [axes]  # make iterable

    for ax, val, disp in zip(axes, values, disparity_maps):
        ax.imshow(disp, cmap='gray')
        ax.set_title(f"{param_name}: {val}")
        ax.axis('off')

    fig.suptitle(title, fontsize=18)
    plt.tight_layout(rect=[0, 0.03, 1, 0.92])

    os.makedirs("result", exist_ok=True)
    filename = f"./result/idx{idx}_sgm_{param_name.lower()}_comparison.png"
    plt.savefig(filename)
    print(f"Saved figure to: {filename}")
    plt.show()






# end = time.time()
# print(f"ZNCC computation time: {end - start:.4f} seconds")

idx = "6"
left = cv2.imread(f'./source/{idx}_left.png', cv2.IMREAD_GRAYSCALE)
right = cv2.imread(f'./source/{idx}_right.png', cv2.IMREAD_GRAYSCALE)
#show_images_side_by_side(left, right)

# shared parameters
window_size = 5
max_disp = 64
directions = [
    "left-right", "right-left", "up-down", "down-up",
    "top-right-bottom-left", "bottom-left-top-right",
    "bottom-right-top-left", "top-left-bottom-right"
]

# ----------- P2 effect -----------
P2_vals = [3, 5, 7, 9]
disparity_maps_P2 = run_sgm_varying_P2(left, right, P2_vals, directions, window_size, max_disp)
show_image_by_window(P2_vals, disparity_maps_P2, idx="6", param_name="P2", title="Effect of different P2 values")

# ----------- P1 effect -----------
P1_vals = [1, 3, 5, 7]
disparity_maps_P1 = run_sgm_varying_P1(left, right, P1_vals, directions, window_size, max_disp)
show_image_by_window(P1_vals, disparity_maps_P1, idx="6", param_name="P1", title="Effect of different P1 values")
