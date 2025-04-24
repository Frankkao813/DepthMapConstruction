import numpy as np
import cv2  # Optional, only for loading and resizing images
import time
import matplotlib.pyplot as plt
from utils import show_images_side_by_side
import numpy as np
import cv2
from numba import njit

# Place this at the top-level of your script so it's not redefined
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

def compute_zncc_fast(left_img, right_img, window_size, max_disparity):
    """
    Fast ZNCC computation using convolution (cv2.filter2D)
    """
    assert left_img.shape == right_img.shape
    H, W = left_img.shape
    window_area = window_size ** 2
    pad = window_size // 2

    # Convert to float32 for precision and padding 0 around the image
    left = np.pad(left_img.astype(np.float32), pad, mode='constant')
    right = np.pad(right_img.astype(np.float32), pad, mode='constant')

    sum = np.ones((window_size, window_size), dtype=np.float32)
    mean = np.ones((window_size, window_size), dtype=np.float32) / window_area
    zncc_volume = np.zeros((H, W, max_disparity), dtype=np.float32)
    
    # Precompute image statistics
    # local mean for each pixel in the left image
    mean_L = cv2.filter2D(left, -1, mean)
    std_L = cv2.filter2D((left - mean_L)**2, -1, sum)[pad:-pad, pad:-pad]
    err_L = cv2.filter2D((left - mean_L), -1, sum)[pad:-pad, pad:-pad]
    mean_R = cv2.filter2D(right, -1, mean)
    # Observe a small problem, but revsing it may hurt the performance. Or proabably it should look like this
    std_R = cv2.filter2D((right - mean_R)**2, -1, sum)[pad:-pad, pad:-pad]
    err_R = cv2.filter2D((right - mean_R), -1, sum)[pad:-pad, pad:-pad]

    # # define a shift function to shift the image to the left by 1 pixel
    # # (0, 0): no padding on the rows (vertical/top and bottom); (0, 1): padding 1 column on the right (left/right))
    shift = lambda arr: np.pad(arr, ((0, 0), (0, 1)), mode='constant')[:, 1:]

    # I think it is an approcximation, but it yields the result good enough
    for d in range(max_disparity):
        std_R = shift(std_R)
        err_R = shift(err_R)

        denom = np.sqrt(std_L * std_R)
        epsilon = 1e-5  # small value to prevent division by zero
        denom = np.where(denom < epsilon, epsilon, denom)
        zncc = (err_L * err_R) / denom
    
        # zncc = (err_L * err_R) / np.sqrt(std_L * std_R)

        zncc_volume[:, :, d] = zncc



    print(f'meanl shape: {mean_L.shape}')
    #print(f'zncc shape: {zncc.shape}')
    return zncc_volume



import numpy as np
import cv2
import matplotlib.pyplot as plt


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

# deal with the bottom left to top right order

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
    print("bottom-right-top-left order", order)
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
    print("top-left-bottom-right order", order)
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





def show_image_by_window(window_sizes, disparity_maps):
    fig, axes = plt.subplots(1, len(window_sizes), figsize=(15, 5))
    # Ensure axes is always iterable
    if len(window_sizes) == 1:
        axes = [axes]  # wrap in a list

    for i, (w, disp_map) in enumerate(zip(window_sizes, disparity_maps)):
        # disp_vis = disp_map.astype(np.float32)
        # disp_vis = 255 * (disp_vis - disp_vis.min()) / (disp_vis.max() - disp_vis.min() + 1e-5)
        # disp_vis = disp_vis.astype(np.uint8)

        # # Apply color map
        # disp_color = cv2.applyColorMap(disp_vis, cv2.COLORMAP_JET)
        # disp_color = cv2.cvtColor(disp_color, cv2.COLOR_BGR2RGB)  # Convert BGR → RGB for matplotlib

        # axes[i].imshow(disp_color)
        # axes[i].set_title(f"Window Size: {w}")
        # axes[i].axis('off')
        # for i, (w, disp_map) in enumerate(zip(window_sizes, disparity_maps)):
        axes[i].imshow(disp_map, cmap='gray')
        axes[i].set_title(f"Window Size: {w}")
        axes[i].axis('off')
    plt.tight_layout()
    plt.savefig(f"./result/idx{idx}_zncc_faster_window_sgbm.png")

    plt.show()


P1= 1
P2= 3
direction = ["left-right", "right-left", "up-down", "down-up", "top-right-bottom-left", "bottom-left-top-right", "bottom-right-top-left", "top-left-bottom-right"]



idx = "2"
left = cv2.imread(f'./source/{idx}_left.png', cv2.IMREAD_GRAYSCALE)
right = cv2.imread(f'./source/{idx}_right.png', cv2.IMREAD_GRAYSCALE)
show_images_side_by_side(left, right)

# show the effect of different window sizes
window_sizes = [3, 5, 9, 15]
#window_sizes = [5]

start = time.time()
disparity_maps = []
for w in window_sizes:
    zncc = compute_zncc_fast(left, right, window_size=w, max_disparity=64)
    print(f"ZNCC volume shape: {zncc.shape}")
    print(f"ZNCC min: {zncc.min()}, max: {zncc.max()}, mean: {zncc.mean()}")
    # add the updated value to zncc
    for d in direction:
        zncc += compute_sgm_fast(zncc, d, P1=1, P2=3)
    disparity_map = np.argmax(zncc, axis=2)
    print(f"Window size {w}: disparity min={disparity_map.min()}, max={disparity_map.max()}")
    disparity_maps.append(disparity_map)
# create subplots

show_image_by_window(window_sizes, disparity_maps)





# print(f"Time taken: {end - start:.2f} seconds")

# # show the effect of different window sizes
# img_idx = "2"
# window_sizes = [3, 5, 9, 15]
# disparity_maps = []
# time_list = []

# for w in window_sizes:
#     start_time = time.time()
#     zncc = compute_zncc_fast(left, right, window_size=w, max_disparity=64)
#     print(f"ZNCC volume shape: {zncc.shape}")
#     print(f"ZNCC min: {zncc.min()}, max: {zncc.max()}, mean: {zncc.mean()}")
#     print("NaN count in ZNCC:", np.isnan(zncc).sum())

#     disparity_map = np.argmax(zncc, axis=2)
#     disparity_maps.append(disparity_map)
#     end_time = time.time()
#     time_list.append(end_time - start_time)
#     print(f"ZNCC computation time for window size {w}: {end_time - start_time:.4f} seconds")
# # create subplots
# fig, axes = plt.subplots(1, len(window_sizes), figsize=(15, 5))
# for i, (w, disp_map) in enumerate(zip(window_sizes, disparity_maps)):
#     axes[i].imshow(disp_map, cmap='gray')
#     axes[i].set_title(f"Window Size: {w}")
#     axes[i].axis('off')
# plt.tight_layout()
# plt.savefig(f"./result/zncc_faster_window_{idx}_check2.png")
# plt.show()