import numpy as np
import cv2
from numpy.lib.stride_tricks import sliding_window_view
from matplotlib.cm import ScalarMappable
import matplotlib.pyplot as plt
# from utils import show_images_side_by_side
# from scipy.signal import find_peaks
import cv2.ximgproc
from numba import njit



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

    # Initialize ZNCC volume to store matching costs
    zncc_volume = np.zeros((H, W, max_disparity), dtype=np.float32)

    # Iterate over all disparity levels
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

        # Compute numerator: sum of element-wise product over each window
        numerator = np.sum(left_zero_mean * right_zero_mean, axis=(-2, -1))

        # Compute denominator: product of L2 norms (square root of summed squares)
        denom_L = np.sum(left_zero_mean ** 2, axis=(-2, -1))
        denom_R = np.sum(right_zero_mean ** 2, axis=(-2, -1))
        denominator = np.sqrt(denom_L * denom_R)

        # Handle division by zero safely
        zncc = np.zeros_like(numerator)
        valid = denominator > 1e-8  # Avoid division when denominator is too small
        zncc[valid] = numerator[valid] / denominator[valid]

        # Store the ZNCC score for disparity d
        zncc_volume[:, :, d] = zncc

    return zncc_volume

def compute_ncc(left_img, right_img, window_size, max_disparity):
    assert left_img.shape == right_img.shape
    H, W = left_img.shape
    # window_area = window_size ** 2
    pad = window_size // 2

    # Convert to float32 and pad
    left = np.pad(left_img.astype(np.float32), pad, mode='constant')
    right = np.pad(right_img.astype(np.float32), pad, mode='constant')

    # Define convolution kernels
    sum_kernel = np.ones((window_size, window_size), dtype=np.float32)

    # Precompute left image statistics
    left_sq = left ** 2
    # sum_L = cv2.filter2D(left, -1, sum_kernel)[pad:-pad, pad:-pad]
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
        # sum_R = cv2.filter2D(right, -1, sum_kernel)[pad:-pad, pad:-pad]
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
    """
    Compute the SSD (Sum of Squared Differences) cost volume between two stereo images.

    Args:
        left_img (np.ndarray): Left grayscale image, shape (H, W).
        right_img (np.ndarray): Right grayscale image, shape (H, W).
        window_size (int): Size of the local window (must be odd, e.g., 3, 5, 7).
        max_disparity (int): Maximum disparity value to consider.

    Returns:
        np.ndarray: SSD cost volume of shape (H, W, max_disparity),
                    where lower values indicate better matches.
    """
    assert left_img.shape == right_img.shape, "Input images must have the same shape"
    H, W = left_img.shape
    pad = window_size // 2

    # Convert images to float32 for numerical precision, and pad them
    # Padding helps handle windows near the image borders
    left = np.pad(left_img.astype(np.float32), pad, mode='constant')
    right = np.pad(right_img.astype(np.float32), pad, mode='constant')

    # Create a convolution kernel for summing over the local window
    kernel = np.ones((window_size, window_size), dtype=np.float32)

    # Initialize SSD volume to store matching costs for each disparity
    ssd_volume = np.zeros((H, W, max_disparity), dtype=np.float32)

    # Define a helper function to shift the right image one pixel to the left
    # This simulates increasing disparity
    shift = lambda arr: np.pad(arr, ((0, 0), (0, 1)), mode='constant')[:, 1:]

    # Loop over each disparity level
    for d in range(max_disparity):
        # Compute the pixel-wise squared difference between left and (shifted) right
        diff = (left - right) ** 2

        # Sum the squared differences over the local window using convolution
        # Resulting shape after cropping is (H, W)
        ssd = cv2.filter2D(diff, -1, kernel)[pad:-pad, pad:-pad]

        # Store the SSD result at the current disparity level
        ssd_volume[:, :, d] = ssd

        # Shift the right image by one pixel left for the next disparity computation
        right = shift(right)

    return ssd_volume

def parabolic_subpixel_refinement(cost_volume):
    """
    Perform parabolic subpixel refinement with thresholding based on peak cost.

    Args:
        cost_volume (np.ndarray): Cost volume, shape (H, W, D).
        cost_threshold (float): Threshold for the maximum cost value.

    Returns:
        np.ndarray: Refined disparity map (H, W), with invalid regions set to 0.
    """
    H, W, D = cost_volume.shape

    # Step 1: Find best disparity (argmax) and corresponding maximum cost
    best_disparity = np.argmax(cost_volume, axis=2)
    peak_cost = np.max(cost_volume, axis=2)

    # Step 2: Pad cost volume along disparity dimension
    padded_cost = np.pad(cost_volume, ((0, 0), (0, 0), (1, 1)), mode='edge')
    d_indices = best_disparity + 1  # Because of padding

    # Step 3: Get left, center, right costs
    c0 = np.take_along_axis(padded_cost, d_indices[..., None] - 1, axis=2)[..., 0]
    c1 = np.take_along_axis(padded_cost, d_indices[..., None], axis=2)[..., 0]
    c2 = np.take_along_axis(padded_cost, d_indices[..., None] + 1, axis=2)[..., 0]

    # Step 4: Calculate parabola offset
    denom = 2 * (c0 - 2 * c1 + c2)

    offset = np.zeros_like(best_disparity, dtype=np.float32)
    valid_parabola = np.abs(denom) > 1e-6
    offset[valid_parabola] = (c0[valid_parabola] - c2[valid_parabola]) / denom[valid_parabola]

    # Step 5: Refined disparity (initially)
    refined_disparity = best_disparity.astype(np.float32) + offset

    return refined_disparity

def filter_disparity_curves(cost_volume, disparity_map, peak_threshold=0.8, dynamic_range_threshold=0.5, min_peak_prominence=0.05):
    """
    Filter disparity curves based on maximum cost value, dynamic range, and true multi-peak suppression.

    Args:
        cost_volume (np.ndarray): Cost volume of shape (H, W, D).
        disparity_map (np.ndarray): Initial disparity map to be filtered.
        peak_threshold (float): Minimum acceptable peak value.
        dynamic_range_threshold (float): Minimum acceptable (max - min) value.
        min_peak_prominence (float): Minimum prominence required for peaks to be considered independent.

    Returns:
        np.ndarray: Filtered disparity map.
    """
    disparity_map = disparity_map.copy()
    H, W, D = cost_volume.shape

    peak_value = np.max(cost_volume, axis=2)
    min_value = np.min(cost_volume, axis=2)

    condition_peak = peak_value > peak_threshold
    condition_dynamic_range = (peak_value - min_value) > dynamic_range_threshold

    # New: multi-peak detection
    # multi_peak_mask = np.ones((H, W), dtype=bool)  # Assume valid first

    # for i in range(H):
    #     for j in range(W):
    #         cost_curve = cost_volume[i, j, :]
    #         peaks, properties = find_peaks(cost_curve, prominence=min_peak_prominence)

    #         # Valid only if there is exactly 1 prominent peak
    #         if len(peaks) != 1:
    #             multi_peak_mask[i, j] = False

    # Final valid mask
    valid_mask = condition_peak & condition_dynamic_range# & multi_peak_mask
    # valid_mask = np.logical_and(condition_peak, condition_dynamic_range)
    # Set invalid pixels to 0
    disparity_map[~valid_mask] = 0

    # Print statistics
    # total_pixels = H * W
    # failed_peak = np.sum(~condition_peak)
    # failed_dynamic_range = np.sum(~condition_dynamic_range)
    # failed_multi_peak = np.sum(~multi_peak_mask)

    # print(f"--- Filtering Statistics ---")
    # # print(f"Total pixels: {total_pixels}")
    # print(f"Failed peak threshold: {failed_peak} ({failed_peak / total_pixels:.2%})")
    # print(f"Failed dynamic range: {failed_dynamic_range} ({failed_dynamic_range / total_pixels:.2%})")
    # # print(f"Failed multi-peak check: {failed_multi_peak} ({failed_multi_peak / total_pixels:.2%})")
    # print(f"Final valid pixels: {np.sum(valid_mask)} ({np.sum(valid_mask) / total_pixels:.2%})")
    # # print("-----------------------------")

    return disparity_map

def visualize_depth_with_nodata(original_image, depth_map, filled_depth_map, colormap=cv2.COLORMAP_MAGMA, destination=None):
    """
    Normalize and visualize a depth map and its filled version, treating a specific value (e.g., 0) as no-data.
    Also display the original image for comparison.

    Args:
        depth_map (np.ndarray): Original depth map with no-data regions.
        filled_depth_map (np.ndarray): Depth map after filling no-data regions.
        no_data_value (int or float): Value indicating no data (default 0).
        colormap (int): OpenCV colormap type (default MAGMA).
        destination (str or None): Path to save the output figure (optional).
        original_image (np.ndarray or None): Optional original image to show alongside.

    Returns:
        None
    """
    depth_map = depth_map.astype(np.float32)
    filled_depth_map = filled_depth_map.astype(np.float32)

    # Create valid mask from the original depth map
    valid_mask = (depth_map != 0)

    # Normalize both depth maps based on valid depth_map only
    normalized_depth = np.zeros_like(depth_map, dtype=np.uint8)
    normalized_filled_depth = np.zeros_like(filled_depth_map, dtype=np.uint8)

    if np.any(valid_mask):
        min_val = np.min(depth_map[valid_mask])
        max_val = np.max(depth_map[valid_mask])
        scale = 255.0 / (max_val - min_val + 1e-6)

        normalized_valid = np.round((depth_map[valid_mask] - min_val) * scale).clip(0, 255)
        normalized_depth[valid_mask] = normalized_valid.astype(np.uint8)

        normalized_filled = np.round((filled_depth_map - min_val) * scale).clip(0, 255)
        normalized_filled_depth = normalized_filled.astype(np.uint8)
    else:
        min_val, max_val = 0, 1

    # Apply colormap
    color_map = cv2.applyColorMap(normalized_depth, colormap)
    color_map[~valid_mask] = [0, 0, 0]  # No-data region set to black

    color_map_filled = cv2.applyColorMap(normalized_filled_depth, colormap)

    # Set up figure
    fig, axes = plt.subplots(1, 3, figsize=(20, 8))
    ax1, ax2, ax3 = axes

    # Plot original image
    if original_image.ndim == 2:
        ax1.imshow(original_image, cmap='gray')
    else:
        ax1.imshow(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
    ax1.set_title('Original Image')
    ax1.axis('off')

    # Plot depth map with no-data
    ax2.imshow(cv2.cvtColor(color_map, cv2.COLOR_BGR2RGB))
    ax2.set_title('Original Depth Map (with No-Data)')
    ax2.axis('off')

    # Plot filled depth map
    im = ax3.imshow(cv2.cvtColor(color_map_filled, cv2.COLOR_BGR2RGB))
    ax3.set_title('Filled Depth Map')
    ax3.axis('off')

    # Add shared colorbar
    sm = ScalarMappable(cmap=plt.get_cmap('magma'), norm=plt.Normalize(vmin=min_val, vmax=max_val))
    sm.set_array([])
    fig.colorbar(sm, ax=[ax2, ax3], fraction=0.046, pad=0.04).set_label('Depth Value')

    # Save or show
    if destination is not None:
        plt.savefig(destination, bbox_inches='tight', dpi=300)
        print(f"Saved visualization to {destination}")
    else:
        plt.show()

    return

def wls_filter_disparity(disparity_left, left_img, lambda_val=8000, sigma_color=1.5):
    """
    Apply WLS filter to smooth disparity map while preserving edges.

    Args:
        disparity_left (np.ndarray): Raw disparity map (float32 or int16).
        left_img (np.ndarray): Left RGB image (or grayscale).
        lambda_val (float): Smoothness strength parameter (higher = smoother).
        sigma_color (float): Edge preservation parameter (higher = weaker edge preserving).

    Returns:
        np.ndarray: WLS-smoothed disparity map.
    """
    # If input disparity is float, convert to int16 for WLS (OpenCV expects it)
    if disparity_left.dtype == np.float32 or disparity_left.dtype == np.float64:
        disparity_left = (disparity_left * 16).astype(np.int16)  # fixed-point format (16x)

    # Create WLS filter
    wls_filter = cv2.ximgproc.createDisparityWLSFilterGeneric(False)

    # Set filter parameters
    wls_filter.setLambda(lambda_val)
    wls_filter.setSigmaColor(sigma_color)

    # Apply WLS filtering
    filtered_disparity = wls_filter.filter(disparity_left, left_img)

    # Convert back to float32
    filtered_disparity = filtered_disparity.astype(np.float32) / 16.0

    return filtered_disparity

def compute_edge_map(img, method='canny'):
    """
    Compute edge map of an image.
    
    Args:
        img (np.ndarray): Input grayscale image.
        method (str): 'canny' or 'sobel' or 'laplacian'
    
    Returns:
        np.ndarray: Edge map.
    """
    if method == 'canny':
        edges = cv2.Canny(img, 50, 150)
    elif method == 'sobel':
        grad_x = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=3)
        edges = cv2.magnitude(grad_x, grad_y)
        edges = np.uint8(np.clip(edges, 0, 255))
    elif method == 'laplacian':
        edges = cv2.Laplacian(img, cv2.CV_32F)
        edges = np.uint8(np.clip(np.abs(edges), 0, 255))
    else:
        raise ValueError("Unknown edge detection method.")
    return edges

def image_preprocessing(img):

    # 1. Apply Gaussian blur
    blurred = cv2.GaussianBlur(img, (5, 5), sigmaX=1.0)

    # 2. Apply Bilateral filter
    bilateral = cv2.bilateralFilter(blurred, d=9, sigmaColor=75, sigmaSpace=75)

    # 3. Apply Histogram Equalization
    enhanced = cv2.equalizeHist(bilateral)
    
    return enhanced

def visualize_zncc_curve_and_matching(left_img, right_img, zncc_volume, x, y):
    """
    Visualize the ZNCC matching curve and the corresponding matching pixel positions 
    in the left and right images for a given pixel (x, y).

    Args:
        left_img (np.ndarray): Left image (grayscale).
        right_img (np.ndarray): Right image (grayscale).
        zncc_volume (np.ndarray): Precomputed ZNCC cost volume, shape (H, W, max_disparity).
        x (int): x-coordinate of the selected pixel in the left image.
        y (int): y-coordinate of the selected pixel in the left image.
    """

    # Extract the ZNCC curve for the selected pixel (y, x)
    curve = zncc_volume[y, x, :]  # Shape: (max_disparity,)

    # Find the disparity with the maximum ZNCC similarity (i.e., the best match)
    best_disparity = np.argmax(curve)

    # Create a figure with three subplots: left image, right image, and ZNCC curve
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # --- Left image ---
    ax_left = axes[0]
    ax_left.imshow(left_img, cmap='gray')
    ax_left.scatter(x, y, s=100, c='red', marker='x')  # Highlight the selected pixel
    ax_left.set_title(f'Left Image\nSelected Pixel (x={x}, y={y})')
    ax_left.axis('off')

    # --- Right image ---
    ax_right = axes[1]
    ax_right.imshow(right_img, cmap='gray')
    disparities = np.arange(len(curve))
    for d in disparities:
        x_shift = x + d  # Note: shifting to simulate matching in right image
        if 0 <= x_shift < right_img.shape[1]:  # Ensure the shifted position is within bounds
            # Scatter each shifted pixel; highlight the best disparity
            ax_right.scatter(x_shift, y, s=30, alpha=0.5, 
                             label=f'd={d}' if d == best_disparity else None)
    
    ax_right.set_title('Right Image\nMatching Positions')
    ax_right.axis('off')

    # --- ZNCC Curve ---
    ax_curve = axes[2]
    ax_curve.plot(np.arange(len(curve)), curve, marker='o')
    ax_curve.axvline(best_disparity, color='red', linestyle='--', label=f'Best d={best_disparity}')
    ax_curve.set_title('ZNCC Curve')
    ax_curve.set_xlabel('Disparity')
    ax_curve.set_ylabel('ZNCC Similarity')
    ax_curve.grid(True)
    ax_curve.set_ylim([-1.1, 1.1])  # ZNCC range is typically [-1, 1]
    ax_curve.legend()

    # Adjust layout to avoid overlapping
    plt.tight_layout()
    plt.show()

    return

def extract_and_visualize_patches(left_img, right_img, zncc_volume, x, y, window_size=5, disparities=None):
    """
    Extract patches from the left and right images around a given pixel (x, y),
    visualize the matching patches across disparities, show ZNCC scores, 
    and highlight the best matching disparity.

    Args:
        left_img (np.ndarray): Left image (grayscale).
        right_img (np.ndarray): Right image (grayscale).
        zncc_volume (np.ndarray): Precomputed ZNCC cost volume, with shape (H, W, max_disparity).
        x (int): x-coordinate of the selected pixel in the left image.
        y (int): y-coordinate of the selected pixel in the left image.
        window_size (int): Size of the extracted patches (must be odd).
        disparities (list or None): List of disparities to visualize (if None, use all disparities).
    """
    half_w = window_size // 2
    H, W = left_img.shape[:2]
    D = zncc_volume.shape[2]

    if disparities is None:
        disparities = np.arange(D)

    # Safety check: make sure the left patch is within image bounds
    if (y-half_w < 0) or (y+half_w >= H) or (x-half_w < 0) or (x+half_w >= W):
        raise ValueError("Left patch extraction out of image bounds!")
    
    # Initialize variables for consistent intensity scaling across all patches
    vmin = 255
    vmax = 0

    # Extract the patch from the left image
    left_patch = left_img[y-half_w:y+half_w+1, x-half_w:x+half_w+1]
    vmin = min(left_patch.min(), vmin)
    vmax = max(left_patch.max(), vmax)

    # Find the best matching disparity based on maximum ZNCC value
    best_disparity = np.argmax(zncc_volume[y, x, :])
    print("Best disparity:", best_disparity)

    # Extract corresponding patches from the right image for each disparity
    right_patches = []
    for d in disparities:
        x_shift = x + d  # Note: matching pixel in the right image is at x + d
        if (x_shift - half_w >= 0) and (x_shift + half_w < W):
            patch = right_img[y-half_w:y+half_w+1, x_shift-half_w:x_shift+half_w+1]
            vmin = min(patch.min(), vmin)
            vmax = max(patch.max(), vmax)
            zncc_score = zncc_volume[y, x, d]
            right_patches.append((d, patch, zncc_score))
        else:
            right_patches.append((d, None, None))  # Mark as out-of-bounds if cannot extract patch

    # Set up the figure for visualization
    n = len(disparities) + 1  # +1 for the left patch
    fig, axes = plt.subplots(1, n, figsize=(3*n, 3))

    # Add a global title indicating pixel location and best matching disparity
    fig.suptitle(f"Patch Matching Visualization at (x={x}, y={y}, best match={best_disparity})", fontsize=16)
    
    # Visualize the left patch
    if left_patch.ndim == 2:
        axes[0].imshow(left_patch, cmap='gray', vmin=vmin, vmax=vmax)
    axes[0].set_title(f'Left Patch\n(x={x}, y={y})')
    axes[0].axis('off')

    # Visualize each extracted right patch
    for i, (d, patch, score) in enumerate(right_patches):
        ax = axes[i+1]
        if patch is not None:
            ax.imshow(patch, cmap='gray', vmin=vmin, vmax=vmax)
            ax.set_title(f'Right Patch\nd={d}\nZNCC={score:.2f}')
        else:
            ax.set_title(f'd={d}\nOut of bounds')
            ax.axis('off')
        ax.axis('off')

    # Adjust layout to avoid overlap between subplots
    plt.tight_layout()
    plt.show()

    return

def get_depth_map(left_img, right_img, mode="ncc", window_size=7, max_disparity=40):
    """
    Compute depth map using ZNCC cost volume and parabolic subpixel refinement.

    Args:
        left_img (np.ndarray): Left grayscale image.
        right_img (np.ndarray): Right grayscale image.
        window_size (int): Size of the local window for ZNCC.
        max_disparity (int): Maximum disparity to consider.

    Returns:
        np.ndarray: Refined disparity map.
    """
    # Compute cost volume
    # left_img = compute_edge_map(left_img, method='canny')
    # right_img = compute_edge_map(right_img, method='canny')
    process_left_image = image_preprocessing(left_img)
    process_right_image = image_preprocessing(right_img)

    peak_threshold = 0.8
    dynamic_range_threshold = 0.5
    if mode == "zncc":
        cost_volume = compute_zncc(process_left_image, process_right_image, window_size, max_disparity)
        peak_threshold = 0.85
        dynamic_range_threshold = 0.5
    elif mode == "ncc":
        cost_volume = compute_ncc(process_left_image, process_right_image, window_size, max_disparity)
        peak_threshold = 0.8
        dynamic_range_threshold = 0.02
    # elif mode == "ssd":
    #     cost_volume = compute_ssd(process_left_image, process_right_image, window_size, max_disparity)
    else:
        raise ValueError("Unknown mode. Choose from 'zncc', 'ncc', or 'ssd'.")

    # Visualize the cost volume for a specific pixel
    # x, y = 455, 150
    # visualize_zncc_curve_and_matching(left_img, right_img, cost_volume, x=x, y=y)
    # extract_and_visualize_patches(left_img, right_img, cost_volume, x=x, y=y, window_size=window_size, disparities=range(40, 80, 2))

    # Parabolic subpixel refinement
    # refined_disparity = np.argmax(cost_volume, axis=2)
    refined_disparity = parabolic_subpixel_refinement(cost_volume)
    refined_disparity = filter_disparity_curves(cost_volume, refined_disparity, peak_threshold=peak_threshold, dynamic_range_threshold=dynamic_range_threshold)

    wls_filter = cv2.ximgproc.createDisparityWLSFilterGeneric(False)
    filled_disparity = wls_filter.filter(refined_disparity, left_img)

    return refined_disparity, filled_disparity
