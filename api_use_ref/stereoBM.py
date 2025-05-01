import cv2
import numpy as np
import matplotlib.pyplot as plt


def compute_stereoBM(left_image_path, right_image_path, num_disparities=64, block_size=9):
    """
    Compute the disparity map using OpenCV's StereoBM algorithm.
    
    :param left_image_path: Path to the left image
    :param right_image_path: Path to the right image
    :param num_disparities: Number of disparities (must be divisible by 16)
    :param block_size: Block size for matching (must be odd)
    :return: Normalized disparity map
    """
    # Load images in grayscale
    left_img = cv2.imread(left_image_path, cv2.IMREAD_GRAYSCALE)
    right_img = cv2.imread(right_image_path, cv2.IMREAD_GRAYSCALE)

    if left_img is None or right_img is None:
        raise ValueError("Error loading images. Check file paths.")

    # Display Left and Right Images as Subplots
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(left_img, cmap="gray")
    axes[0].set_title("Left Image")
    axes[0].axis("off")

    axes[1].imshow(right_img, cmap="gray")
    axes[1].set_title("Right Image")
    axes[1].axis("off")

    plt.show()

    # Create StereoBM object
    stereo = cv2.StereoBM_create(numDisparities=num_disparities, blockSize=block_size)

    # Compute disparity map
    disparity = stereo.compute(left_img, right_img)

    # Normalize for better visualization
    disparity_normalized = cv2.normalize(disparity, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    disparity_normalized = np.uint8(disparity_normalized)

    return disparity_normalized
    #return disparity


if __name__ == "__main__":
    left_image_path = "source/2_left.png"   # Update with actual path
    right_image_path = "source/2_right.png" # Update with actual path

    # Compute the depth map using StereoBM
    depth_map = compute_stereoBM(left_image_path, right_image_path)

    # Display the depth map
    cv2.imshow("Depth Map - StereoBM", depth_map)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Optionally save the depth map
    cv2.imwrite("depth_map_stereoBM.jpg", depth_map)
