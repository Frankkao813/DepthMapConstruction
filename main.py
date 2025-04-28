import os
import cv2
import numpy as np
from cost_function import get_depth_map, visualize_depth_with_nodata
import matplotlib.pyplot as plt

def process_stereo_pairs(root_folder, output_folder):
    """
    Traverse all subfolders and process stereo image pairs.

    Args:
        root_folder (str): The root directory containing middle folders.
        output_folder (str): Where to save output images.
        process_function (callable): A function that takes (left_img, right_img) and returns result image.

    Returns:
        None
    """
    # Ensure output directory exists
    os.makedirs(output_folder, exist_ok=True)

    # Traverse all folders
    for middle_folder in os.listdir(root_folder):
        middle_path = os.path.join(root_folder, middle_folder)
        if not os.path.isdir(middle_path):
            continue  # Skip files

        # Now traverse small folders inside each middle folder
        for small_folder in os.listdir(middle_path):
            for mode in ["zncc"]:
                for window_size in [15]:
                    for max_disparity in [80]:

                        # if small_folder != "MotorcycleE":
                        #     continue

                        small_folder_path = os.path.join(middle_path, small_folder)
                        if not os.path.isdir(small_folder_path):
                            continue  # Skip files

                        # Prepare file paths
                        im0_path = os.path.join(small_folder_path, "im0.png")
                        im1_path = os.path.join(small_folder_path, "im1.png")

                        if not (os.path.exists(im0_path) and os.path.exists(im1_path)):
                            print(f"Warning: Missing im0.png or im1.png in {small_folder_path}")
                            continue

                        # Read images
                        right_img = cv2.imread(im0_path, cv2.IMREAD_GRAYSCALE)
                        left_img = cv2.imread(im1_path, cv2.IMREAD_GRAYSCALE)

                        if left_img is None or right_img is None:
                            print(f"Warning: Failed to load images in {small_folder_path}")
                            continue

                        # Process (e.g., compute disparity map)
                        depth_map, filled_depth_map = get_depth_map(left_img, right_img, mode=mode, window_size=window_size, max_disparity=max_disparity)
                        
                        # Save output using small folder name
                        output_path = os.path.join(output_folder, f"{small_folder}_{mode}_w{window_size}_d{max_disparity}.png")
                        # visualize_depth_with_nodata(left_img, depth_map, filled_depth_map, destination=output_path)
                        # print(f"Saved: {output_path}")
    return

def process_stereo_pairs_SGBM(root_folder, output_folder):
    window_size = 5  # Matching block size
    min_disp = 0
    num_disp = 16 * 6  # Must be divisible by 16
    block_size = window_size

    stereo = cv2.StereoSGBM_create(
        minDisparity=min_disp,
        numDisparities=num_disp,
        blockSize=block_size,
        P1=8 * 1 * window_size**2,
        P2=32 * 1 * window_size**2,
        disp12MaxDiff=1,
        uniquenessRatio=10,
        speckleWindowSize=100,
        speckleRange=32,
        preFilterCap=63,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
)

    # Ensure output directory exists
    os.makedirs(output_folder, exist_ok=True)

    # Traverse all folders
    for middle_folder in os.listdir(root_folder):
        middle_path = os.path.join(root_folder, middle_folder)
        if not os.path.isdir(middle_path):
            continue  # Skip files

        # Now traverse small folders inside each middle folder
        for small_folder in os.listdir(middle_path):
            for max_disparity in [80]:

                # if small_folder != "Vintage" and small_folder != "Adirondack":
                #     continue

                small_folder_path = os.path.join(middle_path, small_folder)
                if not os.path.isdir(small_folder_path):
                    continue  # Skip files

                # Prepare file paths
                im0_path = os.path.join(small_folder_path, "im0.png")
                im1_path = os.path.join(small_folder_path, "im1.png")

                if not (os.path.exists(im0_path) and os.path.exists(im1_path)):
                    print(f"Warning: Missing im0.png or im1.png in {small_folder_path}")
                    continue

                # Read images
                right_img = cv2.imread(im0_path, cv2.IMREAD_GRAYSCALE)
                left_img = cv2.imread(im1_path, cv2.IMREAD_GRAYSCALE)

                if left_img is None or right_img is None:
                    print(f"Warning: Failed to load images in {small_folder_path}")
                    continue

                # Process (e.g., compute disparity map)
                depth_map = stereo.compute(right_img, left_img).astype(np.float32) / 16.0
                disp_vis = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX)
                disp_vis = np.uint8(disp_vis)

                # Show the result
                plt.figure(figsize=(10, 4))
                plt.imshow(disp_vis, cmap='magma')
                plt.colorbar(label='Disparity')
                plt.title('Disparity Map (SGBM)')
                plt.axis('off')
                # Save output using small folder name
                output_path = os.path.join(output_folder, f"{small_folder}_sgbm.png")
                plt.savefig(output_path, bbox_inches='tight', dpi=300)
                # plt.show()
                # print(f"Saved: {output_path}")
    return

if __name__ == "__main__":

    # Define root and output folders
    current_dir = os.path.dirname(os.path.abspath(__file__))
    root_folder = os.path.join(current_dir, "dataset")
    output_folder = "test_result"
    
    process_stereo_pairs(root_folder, output_folder)

    # process_stereo_pairs_SGBM(root_folder, output_folder)