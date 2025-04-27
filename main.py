import os
import cv2
import numpy as np
from cost_function import get_depth_map, visualize_depth_with_nodata

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
                depth_map, filled_depth_map = get_depth_map(left_img, right_img, max_disparity=max_disparity)
                
                # Save output using small folder name
                output_path = os.path.join(output_folder, f"{small_folder}_zncc_{max_disparity}.png")
                visualize_depth_with_nodata(left_img, depth_map, filled_depth_map, destination=output_path)
                print(f"Saved: {output_path}")
    return

if __name__ == "__main__":

    # Define root and output folders
    current_dir = os.path.dirname(os.path.abspath(__file__))
    root_folder = os.path.join(current_dir, "dataset")
    output_folder = "test_result"
    
    process_stereo_pairs(root_folder, output_folder)