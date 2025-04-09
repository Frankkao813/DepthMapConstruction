import matplotlib.pyplot as plt
import numpy as np

def show_images_side_by_side(left, right):
    # Show images side by side
    plt.figure(figsize=(10, 4))  # Optional: make the figure wider

    plt.subplot(1, 2, 1)
    plt.imshow(left, cmap='gray')
    plt.title('Left Image')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(right, cmap='gray')
    plt.title('Right Image')
    plt.axis('off')

    plt.tight_layout()
    plt.savefig("./result/left_right_image.png")
    print("Left and Right images are saved in the result folder")