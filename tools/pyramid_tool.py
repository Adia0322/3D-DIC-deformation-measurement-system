# import config as CF
import cv2
import numpy as np
import matplotlib.pyplot as plt

def build_pyramid(img, levels=3):
    pyramid = [img]
    current = img
    for _ in range(1, levels):
        current = cv2.pyrDown(current) # downsample using Gaussian pyramid
        pyramid.append(current)

    return pyramid


def show_pyramid(pyramid):
    plt.figure(figsize=(10, 4))

    for i, img in enumerate(pyramid):
        plt.subplot(1, len(pyramid), i + 1)
        plt.imshow(img, cmap='gray')
        plt.title(f"Level {i}\n{img.shape}")
        plt.axis("off")

    plt.tight_layout()
    plt.show()


def get_pyramid_level(img, levels, max_level=3):
    pyramid_img_list = build_pyramid(img, levels=max_level)
    return pyramid_img_list[levels]



if __name__ == "__main__":
    # read image (grayscale)
    img = cv2.imread("data/1.jpg", cv2.IMREAD_GRAYSCALE)

    if img is None:
        raise ValueError("Image not found")

    pyramid = build_pyramid(img, levels=4)

    show_pyramid(pyramid)