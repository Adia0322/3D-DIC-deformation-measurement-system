import cv2 as cv
import numpy as np
import os

def rotate_image(image, angle=90, center=None, scale=1.0, is_enable=False):
    if not is_enable: return image
    (h, w) = image.shape[:2]
    if center is None:
        center = (w / 2, h / 2)
    M = cv.getRotationMatrix2D(center, angle, scale)
    rotated = cv.warpAffine(image, M, (w, h))
    return rotated

def delete_old_image(jpg_files):
    for jpg_file in jpg_files:
        try:
            os.remove(jpg_file)
        except OSError as e:
            print(f"Error:{ e.strerror}")

def convert_gray_img(img):
    return

def check_file_path(img_path):
    if not os.path.exists(img_path):
        return 1
    return 0

def run_Gaussian_blur(img, is_enable=False):
    if not is_enable: return img
    print("TEST_GAUSSIANBLUR_EN: 1")
    img_processed = cv.GaussianBlur(img, (3,3), sigmaX=1, sigmaY=1)
    return img_processed

def extract_patch_bicubic(img, x, y, size):
    if size % 2 == 0: 
        print(f"[WARNING] even number size: {size}!")
    if img is None or img.size == 0:
        return None
    h, w = img.shape
    half = size // 2
    x_min, x_max = half, w - 1 - half
    y_min, y_max = half, h - 1 - half
    if x < x_min or x > x_max or y < y_min or y > y_max:
        print("[WARNING] x,y out of limit, will fix to valid value!")
    x_clipped = np.clip(x, x_min, x_max)
    y_clipped = np.clip(y, y_min, y_max)
    xs = np.linspace(x_clipped - half, x_clipped + half, size)
    ys = np.linspace(y_clipped - half, y_clipped + half, size)
    grid_x, grid_y = np.meshgrid(xs, ys)
    map_x = grid_x.astype(np.float32) # [NOTICE] OpenCV remap need float32
    map_y = grid_y.astype(np.float32)

    patch = cv.remap(
        img,
        map_x,
        map_y,
        interpolation=cv.INTER_CUBIC,
        borderMode=cv.BORDER_REFLECT
    )
    return patch
