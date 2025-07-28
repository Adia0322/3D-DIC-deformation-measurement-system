import cv2 as cv
import os

def rotate_image(image, angle, center=None, scale=1.0):
    (h, w) = image.shape[:2]
    # define rotation center,
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

def click_event(event, x, y, flags, params):
    if event == cv.EVENT_LBUTTONDOWN:
  
        global u1, v1
        print("點選的座標:",x, ' ', y)
        u1 = y
        v1 = x      