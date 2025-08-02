import cv2 as cv
import os

class click_recorder:
    def __init__(self):
        self.x = None
        self.y = None

    def callback(self, event, x, y, flags, img):
        if event == cv.EVENT_LBUTTONDOWN:
            self.x = x
            self.y = y
            print("coordinate(x,y): ", x, y)
            cv.putText(img, f'{x},{y}', (x, y),
                       cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
