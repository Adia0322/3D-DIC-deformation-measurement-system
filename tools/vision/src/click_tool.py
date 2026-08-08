import cv2 as cv
import numpy as np
class click_recorder:
    def __init__(self, window_name):
        self.x              = None
        self.y              = None
        self.window_name    = window_name

    def callback(self, event, x, y, flags, img):
        if event == cv.EVENT_LBUTTONDOWN:
            self.x = x 
            self.y = y
            print(f"coordinate(x,y): {x},{y}")
            cv.circle(img, (x, y), radius=5, color=(0, 0, 255), thickness=-1)
            cv.putText(img, f'{x},{y}', (x+10, y-10), 
                       cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            cv.imshow(self.window_name, img)

def get_click_point(img, window_name, text):
    cv.putText(img, text, (20, 60),
               cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv.namedWindow(window_name, cv.WINDOW_NORMAL)
    cv.imshow(window_name, img)
    recorder = click_recorder(window_name)
    print(f'Please {text} by clicking on the image.')
    cv.setMouseCallback(window_name, recorder.callback, img)
    cv.waitKey(0)
    cv.destroyAllWindows()
    return recorder.x, recorder.y

