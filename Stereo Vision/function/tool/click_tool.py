import cv2 as cv

class click_recorder:
    def __init__(self):
        self.x = None
        self.y = None

    def callback_cam1(self, event, x, y, flags, img):
        if event == cv.EVENT_LBUTTONDOWN:
            self.x = x
            self.y = y
            print("coordinate(x,y): ", x, y)
            # 在影像上畫圓
            cv.circle(img, (x, y), radius=10, color=(0, 0, 255), thickness=3)
            # 在影像上寫文字
            cv.putText(img, f'{x},{y}', (x+10, y-10), 
                       cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            cv.imshow('img_1B_rec_temp', img)

    def callback_cam2(self, event, x, y, flags, img):
        if event == cv.EVENT_LBUTTONDOWN:
            self.x = x
            self.y = y
            print("coordinate(x,y): ", x, y)
            # 在影像上畫圓
            cv.circle(img, (x, y), radius=10, color=(0, 0, 255), thickness=3)
            # 在影像上寫文字
            cv.putText(img, f'{x},{y}', (x+10, y-10), 
                       cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            cv.imshow('img_2B_rec_temp', img)
