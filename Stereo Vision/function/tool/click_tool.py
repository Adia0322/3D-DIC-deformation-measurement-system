import cv2 as cv

class click_recorder:
    def __init__(self):
        self.x = None
        self.y = None

    def callback(self, event, x, y, flags, img):
        if event == cv.EVENT_LBUTTONDOWN:
            self.x = x
            self.y = y
            print("coordinate(x,y): ", x, y)
            # 在影像上畫圓
            cv.circle(img, (x, y), radius=5, color=(0, 0, 255), thickness=-1)
            # 在影像上寫文字
            cv.putText(img, f'{x},{y}', (x+10, y-10), 
                       cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            # ⚠️ 要更新視窗，不然不會看到變化
            cv.imshow('Image', img)
