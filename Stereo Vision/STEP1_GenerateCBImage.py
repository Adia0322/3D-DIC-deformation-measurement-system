
### ===== Generate_Chessboard_Image ===== ###
print("Start STEP1_GenerateCBImage.py !")
import cv2 as cv
import glob
import Config as CF
import Config_user as CF_user
from function.image_process import rotate_image
from function.image_process import delete_old_image

## Delete old images
left_jpg_files = glob.glob(f"{CF.IMAGE_STEREO_LEFT_DIR}*.jpg")
right_jpg_files = glob.glob(f"{CF.IMAGE_STEREO_RIGHT_DIR}*.jpg")

delete_old_image(left_jpg_files)
delete_old_image(right_jpg_files)

print('Opening Cameras...')
cap1 = cv.VideoCapture(CF_user.CAM1_ID, cv.CAP_DSHOW)
cap2 = cv.VideoCapture(CF_user.CAM2_ID, cv.CAP_DSHOW)

# close auto setting
cap1.set(cv.CAP_PROP_BUFFERSIZE,0)
cap1.set(cv.CAP_PROP_AUTOFOCUS,0)
cap1.set(cv.CAP_PROP_AUTO_WB,0)
cap2.set(cv.CAP_PROP_BUFFERSIZE,0)
cap2.set(cv.CAP_PROP_AUTOFOCUS,0)
cap2.set(cv.CAP_PROP_AUTO_WB,0)
# set camera focal manully
if CF_user.CAM_MANUAL_FOCAL_EN == 1:
    cap1.set(28,CF_user.CAM1_FOCAL)
    cap2.set(28,CF_user.CAM2_FOCAL)

cv.namedWindow("image1", cv.WINDOW_NORMAL)
cv.namedWindow("image2", cv.WINDOW_NORMAL)
img_cnt = 0
while (cap1.isOpened() and cap1.isOpened()):
    check_if_success1, img1 = cap1.read() 
    check_if_success2, img2 = cap2.read()
    
    # wait 5 ms, return ASCII code
    k = cv.waitKey(5)

    img1 = rotate_image(img1,0)
    img2 = rotate_image(img2,0)
    
    # esc: break
    if k == 27: 
        break
    
    elif k == ord('s'):  # s: save image
        cv.imwrite('./images/StereoLeft/camera1.image' + str(img_cnt) + '.jpg', img1)
        cv.imwrite('./images/StereoRight/camera2.image' + str(img_cnt) + '.jpg', img2)
        print("images save!")
        img_cnt += 1
    
    cv.imshow('img1', img1) 
    cv.imshow('img2', img2)

cap1.release()
cap2.release()
cv.destroyAllWindows()
print("Finished STEP1_GenerateCBImage.py !")