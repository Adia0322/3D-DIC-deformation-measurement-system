"""
Stereo Vision

執行前注意事項:
    1.相機對應編號
    
手動影像存檔位址:
    ./images/Target/camera1/
"""
import cv2 as cv
import Config as CF
import Config_user as CF_user
from function.image_processing import rotate_image
from function.image_processing import click_event
### ===== 參數設定 ===== ###
# camera index
cam_index_left = 1
cam_index_right = 0
img_cnt = 1   # 相片編號

print("\n Stereo_DIC_PSO_ICGN ")

# Camera parameters to undistort and rectify images
cv_file = cv.FileStorage()
cv_file.open('stereoMap.xml', cv.FileStorage_READ)

stereoMapL_x = cv_file.getNode('stereoMapL_x').mat()
stereoMapL_y = cv_file.getNode('stereoMapL_y').mat()
stereoMapR_x = cv_file.getNode('stereoMapR_x').mat()
stereoMapR_y = cv_file.getNode('stereoMapR_y').mat()

# Open both cameras (注意相機編號!!!)
cap_left =  cv.VideoCapture(cam_index_left, cv.CAP_DSHOW)
cap_right = cv.VideoCapture(cam_index_right, cv.CAP_DSHOW)                    

# close auto setting
# cap_left.set(21,0)
# cap_right.set(21,0)
# cap_left.set(39,0)
# cap_right.set(39,0)
# 手動設定相機焦距(若相機無自動對焦) cv2.CAP_PROP_FOCUS
if CF_user.CAM_MANUAL_FOCAL_EN == 1:
    cap_left.set(cv.CAP_PROP_FOCUS,CF_user.CAM1_FOCAL)
    cap_right.set(cv.CAP_PROP_FOCUS,CF_user.CAM2_FOCAL)

cv.namedWindow("frame left", cv.WINDOW_NORMAL)
cv.namedWindow("frame right", cv.WINDOW_NORMAL)

# origin text corrfinate in img_C1_new (not important)
u1 = -10
v1 = -10

cv.setMouseCallback('frame left', click_event)
while(cap_right.isOpened() and cap_left.isOpened()):
    succes_left, frame_left0 = cap_left.read()
    succes_right, frame_right0 = cap_right.read()

    frame_left_ori = rotate_image(frame_left0,0)
    frame_right_ori = rotate_image(frame_right0,0)

    # Undistort and rectify images
    frame_left_rec = cv.remap(frame_left_ori, stereoMapL_x, stereoMapL_y, cv.INTER_LANCZOS4, cv.BORDER_CONSTANT, 0)
    frame_right_rec = cv.remap(frame_right_ori, stereoMapR_x, stereoMapR_y, cv.INTER_LANCZOS4, cv.BORDER_CONSTANT, 0)
    
    cv.putText(frame_left_rec, str(v1) + ',' +str(u1), (v1+10, u1-10),\
               cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    frame_left_rec = cv.circle(frame_left_rec, (v1, u1), 5, (0, 255, 255), 1)
    
    cv.imshow("frame left", frame_left_rec)
    cv.imshow("frame right", frame_right_rec) 

    k = cv.waitKey(5)
    # ESC: break
    if k==27 or img_cnt==11: 
        break
    
    # s: save image
    elif k == ord('s'):
        if CF_user.TEST_MODE_EN == 0:
            cv.imwrite(f"{CF.IMAGE_TARGET_IN_CAM1_DIR}/{CF_user.TEST_LOAD}kg_image" + str(img_cnt) + '.jpg', frame_left_ori)
            cv.imwrite(f"{CF.IMAGE_TARGET_IN_CAM2_DIR}/{CF_user.TEST_LOAD}kg_image" + str(img_cnt) + '.jpg', frame_right_ori)
            print(f"{CF_user.TEST_LOAD}kg_image" + str(img_cnt) + ".jpg" + " save!")
            img_cnt += 1
        elif CF_user.TEST_MODE_EN == 1:
            cv.imwrite(f"{CF.IMAGE_TARGET_OUT_CAM1_DIR}/{CF_user.TEST_LOAD}kg_image" + str(img_cnt) + '.jpg', frame_left_ori)
            cv.imwrite(f"{CF.IMAGE_TARGET_OUT_CAM2_DIR}/{CF_user.TEST_LOAD}kg_image" + str(img_cnt) + '.jpg', frame_right_ori)
            print(f"{CF_user.TEST_LOAD}kg_image" + str(img_cnt) + ".jpg" + " save!")
            img_cnt += 1
        else:
            print(f"[ERROR] TEST_MODE_EN={CF_user.TEST_MODE_EN}")
        

# Release and destroy all windows before termination
cap_right.release()
cap_left.release()
cv.destroyAllWindows()
print('### ===== Finished STEP3 ===== ###')