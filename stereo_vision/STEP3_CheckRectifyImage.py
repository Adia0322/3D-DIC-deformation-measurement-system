"""
Stereo Vision

執行前注意事項:
    1.相機對應編號
    
手動影像存檔位址:
    ./images/Target/cam1/
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

if CF_user.CAM_BUFFERSIZE_EN:
    cap_left.set(cv.CAP_PROP_BUFFERSIZE,0)
    cap_right.set(cv.CAP_PROP_BUFFERSIZE,0)
# white balance
if CF_user.CAM_AUTO_WB_EN:
    cap_left.set(cv.CAP_PROP_AUTO_WB,0)
    cap_right.set(cv.CAP_PROP_AUTO_WB,0)
#  auto focus
if CF_user.CAM_AUTO_FOCAL_EN:
    cap_left.set(cv.CAP_PROP_AUTOFOCUS,0)
    cap_right.set(cv.CAP_PROP_AUTOFOCUS,0)
else:
    cap_left.set(cv.CAP_PROP_FOCUS, CF_user.CAM1_FOCAL)
    cap_right.set(cv.CAP_PROP_FOCUS, CF_user.CAM2_FOCAL)

cv.namedWindow("frame left", cv.WINDOW_NORMAL)
cv.namedWindow("frame right", cv.WINDOW_NORMAL)

# origin text corrfinate in img_C1_new (not important)
row_c1 = -10
col_c1 = -10

cv.setMouseCallback('frame left', click_event)
while(cap_right.isOpened() and cap_left.isOpened()):
    succes_left, frame_left0 = cap_left.read()
    succes_right, frame_right0 = cap_right.read()

    frame_left_ori = rotate_image(frame_left0,0)
    frame_right_ori = rotate_image(frame_right0,0)

    # Undistort and rectify images
    frame_left_rec = cv.remap(frame_left_ori, stereoMapL_x, stereoMapL_y, cv.INTER_LANCZOS4, cv.BORDER_CONSTANT, 0)
    frame_right_rec = cv.remap(frame_right_ori, stereoMapR_x, stereoMapR_y, cv.INTER_LANCZOS4, cv.BORDER_CONSTANT, 0)
    
    cv.putText(frame_left_rec, str(col_c1) + ',' +str(row_c1), (col_c1+10, row_c1-10),\
               cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    frame_left_rec = cv.circle(frame_left_rec, (col_c1, row_c1), 5, (0, 255, 255), 1)
    
    cv.imshow("frame left", frame_left_rec)
    cv.imshow("frame right", frame_right_rec) 

    k = cv.waitKey(5)
    # ESC: break
    if k==27 or img_cnt==11:
        break
    
    # s: save image
    elif k == ord('s'):
        if CF_user.TEST_MODE_EN == 0:
            save_path_1 = f"{CF.IMAGE_TARGET_IN_CAM1_DIR}/{CF_user.LOAD_CUR}_{CF_user.LOAD_MAX}kg_image" + str(img_cnt) + '.jpg'
            save_path_2 = f"{CF.IMAGE_TARGET_IN_CAM2_DIR}/{CF_user.LOAD_CUR}_{CF_user.LOAD_MAX}kg_image" + str(img_cnt) + '.jpg'
            cv.imwrite(save_path_1, frame_left_ori)
            cv.imwrite(save_path_2, frame_right_ori)
            print(f'save: {save_path_1}')
            print(f'save: {save_path_2}\n')
            img_cnt += 1
        elif CF_user.TEST_MODE_EN == 1:
            save_path_1 = f"{CF.IMAGE_TARGET_OUT_CAM1_DIR}/{CF_user.LOAD_CUR}_{CF_user.LOAD_MAX}kg_image" + str(img_cnt) + '.jpg'
            save_path_2 = f"{CF.IMAGE_TARGET_OUT_CAM2_DIR}/{CF_user.LOAD_CUR}_{CF_user.LOAD_MAX}kg_image" + str(img_cnt) + '.jpg'
            cv.imwrite(save_path_1, frame_left_ori)
            cv.imwrite(save_path_2, frame_right_ori)
            print(f'save: {save_path_1}')
            print(f'save: {save_path_2}\n')
            img_cnt += 1
        else:
            print(f"[ERROR] TEST_MODE_EN={CF_user.TEST_MODE_EN} (invalid value)")
        

cap_right.release()
cap_left.release()
cv.destroyAllWindows()
print('### ===== Finished STEP3 ===== ###')