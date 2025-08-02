
print("\n<< Stereo_DIC_PSO_ICGN >>")

import numpy as np
import cv2 as cv
import os
import time
import Config as CF
import Config_user as CF_user
import PSO_ICGN_1B2B
import PSO_ICGN_1B1A
import PSO_ICGN_2B2A
import CubicCoef_1B2B_all
import CubicCoef_1B1A
import CubicCoef_2B2A
import Hessian
import Image_Calibration as Img_cal    
import Points2Plane
from function.image_processing import rotate_image
from function.click_tool import click_recorder

print(f"pwd: {os.getcwd()}")
print(f"WORKSPACE: {CF.WORKSPACE}")
print(r"C:\Users\User\Documents\Code\3D-DIC_measurement_system\Stereo Vision\stereoMap.xml")
# folder address
#CF_user.TEST_IMG_DIR = 'Target20230901-1'

# in-plane:0, out-of-plane:1
if CF_user.TEST_MODE_EN == 0:
    force_direction = str('in')
else:
    force_direction = str('out')

# reference image path
file_name = f"{CF_user.LOAD_MIN}_{CF_user.LOAD_MAX}kg_image1.jpg"
if CF_user.TEST_MODE_EN == 0:
    img_1B_address = os.path.join(CF.IMAGE_TARGET_IN_CAM1_DIR, file_name)
    img_2B_address = os.path.join(CF.IMAGE_TARGET_IN_CAM2_DIR, file_name)
elif CF_user.TEST_MODE_EN == 1:
    img_1B_address = os.path.join(CF.IMAGE_TARGET_OUT_CAM1_DIR, file_name)
    img_2B_address = os.path.join(CF.IMAGE_TARGET_OUT_CAM2_DIR, file_name)
else:
    print(f"[ERROR] TEST_MODE_EN={CF_user.TEST_MODE_EN} (Invalid!)")
    
print(f"img_1B_address: {img_1B_address}")
print(f"img_2B_address: {img_2B_address}")
img_1B = cv.imread(str(img_1B_address))
img_2B = cv.imread(str(img_2B_address))

if CF_user.TEST_ROTATE_IMG_EN == 1:
    img_1B = rotate_image(img_1B, -90)
    img_2B = rotate_image(img_2B, 90)

## image rectification
if CF_user.TEST_REC_IMG_EN == 1:
    img_1B_rec, img_2B_rec = Img_cal.undistortRectify(img_1B, img_2B)
else:
    img_1B_rec = img_1B
    img_2B_rec = img_2B

## copy rec images
img_1B_rec_temp = img_1B_rec
img_2B_rec_temp = img_2B_rec

## Select first point in left image
cv.putText(img_1B_rec_temp, 'set a reference point on img_1B', (20, 60),\
            cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
cv.namedWindow("img_1B_rec_temp", cv.WINDOW_NORMAL)
cv.namedWindow("img_2B_rec_temp", cv.WINDOW_NORMAL)
cv.imshow("img_1B_rec_temp", img_1B_rec_temp)
cv.imshow("img_2B_rec_temp", img_2B_rec_temp)

coor_1B = click_recorder()
print('Please set a reference point')
cv.setMouseCallback('img_1B_rec_temp', coor_1B.callback, img_1B_rec_temp)
cv.waitKey(0) 
cv.destroyAllWindows() 

## Select corresponding points in right image
cv.putText(img_2B_rec_temp, 'set a corresponding point on img_2B', (20, 60),\
            cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
cv.namedWindow("img_1B_rec_temp", cv.WINDOW_NORMAL)
cv.namedWindow("img_2B_rec_temp", cv.WINDOW_NORMAL)
cv.imshow("img_1B_rec_temp", img_1B_rec_temp)
cv.imshow("img_2B_rec_temp", img_2B_rec_temp)

print('Please choose a reference point')
coor_2B = click_recorder()
cv.setMouseCallback('img_2B_rec_temp', coor_2B.callback, img_2B_rec_temp)
cv.waitKey(0) 
cv.destroyAllWindows() 

## Read the image calibration file and obtain the projection matrix.
cv_file = cv.FileStorage()
map_path =  f"{CF.WORKSPACE}/stereoMap.xml"
cv_file.open(map_path, cv.FileStorage_READ)
Q = cv_file.getNode('Q').mat()

""" =============== parameters ==============="""
## Set the number of analysis points.
side_len = int(np.sqrt(CF_user.TEST_POINT_ARRAY))
side_len_half = int((side_len-1)/2)

# Due to the large distance (disparity) between the two cameras, a translational distance is set to facilitate DIC in quickly finding corresponding points in the 2B image.
translate_1B2B = coor_1B.y - coor_2B.y

# focal (unit:pixel)
focal = Q[2][3] 

# baseline (unit:mm)
baseline = 1/Q[3][2]

# The xy coordinates of the center points of the two cameras.
principal_x = -Q[0][3]
principal_y = -Q[1][3]


print("Pre-processing...")
if CF_user.TEST_GAUSSIANBLUR_EN == 1:
    img_1B_rec = cv.GaussianBlur(img_1B_rec, (3,3), sigmaX=1, sigmaY=1)
    img_2B_rec = cv.GaussianBlur(img_2B_rec, (3,3), sigmaX=1, sigmaY=1)
    print("TEST_GAUSSIANBLUR_EN: 1")

""" 預先計算影像梯度 插值係數 等等資訊 """
""" ============= Compute image gradient Part1 =============="""
# Convert to gray image
img_1B_rec_gray = cv.cvtColor(img_1B_rec, cv.COLOR_BGR2GRAY)
img_2B_rec_gray = cv.cvtColor(img_2B_rec, cv.COLOR_BGR2GRAY)
# precompute the img_bef image gradient by Sobel operator
# camera1 bef_image
Sobel_1B_u = cv.Sobel(img_1B_rec_gray, cv.CV_64F, 0, 1)*0.125 # y方向
Sobel_1B_v = cv.Sobel(img_1B_rec_gray, cv.CV_64F, 1, 0)*0.125 # x方向

""" ========== Interpolation: Bicubic part1 =========="""
# 填充額外數值以供插值係數計算: 2B_full
img_2B_rec_gray_pad = np.pad(img_2B_rec_gray,[(1,2),(1,2)], mode = 'edge')
row_2B, col_2B = img_2B_rec_gray.shape
CubicCoef_1B2B_ALL, Cubic_Xinv =\
    CubicCoef_1B2B_all.CalculateALL(img_2B_rec_gray_pad, row_2B, col_2B)

# storage area
C1B_points = np.zeros((side_len,side_len,2), dtype=int)
C2B_points = np.zeros((side_len,side_len,2), dtype=float)
WC_bef_zone = np.zeros((side_len,side_len,3), dtype=float)
WC_aft_zone = np.zeros((side_len,side_len,3), dtype=float)
H1B1A_inv_all = np.zeros((side_len,side_len,6,6), dtype=float)
H2B2A_inv_all = np.zeros((side_len,side_len,6,6), dtype=float)
J1B1A_all = np.zeros((side_len,side_len,CF_user.TEST_SUBSET_SIZE_1B1A,CF_user.TEST_SUBSET_SIZE_1B1A,6), dtype=float)
J2B2A_all = np.zeros((side_len,side_len,CF_user.TEST_SUBSET_SIZE_2B2A,CF_user.TEST_SUBSET_SIZE_2B2A,6), dtype=float)
img_2B_sub_zone = np.zeros((side_len,side_len,CF_user.TEST_SUBSET_SIZE_2B2A,CF_user.TEST_SUBSET_SIZE_2B2A), dtype=float)
disM = np.zeros((side_len,side_len,3), dtype=float)
disM_out = np.zeros((side_len,side_len), dtype=float)
disM_in_1 = np.zeros((side_len,side_len), dtype=float)
disM_in_2 = np.zeros((side_len,side_len), dtype=float)
stress_in = np.zeros((side_len,side_len), dtype=float)
stress_out = np.zeros((side_len,side_len), dtype=float)

# start point : (C1_B_x, C1_B_y)
C1_B_x = coor_1B.x
C1_B_y = coor_1B.y

## Corrsponding points
for P in range(-side_len_half,side_len_half+1,1):
    for L in range(-side_len_half,side_len_half+1,1):  
        C1_B_x = int(CF_user.TEST_INTERVAL*L + C1_B_x)
        C1_B_y = int(CF_user.TEST_INTERVAL*P + C1_B_y)
        C1B_points[P+side_len_half][L+side_len_half][0] = C1_B_y
        C1B_points[P+side_len_half][L+side_len_half][1] = C1_B_x

        """ ============= Compute image gradient Part1 =============="""
        # Convert to gray image
        img_1B_rec_gray = cv.cvtColor(img_1B_rec, cv.COLOR_BGR2GRAY)
        img_2B_rec_gray = cv.cvtColor(img_2B_rec, cv.COLOR_BGR2GRAY)

        # Image gradient of 1B2B
        Len_1B2B = int(0.5*(CF_user.TEST_SUBSET_SIZE_2B2A-1))
        IGrad_1B2B_u = Sobel_1B_u[C1_B_y-Len_1B2B:C1_B_y+Len_1B2B+1,\
                                  C1_B_x-Len_1B2B:C1_B_x+Len_1B2B+1]
        IGrad_1B2B_v = Sobel_1B_v[C1_B_y-Len_1B2B:C1_B_y+Len_1B2B+1,\
                                  C1_B_x-Len_1B2B:C1_B_x+Len_1B2B+1]
        H_inv_1B2B, J_1B2B = Hessian.Calculate(CF_user.TEST_SUBSET_SIZE_2B2A, IGrad_1B2B_u, IGrad_1B2B_v)
        
        # 從全圖插值表找插值(用於1B2B搜尋)
        Length_1B2B = int(0.5*(CF_user.TEST_SUBSET_SIZE_2B2A-1)+0.5*(CF_user.TEST_SCAN_SIZE_1B2B-1))
        CubicCoef_1B2B = CubicCoef_1B2B_ALL[C1_B_y-Length_1B2B:C1_B_y+Length_1B2B+1,\
                                            C1_B_x-translate_1B2B-Length_1B2B:C1_B_x-translate_1B2B+Length_1B2B+1]
        # 1B2B尋找對應點與2B之影像梯度
        C2_B_x, C2_B_y, Sobel_2B_u, Sobel_2B_v, img_2B_sub =\
        PSO_ICGN_1B2B.Calculate_1B2B(img_1B_rec_gray, img_2B_rec_gray,\
                                     C1_B_x, C1_B_y,\
                                     CF_user.TEST_SUBSET_SIZE_2B2A, CF_user.TEST_SCAN_SIZE_1B2B, H_inv_1B2B,\
                                     J_1B2B, CubicCoef_1B2B, translate_1B2B)
            
        C2B_points[P+side_len_half][L+side_len_half][0] = C2_B_y
        C2B_points[P+side_len_half][L+side_len_half][1] = C2_B_x
        """ 計算初始三維座標 """
        # 計算視差 xl-xr (unit:pixel)
        Disparity_1B2B = (C1_B_x - C2_B_x) 
        Disparity_1B2B_reci = np.divide(1, Disparity_1B2B)
        # 3D coordinate of Reference point (initial)
        X_origin = (C1_B_x-principal_x)*baseline*Disparity_1B2B_reci
        Y_origin = (C1_B_y-principal_y)*baseline*Disparity_1B2B_reci
        Z_origin = focal*baseline*Disparity_1B2B_reci
        WC_bef_zone[P+side_len_half][L+side_len_half][0] = X_origin
        WC_bef_zone[P+side_len_half][L+side_len_half][1] = Y_origin
        WC_bef_zone[P+side_len_half][L+side_len_half][2] = Z_origin
        
        # << 預計算H_inv_2A2B, J_2A2B >>
        # 1B1A
        Len_1B1A = int(0.5*(CF_user.TEST_SUBSET_SIZE_1B1A-1))
        IGrad_1B1A_u = Sobel_1B_u[C1_B_y-Len_1B1A:C1_B_y+Len_1B1A+1,\
                                  C1_B_x-Len_1B1A:C1_B_x+Len_1B1A+1]
        IGrad_1B1A_v = Sobel_1B_v[C1_B_y-Len_1B1A:C1_B_y+Len_1B1A+1,\
                                  C1_B_x-Len_1B1A:C1_B_x+Len_1B1A+1]
        H_inv_1B1A, J_1B1A =\
            Hessian.Calculate(CF_user.TEST_SUBSET_SIZE_1B1A, IGrad_1B1A_u, IGrad_1B1A_v)
        # store H and J
        H1B1A_inv_all[P+side_len_half][L+side_len_half][:][:] = H_inv_1B1A[:][:]
        J1B1A_all[P+side_len_half][L+side_len_half][:][:][:] = J_1B1A[:][:][:]
        
        # 2B2A (注意:影像梯度矩陣尺寸需調整至2B2A尺寸)
        Len_2B2A = int(0.5*(CF_user.TEST_SUBSET_SIZE_2B2A-1))
        IGrad_2B2A_u = Sobel_2B_u
        IGrad_2B2A_v = Sobel_2B_v
        H_inv_2B2A, J_2B2A =\
            Hessian.Calculate(CF_user.TEST_SUBSET_SIZE_2B2A, IGrad_2B2A_u, IGrad_2B2A_v) 
        H_inv_2B2A_test = H_inv_2B2A 
        # store H and J
        H2B2A_inv_all[P+side_len_half][L+side_len_half][:][:] = H_inv_2B2A[:][:]
        J2B2A_all[P+side_len_half][L+side_len_half][:][:][:] = J_2B2A[:][:][:]
        # store img_2B_sub
        img_2B_sub_zone[P+side_len_half][L+side_len_half][:][:] = img_2B_sub

# 指定陣列中心之追蹤點在2B之位置
u2 = C2B_points[side_len_half][side_len_half][0]
v2 = C2B_points[side_len_half][side_len_half][1]

# ==================================================================

"""  決定擬合平面與追蹤點第4個點  """
# 平面法向量
nVector = Points2Plane.normalVector(WC_bef_zone, side_len)
# 正規化
nVector = nVector/np.linalg.norm(nVector)

dis_sum = 0
img_idx = 1
for img_idx in range(1,2,1):

    img_1A_path = './images/'+CF_user.TEST_IMG_DIR+'/'+str(force_direction)+'/camera1/cal_'+\
                      str(CF_user.LOAD_MAX)+'_'+str(CF_user.LOAD_MAX)+'kg_0cm.image'+str(img_idx)+'.jpg'
    img_2A_path = './images/'+CF_user.TEST_IMG_DIR+'/'+str(force_direction)+'/camera2/cal_'+\
                      str(CF_user.LOAD_MAX)+'_'+str(CF_user.LOAD_MAX)+'kg_0cm.image'+str(img_idx)+'.jpg'
    
    img_1A = cv.imread(str(img_1A_path))
    img_2A = cv.imread(str(img_2A_path))
    
    # rotate image
    if CF_user.TEST_ROTATE_IMG_EN == 1:
        img_1A = rotate_image(img_1A, -90)
        img_2A = rotate_image(img_2A, 90)
    
    # image rectification
    if CF_user.TEST_REC_IMG_EN == 1:
        img_1A_rec, img_2A_rec = Img_cal.undistortRectify(img_1A, img_2A)
    else:
        img_1A_rec = img_1A
        img_2A_rec = img_2A
    
    # Gaussian blur
    if CF_user.TEST_GAUSSIANBLUR_EN == 1:
        img_1A_rec = cv.GaussianBlur(img_1A_rec, (3,3), sigmaX=1, sigmaY=1)
        img_2A_rec = cv.GaussianBlur(img_2A_rec, (3,3), sigmaX=1, sigmaY=1)
        print("Applied Gaussian blur!")
    
    # Convert to gray image
    img_1A_rec_gray = cv.cvtColor(img_1A_rec, cv.COLOR_BGR2GRAY)
    img_2A_rec_gray = cv.cvtColor(img_2A_rec, cv.COLOR_BGR2GRAY)
    # 計算1A影像插值係數
    # Length
    Length = int(0.5*(CF_user.TEST_SUBSET_SIZE_1B1A-1)+0.5*(CF_user.TEST_SCAN_SIZE_1B1A-1)+20)
    Length_1B1A = int(0.5*(CF_user.TEST_SUBSET_SIZE_1B1A-1)+0.5*(CF_user.TEST_SCAN_SIZE_1B1A-1))
    Length_2B2A = int(0.5*(CF_user.TEST_SUBSET_SIZE_2B2A-1)+0.5*(CF_user.TEST_SCAN_SIZE_2B2A-1))
    
    start2 = time.time()
    
    # 同張影像陣列目標點追蹤
    for P in range(-side_len_half,side_len_half+1,1):
        for L in range(-side_len_half,side_len_half+1,1):
            C1_B_y = C1B_points[P+side_len_half][L+side_len_half][0] #integer
            C1_B_x = C1B_points[P+side_len_half][L+side_len_half][1] #integer
            C2_B_y = C2B_points[P+side_len_half][L+side_len_half][0] #decimal
            C2_B_x = C2B_points[P+side_len_half][L+side_len_half][1] #decimal
            
            # Time start
            start = time.time()
            
            # Length = Length_1B1A = Length_2B2A
            Length = int(0.5*(CF_user.TEST_SUBSET_SIZE_1B1A-1)+0.5*(CF_user.TEST_SCAN_SIZE_1B1A-1))
            
            # Time start _1B1A
            start_1B1A = time.time()
            
            # 插值係數1B1A
            Gvalue_1B1A = img_1A_rec_gray[int(C1_B_y)-Length-1:int(C1_B_y)+Length+3,\
                                          int(C1_B_x)-Length-1:int(C1_B_x)+Length+3] 
            Cubic_coef_1B1A =\
                CubicCoef_1B1A.Calculate(Cubic_Xinv, Length, Gvalue_1B1A)
            # H, J
            H_inv_1B1A[:][:] = H1B1A_inv_all[P+side_len_half][L+side_len_half][:][:]
            J_1B1A[:][:][:] = J1B1A_all[P+side_len_half][L+side_len_half][:][:][:]
    
            # 搜尋對應點
            C1_A_x, C1_A_y, Coef_1B1A =\
                PSO_ICGN_1B1A.Calculate_1B1A(img_1B_rec_gray, img_1A_rec_gray,\
                                            C1_B_x, C1_B_y,\
                                            CF_user.TEST_SUBSET_SIZE_1B1A,\
                                            H_inv_1B1A, J_1B1A, Cubic_coef_1B1A)
            # Time end!
            end_1B1A = time.time()
            time_1B1A = end_1B1A - start_1B1A
            # print('time_1B1A:',time_1B1A)
            
            # Time start _2B2A
            start_2B2A = time.time()
            
            # 提取1B2B計算之img_2B灰階值矩陣
            img_2B_sub = img_2B_sub_zone[P+side_len_half][L+side_len_half]

            # 插值係數2B2A      
            Gvalue_2B2A = img_2A_rec_gray[int(C2_B_y)-Length-1:int(C2_B_y)+Length+3,\
                                          int(C2_B_x)-Length-1:int(C2_B_x)+Length+3] 
            Cubic_coef_2B2A =\
                CubicCoef_2B2A.Calculate(Cubic_Xinv, Length, Gvalue_2B2A)           
            # H, J
            H_inv_2B2A[:][:] = H2B2A_inv_all[P+side_len_half][L+side_len_half][:][:]
            J_2B2A[:][:][:] = J2B2A_all[P+side_len_half][L+side_len_half][:][:][:]         
            # 搜尋對應點
            C2_A_x, C2_A_y, Coef_2B2A =\
                PSO_ICGN_2B2A.Calculate_2B2A(img_2A_rec_gray,\
                                             C2_B_x, C2_B_y,\
                                             CF_user.TEST_SUBSET_SIZE_2B2A,\
                                             H_inv_2B2A, J_2B2A,\
                                             Cubic_coef_2B2A,\
                                             img_2B_sub)
            # Time end!
            end_2B2A = time.time()
            time_2B2A = end_2B2A - start_2B2A
            # print('time_2B2A:',time_2B2A)
            
            """ 計算當前目標點之世界座標  """
            # 計算視差 xl-xr (unit:pixel)
            Disparity_1A2A = (C1_A_x - C2_A_x)
            Disparity_1A2A_reci = np.divide(1, Disparity_1A2A)
            X_after = (C1_A_x-principal_x)*baseline*Disparity_1A2A_reci
            Y_after = (C1_A_y-principal_y)*baseline*Disparity_1A2A_reci
            Z_after = focal*baseline*Disparity_1A2A_reci
            
            # Displacement between reference point and target point
            WC_aft_zone[P+side_len_half][L+side_len_half][0] = X_after
            WC_aft_zone[P+side_len_half][L+side_len_half][1] = Y_after
            WC_aft_zone[P+side_len_half][L+side_len_half][2] = Z_after
            disM[P+side_len_half][L+side_len_half][:] = WC_aft_zone[P+side_len_half][L+side_len_half][:] - WC_bef_zone[P+side_len_half][L+side_len_half][:]
            # out:z, in1:x(水平向右+), in2:y(垂直向下+)
            dis_out = WC_aft_zone[P+side_len_half][L+side_len_half][2]-WC_bef_zone[P+side_len_half][L+side_len_half][2]
            dis_out2 = np.dot(disM[P+side_len_half][L+side_len_half],nVector)
            dis_in_1 = WC_aft_zone[P+side_len_half][L+side_len_half][0]-WC_bef_zone[P+side_len_half][L+side_len_half][0]
            dis_in_2 = WC_aft_zone[P+side_len_half][L+side_len_half][1]-WC_bef_zone[P+side_len_half][L+side_len_half][1]
            dis_in_sum = np.sqrt(dis_in_1**2 + dis_in_2**2)
            
            if CF_user.TEST_MODE_EN == 0: # in plane
                print(np.round(dis_in_sum, 6))
                dis_sum += dis_in_sum
            else: # out of plane
                print(np.round(dis_out2, 6))
                dis_sum += dis_out
            
            img_1A_rec = cv.circle(img_1A_rec, (int(C1_A_x), int(C1_A_y)), 5,\
                                  (0, 255, 255), 1)  
            img_2A_rec = cv.circle(img_2A_rec, (int(C2_A_x), int(C2_A_y)), 5,\
                                  (0, 255, 255), 1)  
            # Time end!
            end = time.time()
            total_time = end - start 
            
            disM_out[P+side_len_half][L+side_len_half] = dis_out
            disM_in_1[P+side_len_half][L+side_len_half] = dis_in_1
            disM_in_2[P+side_len_half][L+side_len_half] = dis_in_2

    end2 = time.time()
    total_time2 = end2 - start2
    

cv.imshow('img_1A_rec', img_1A_rec)
cv.imshow('img_2A_rec', img_2A_rec)
cv.waitKey(0)
cv.destroyAllWindows()
cv.imwrite('disField/img_1A_rec.jpg', img_1A_rec)
cv.imwrite('disField/img_2A_rec.jpg', img_2A_rec)


print('Average time per point: ', total_time/(CF_user.TEST_POINT_ARRAY*10))
print('Average dis:',dis_sum/(img_idx*side_len*side_len))
print("End")

