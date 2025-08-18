## ===== DIC (digital image correlation) ===== ##
from function.interpolation import get_subpixel_cv
import numpy as np
from ctypes import cdll, c_int, c_double, POINTER
import cv2 as cv
import Config as CF
def find_pt_info_1B2B(img_1B,
                      img_2B,
                      C1_B_x, C1_B_y,
                      TEST_SUBSET_SIZE_1B2B,
                      H_inv_1B2B,
                      J_1B2B,
                      trans):

       ## Initial setting ##
       Size = TEST_SUBSET_SIZE_1B2B
       # 子集合之半邊長
       Len = int(0.5*(Size-1)) 
       # half size of coef_max_range_1B2B (get the center of warp function)
       C2_B_x_guess = C1_B_x - trans
       C2_B_y_guess = C1_B_y
       img_bef = np.array(img_1B, dtype=np.int32)
       img_aft = np.array(img_2B, dtype=np.int32)

       # 建立位移暫存區
       Displacement = np.zeros((2,), dtype=np.int32) # 依序為 [y, x]
       # 係數index、CoefValue
       CoefValue = np.zeros((2,), dtype=np.float64)
       # 所選取目標點的位置 
       Object_point = np.array((C2_B_y_guess, C2_B_x_guess), dtype=np.int32)

       # 建構變形前後影像之子矩陣: img_bef_sub
       img_bef_sub = img_bef[C1_B_y-Len:C1_B_y+Len+1,\
                             C1_B_x-Len:C1_B_x+Len+1]  

       # Reference subset (undeformed subset)
       Mean_bef = np.array(np.mean(img_bef_sub), dtype=np.float64)
       img_bef_sub = img_bef_sub.astype(np.int32)# 將float轉int

       # Target subset (deformed subset)
       img_aft_sub = np.zeros((Size, Size), dtype=np.int32)

       """ =============== Compute integer displacement ==============="""
       #============================ 使用C計算位移 ============================#
       # 載入 dll 動態連結檔案: test_2D_DIC_displacement.dll
       m = cdll.LoadLibrary(f'{CF.DLL_DIR}/PSO_ICGN_1B2B.dll')

       # 設定 dll 檔案中 SCAN 函數的參數資料型態:
       m.SCAN.argtypes = [POINTER(c_int), POINTER(c_int), POINTER(c_int),\
                          POINTER(c_double), POINTER(c_int), POINTER(c_int),\
                          POINTER(c_double)]

       # 設定 dll 檔案中 SCAN 函數的傳回值資料型態
       m.SCAN.restype = None

       # 取得陣列指標 7個
       img_aft_Ptr = img_aft.ctypes.data_as(POINTER(c_int))
       img_aft_sub_Ptr = img_aft_sub.ctypes.data_as(POINTER(c_int))
       img_bef_sub_Ptr = img_bef_sub.ctypes.data_as(POINTER(c_int))
       Mean_bef_Ptr = Mean_bef.ctypes.data_as(POINTER(c_double))
       Object_point_Ptr = Object_point.ctypes.data_as(POINTER(c_int))
       Displacement_Ptr = Displacement.ctypes.data_as(POINTER(c_int))
       CoefValue_Ptr = CoefValue.ctypes.data_as(POINTER(c_double))                        

       # 呼叫 dll 檔案中的 SCAN 函數 
       m.SCAN(img_aft_Ptr, img_aft_sub_Ptr, img_bef_sub_Ptr,\
       Mean_bef_Ptr, Object_point_Ptr, Displacement_Ptr, CoefValue_Ptr)

       #=================== 位移計算完成 計算結果在Displacement =========================#
       # Integer displacement for subpixels algorithm
       int_dis_y = Displacement[0] # y
       int_dis_x = Displacement[1] # x
       print(f"(int_dis_x,int_dis_y)=({int_dis_x},{int_dis_y})")

       # Reference subset (Simplied: delta_p = 0) !!!!!!!!
       ref_matrix_f = img_bef_sub
       # Mean of Reference subset 
       f_average = np.mean(ref_matrix_f)
       # Delta_f
       delta_f = np.std(ref_matrix_f, ddof=0)

       # define displacement vector: P
       x = int_dis_x # obtain by PSO
       xx = 0
       xy = 0
       y = int_dis_y # obtain by PSO
       yx = 0
       yy = 0
       # warp function coefficient of deformed subset
       warp_aft_coef = np.array([(1+xx, xy, x),\
                            (yx, 1+yy, y),\
                            (0, 0, 1)], dtype=np.float64)

       """========================== Iteration ============================="""
       cnt = 0
       limit = 0.1
       while limit > 0.0001 and cnt < 30:
       # Average gray value of deformed subset points(with interpolation)
              target_matrix_g = np.zeros((Size,Size), dtype=np.float64)
              for y1 in range(0,Size,1):
                     for x1 in range(0,Size,1):
                            position = np.transpose(np.array([x1-Len, y1-Len, 1], dtype=np.float64))
                            position_warp = warp_aft_coef.dot(position)
                            local_x = position_warp[0]
                            local_y = position_warp[1]
                            tmp_x = C2_B_x_guess + local_x
                            tmp_y = C2_B_y_guess + local_y
                            # print(f"tmp_x: {tmp_x}")
                            # print(f"tmp_y: {tmp_y}")
                            target_matrix_g[y1][x1] = get_subpixel_cv(img_aft, tmp_x, tmp_y)
                            # print(f"target_matrix_g[y1][x1]: {target_matrix_g[y1][x1]}")

              # print(f"limit: {limit}")
              # compute g_average
              g_average = np.mean(target_matrix_g)

              # compute delata_g (standard deviation)
              delta_g = np.std(target_matrix_g, ddof=0)

              corelation_sum = np.zeros(6, dtype=np.float64) 
              eps = 1e-12
              ratio = delta_f / (delta_g + eps) # prevent zero
              residual_F_G = (ref_matrix_f - f_average) - ratio*(target_matrix_g - g_average)
              for y2 in range(0,Size,1):
                     for x2 in range(0,Size,1):
                            corelation_sum += np.transpose(J_1B2B[y2][x2][:])*residual_F_G[y2][x2]

              corelation_sum = corelation_sum.reshape(6,1) 
              delta_P = (-H_inv_1B2B @ corelation_sum).flatten() # flatten turn 2d array to 1d array to get scalar
              # Update limit (if limit is enough small, then quit)
              limit = np.sqrt(np.square(delta_P[0]) + np.square(delta_P[1]*Len)+
                            np.square(delta_P[2]*Len) + np.square(delta_P[3])+
                            np.square(delta_P[4]*Len) + np.square(delta_P[5]*Len))

              warp_inc_coef = np.array([[1+delta_P[1], delta_P[2], delta_P[0]],
                                       [delta_P[4], 1+delta_P[5], delta_P[3]],
                                       [0, 0, 1]], dtype=np.float64)

              warp_inc_coef_inv = np.linalg.inv(warp_inc_coef)
              # update warp function
              warp_aft_coef = warp_aft_coef @ warp_inc_coef_inv
              # count
              cnt += 1
              # print(f"limit={limit}")

       
       # Subpixel displacement
       X = warp_aft_coef[0][2] # 水平
       Y = warp_aft_coef[1][2] # 垂直
       print(f"(X,int_dis_y)=({X},{Y})")
       C2_B_y = Y + C2_B_y_guess
       C2_B_x = X + C2_B_x_guess

       ## ===== 計算(C2_B_x,C2_B_y)周圍的影像梯度等資訊 =====
       img_2B_sub = np.zeros((Size,Size), dtype=np.float64)
       for y3 in range(-Len,Len+1,1):
              for x3 in range(-Len,Len+1,1):
                     x_2B = C2_B_x+x3
                     y_2B = C2_B_y+y3
                     img_2B_sub[y3+Len][x3+Len] = get_subpixel_cv(img_2B, x_2B, y_2B)

       # padding (otherwise sobel will get bad result in boarder)
       pad = Len + 1  # Sobel need more 1 pixel to expand boarder
       img_2B_sub_pad = cv.copyMakeBorder(img_2B_sub, pad, pad, pad, pad, borderType=cv.BORDER_REFLECT)
       sobel_2B_y = cv.Sobel(img_2B_sub_pad, cv.CV_64F, 0, 1)*0.125 # y方向
       sobel_2B_x = cv.Sobel(img_2B_sub_pad, cv.CV_64F, 1, 0)*0.125 # x方向

       sobel_2B_x = sobel_2B_x[pad:-pad, pad:-pad]
       sobel_2B_y = sobel_2B_y[pad:-pad, pad:-pad]

       return C2_B_x, C2_B_y, sobel_2B_x, sobel_2B_y, img_2B_sub
