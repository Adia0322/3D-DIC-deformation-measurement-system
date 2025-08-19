from function.interpolation import get_subpixel_cv
import numpy as np
from ctypes import cdll, c_int, c_double, POINTER
import cv2 as cv
import Config as CF

## ===== 2B2A =====
def find_pt_2B2A(img_2A,
                C2_B_x,
                C2_B_y,
                TEST_SUBSET_SIZE_2B2A,
                H_inv_2B2A,
                J_2B2A,
                Cubic_coef_2B2A,
                img_bef_sub):
    ## Initial setting ##
    Size = TEST_SUBSET_SIZE_2B2A
    Len = int(0.5*(Size-1))

    img_aft = np.array(img_2A, dtype=np.int32)

    # 建立位移暫存區
    Displacement = np.zeros((2,), dtype=np.int32) #  [y,x]
    # index、CoefValue
    CoefValue = np.zeros((2,), dtype=np.float64)
    # 所選取目標點的位置 (integer)
    Object_point = np.array((int(C2_B_y),int(C2_B_x)), dtype=np.int32)
                            
    # Reference subset (undeformed subset)
    Mean_bef = np.array(np.mean(img_bef_sub), dtype=np.float64)

    img_aft_sub = np.zeros((Size,Size), dtype=np.int32)

    """ ===== Compute integer displacement ====="""
    # 載入 dll 動態連結檔案:
    m = cdll.LoadLibrary(f'{CF.DLL_DIR}/PSO_ICGN_2B2A.dll')

    # 設定 dll 檔案中 SCAN 函數的參數資料型態:
    m.SCAN.argtypes = [POINTER(c_int), POINTER(c_int), POINTER(c_double),\
                       POINTER(c_double), POINTER(c_int), POINTER(c_int),\
                       POINTER(c_double)]

    # 設定 dll 檔案中 SCAN 函數的傳回值資料型態
    m.SCAN.restype = None

    # 取得陣列指標 7個
    img_aft_Ptr = img_aft.ctypes.data_as(POINTER(c_int))
    img_aft_sub_Ptr = img_aft_sub.ctypes.data_as(POINTER(c_int))
    img_bef_sub_Ptr = img_bef_sub.ctypes.data_as(POINTER(c_double))
    Mean_bef_Ptr = Mean_bef.ctypes.data_as(POINTER(c_double))
    Object_point_Ptr = Object_point.ctypes.data_as(POINTER(c_int))
    Displacement_Ptr = Displacement.ctypes.data_as(POINTER(c_int))
    CoefValue_Ptr = CoefValue.ctypes.data_as(POINTER(c_double))

    # 呼叫 dll 檔案中的 SCAN 函數 
    m.SCAN(img_aft_Ptr, img_aft_sub_Ptr, img_bef_sub_Ptr,\
           Mean_bef_Ptr, Object_point_Ptr, Displacement_Ptr, CoefValue_Ptr)
    
    # Integer displacement for initial guess of subpixels algorithm
    int_dis_y = Displacement[0] # y
    int_dis_x = Displacement[1] # x
    CoefValue = CoefValue[1]
    
    # Reference subset
    ref_matrix_f = img_bef_sub 
    # Mean of Reference subset
    f_average = np.mean(ref_matrix_f)
    # Delta_f
    delta_f = np.sqrt(np.sum(np.square(ref_matrix_f-f_average)))

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
        
    """========== Iteration =========="""
    # f:referenve subset, g:target subset
    cnt = 0
    limit = 0.1
    while limit > 0.001 and cnt < 20:
        # Average gray value of deformed subset points(with interpolation)
        target_matrix_g = np.zeros((Size,Size), dtype=np.float64)
        
        ## ========== Compute target_matrix_g in C ==========
        # call dll
        m = cdll.LoadLibrary(f'{CF.DLL_DIR}/iteration_2B2A.dll')
        # set input data type
        m.Gvalue_g.argtypes = [POINTER(c_double), POINTER(c_double), POINTER(c_double)]
        # return data type
        m.Bicubic.restype = None
        # get 3 pointers
        Gvalue_g_Ptr = target_matrix_g.ctypes.data_as(POINTER(c_double))
        Cubic_coef_2B2A_Ptr = Cubic_coef_2B2A.ctypes.data_as(POINTER(c_double))
        warp_aft_coef_Ptr = warp_aft_coef.ctypes.data_as(POINTER(c_double))
        # call function: Gvalue_g
        m.Gvalue_g(Gvalue_g_Ptr, Cubic_coef_2B2A_Ptr, warp_aft_coef_Ptr)
        ## ==========
        
        # compute g_average
        g_average = np.mean(target_matrix_g)
        # compute delata_g
        delta_g = np.std(target_matrix_g, ddof=0)
        # construct correlation_sum matrix
        correlation_sum = np.zeros((6,), dtype=np.float64)
        eps = 1e-12
        ratio = delta_f / (delta_g + eps) # prevent zero
        # residual_F_G
        residual_F_G = (ref_matrix_f-f_average) - ratio*(target_matrix_g-g_average)
        
        ## ===== Compute correlation_sum in C =====
        # 載入 dll 動態連結檔案:
        m = cdll.LoadLibrary(f'{CF.DLL_DIR}/iteration_2B2A.dll')
        # 設定 dll 檔案中 correlation_sum 函數的參數資料型態:
        m.CorrSum.argtypes = [POINTER(c_double), POINTER(c_double), POINTER(c_double)]
        # 設定 dll 檔案中 correlation_sum 函數的傳回值資料型態
        m.SCAN.restype = None
        # 取得陣列指標 3個
        Correlation_sum_Ptr = correlation_sum.ctypes.data_as(POINTER(c_double))
        dF_dP_Ptr = residual_F_G.ctypes.data_as(POINTER(c_double))
        J_2B2A_Ptr = J_2B2A.ctypes.data_as(POINTER(c_double))
        # 呼叫 dll 檔案中的 correlation_sum 函數 
        m.CorrSum(Correlation_sum_Ptr, dF_dP_Ptr, J_2B2A_Ptr)
        # =====

        correlation_sum = correlation_sum.reshape(6,1)
        # Compute new delta_P
        delta_P = (-H_inv_2B2A @ correlation_sum).flatten()
        # Compute size of limit (if enough small then break while loop)
        limit = np.sqrt(np.square(delta_P[0]) + np.square(delta_P[1]*Len)+\
                        np.square(delta_P[2]*Len) + np.square(delta_P[3])+\
                        np.square(delta_P[4]*Len) + np.square(delta_P[5]*Len))
        # New incremental warp function 
        warp_inc_coef = np.array([(1+delta_P[1], delta_P[2], delta_P[0]),\
                                  (delta_P[4], 1+delta_P[5], delta_P[3]),\
                                  (0, 0, 1)], dtype=np.float64)
        # Inverse new incremental warp function
        warp_inc_coef_inv = np.linalg.inv(warp_inc_coef)
        # Update warp function
        warp_aft_coef = warp_aft_coef @ warp_inc_coef_inv
        # count
        cnt += 1
        
    X = warp_aft_coef[0][2]
    Y = warp_aft_coef[1][2]
    C2_A_y = Y + np.floor(C2_B_y)
    C2_A_x = X + np.floor(C2_B_x)
    return C2_A_x, C2_A_y, CoefValue