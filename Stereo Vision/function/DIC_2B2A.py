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
    Displacement = np.zeros((2,), dtype=np.int32) # 依序為 [u, v]
    # 係數index、CoefValue
    CoefValue = np.zeros((2,), dtype=np.float64) # dtype無double因此使用float 
    # 所選取目標點的位置 (integer)
    Object_point = np.array((int(C2_B_y),int(C2_B_x)), dtype=np.int32)
                            
    # Reference subset (undeformed subset)
    Mean_bef = np.array(np.mean(img_bef_sub), dtype=np.float64)

    img_aft_sub = np.zeros((Size,Size), dtype=np.int32)

    """ =============== Compute integer displacement ==============="""
    #============================ 使用C計算位移 ============================#
    # 載入 dll 動態連結檔案: test_2D_DIC_displacement.dll
    m = cdll.LoadLibrary(f'{CF.DLL_DIR}/PSO_ICGN_2B2A.dll')

    # 設定 dll 檔案中 SCAN 函數的參數資料型態:
    m.SCAN.argtypes = [POINTER(c_int), POINTER(c_int), POINTER(c_double),\
                       POINTER(c_double), POINTER(c_int), POINTER(c_int),\
                       POINTER(c_double)]

    # 設定 dll 檔案中 SCAN 函數的傳回值資料型態
    m.SCAN.restype = c_int

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
        
    #=================== 位移計算完成 計算結果在Displacement =========================#
    #print("\nPSO_ICGN_1B1A整數位移(單位:pixels):")
    #print("垂直:", Displacement[0])  #垂直以下為正
    #print("水平:", Displacement[1])  #水平以右為正
    
    # Integer displacement for subpixels algorithm
    Int_u = Displacement[0]
    Int_v = Displacement[1]
    CoefValue = CoefValue[1]
    
    """============================ Precomputation ==========================="""
    # define Incremental displacement vector: delta_P
    u_inc = 0.01
    ux_inc = 0.01
    uy_inc = 0.01
    v_inc = 0.01
    vx_inc = 0.01
    vy_inc = 0.01
    delta_P = np.array([u_inc, ux_inc, uy_inc, v_inc, vx_inc, vy_inc], dtype=np.float64)

    # warp function coefficient  (with increment)
    warp_inc_coef = np.array([(1+ux_inc, uy_inc, u_inc),\
                              (vx_inc, 1+vy_inc, v_inc),\
                              (0, 0, 1)], dtype=np.float64)
        
    # Reference subset (Simplied: delta_p = 0) !!!!!!!!
    ref_matrix_f = img_bef_sub 
    # Mean of Reference subset
    f_average = np.mean(ref_matrix_f)
    # Delta_f
    delta_f = np.sqrt(np.sum(np.square(ref_matrix_f-f_average)))

    # define displacement vector: P
    u = Int_u
    ux = 0
    uy = 0
    v = Int_v
    vx = 0
    vy = 0
    # warp function coefficient of deformed subset
    warp_aft_coef = np.array([(1+ux, uy, u),\
                              (vx, 1+vy, v),\
                              (0, 0, 1)], dtype=np.float64)
        
    """========================== Iteration ============================="""
    # f:referenve subset, g:target subset
    cnt = 0
    limit = np.sqrt(np.square(delta_P[0]) + np.square(delta_P[1]*Len)+\
                    np.square(delta_P[2]*Len) + np.square(delta_P[3])+\
                    np.square(delta_P[4]*Len) + np.square(delta_P[5]*Len))
    while limit > 0.0001 and cnt < 30:
        # Average gray value of deformed subset points(with interpolation)
        target_matrix_g = np.zeros((Size,Size), dtype=np.float64)
        
        #====================== Compute target_matrix_g in C =======================#
        # 載入 dll 動態連結檔案:
        m = cdll.LoadLibrary(f'{CF.DLL_DIR}/iteration_2B2A.dll')
        # 設定 dll 檔案中 Bicubic 函數的參數資料型態:
        m.target_matrix_g.argtypes = [POINTER(c_double), POINTER(c_double), POINTER(c_double)]
        # 設定 dll 檔案中 Bicubic 函數的傳回值資料型態
        #m.Bicubic.restype = c_int
        # 取得陣列指標 3個
        Gvalue_g_Ptr = target_matrix_g.ctypes.data_as(POINTER(c_double))
        Cubic_coef_2B2A_Ptr = Cubic_coef_2B2A.ctypes.data_as(POINTER(c_double))
        warp_aft_coef_Ptr = warp_aft_coef.ctypes.data_as(POINTER(c_double))
        # 呼叫 dll 檔案中的 target_matrix_g 函數 
        m.target_matrix_g(Gvalue_g_Ptr, Cubic_coef_2B2A_Ptr, warp_aft_coef_Ptr)
        #=====================================================================#
        
        # compute g_average
        g_average = np.mean(target_matrix_g)

        # compute delata_g
        delta_g = np.sqrt(np.sum(np.square(target_matrix_g-g_average)))
        # construct Correlation_sum matrix
        Correlation_sum = np.zeros((6,), dtype=np.float64)
        # dF_dP
        dF_dP = (ref_matrix_f-f_average) - (delta_f/delta_g)*(target_matrix_g-g_average)
        
        #==================== Compute Correlation_sum in C ======================#
        # 載入 dll 動態連結檔案:
        m = cdll.LoadLibrary(f'{CF.DLL_DIR}/iteration_2B2A.dll')
        # 設定 dll 檔案中 Correlation_sum 函數的參數資料型態:
        m.CorrSum.argtypes = [POINTER(c_double), POINTER(c_double), POINTER(c_double)]
        # 設定 dll 檔案中 Correlation_sum 函數的傳回值資料型態
        #m.SCAN.restype = c_int
        # 取得陣列指標 3個
        Correlation_sum_Ptr = Correlation_sum.ctypes.data_as(POINTER(c_double))
        dF_dP_Ptr = dF_dP.ctypes.data_as(POINTER(c_double))
        J_2B2A_Ptr = J_2B2A.ctypes.data_as(POINTER(c_double))
        # 呼叫 dll 檔案中的 Correlation_sum 函數 
        m.CorrSum(Correlation_sum_Ptr, dF_dP_Ptr, J_2B2A_Ptr)
        #===========================================================#
        
        # Compute new delta_P
        delta_P = -H_inv_2B2A.dot(Correlation_sum)
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
        warp_aft_coef = warp_aft_coef.dot(warp_inc_coef_inv)
        # count
        cnt += 1
        
    U = warp_aft_coef[0][2] # 垂直
    V = warp_aft_coef[1][2] # 水平
    C2_A_y = U + np.floor(C2_B_y)
    C2_A_x = V + np.floor(C2_B_x)
    return C2_A_x, C2_A_y, CoefValue