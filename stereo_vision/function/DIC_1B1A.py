
import numpy as np
from ctypes import cdll, c_int, c_double, POINTER
import cv2 as cv
import Config as CF

### ===== 1B1A =====
def find_pt_1B1A(img_1B,
                 img_1A,
                 C1_B_x,
                 C1_B_y,
                 TEST_SUBSET_SIZE_1B1A,
                 H_inv_1B1A,
                 J_1B1A):
    ## Initial setting ##
    Size = TEST_SUBSET_SIZE_1B1A
    # half of subset
    Len = int(0.5*(Size-1))

    img_bef = np.array(img_1B, dtype=np.int32)
    img_aft = np.array(img_1A, dtype=np.int32)

    height, width = img_aft.shape
    # displacement array
    Displacement = np.zeros((2,), dtype=np.int32) # 依序為 [y, x]
    # index & CoefValue of correlation coeffition
    CoefValue = np.zeros((2,), dtype=np.float64)
    # initial point
    Object_point = np.array((C1_B_y,C1_B_x), dtype=np.int32)

    # Reference subset (undeformed subset)
    img_bef_sub = img_bef[C1_B_y-Len:C1_B_y+Len+1,\
                          C1_B_x-Len:C1_B_x+Len+1]
                            
    # Reference subset (undeformed subset)
    Mean_bef = np.array(np.mean(img_bef_sub), dtype=np.float64)
    img_bef_sub = img_bef_sub.astype(np.int32)

    # Target subset (deformed subset)
    img_aft_sub = np.zeros((Size,Size), dtype=np.int32)

    ## ===== measure interger displacment ===== ##
    m = cdll.LoadLibrary(f'{CF.DLL_DIR}/PSO_1B1A.dll')

    m.SCAN.argtypes = [POINTER(c_int), POINTER(c_int), POINTER(c_int),\
                       POINTER(c_double), POINTER(c_int), POINTER(c_int),\
                       POINTER(c_double)]

    # return type
    m.SCAN.restype = None

    # pointers
    img_aft_Ptr = img_aft.ctypes.data_as(POINTER(c_int))
    img_aft_sub_Ptr = img_aft_sub.ctypes.data_as(POINTER(c_int))
    img_bef_sub_Ptr = img_bef_sub.ctypes.data_as(POINTER(c_int))
    Mean_bef_Ptr = Mean_bef.ctypes.data_as(POINTER(c_double))
    Object_point_Ptr = Object_point.ctypes.data_as(POINTER(c_int))
    Displacement_Ptr = Displacement.ctypes.data_as(POINTER(c_int))
    CoefValue_Ptr = CoefValue.ctypes.data_as(POINTER(c_double))

    # call SCAN function
    m.SCAN(img_aft_Ptr,
           img_aft_sub_Ptr,
           img_bef_sub_Ptr,
           Mean_bef_Ptr,
           Object_point_Ptr,
           Displacement_Ptr,
           CoefValue_Ptr)
        
    # result
    int_dis_y = Displacement[0] # y
    int_dis_x = Displacement[1] # x
    CoefValue = CoefValue[1]
    
    ## ========== ##
    # Reference subset
    ref_matrix_f = img_bef_sub
    # Mean of Reference subset
    f_average = np.mean(ref_matrix_f)
    # Delta_f
    delta_f = np.std(ref_matrix_f, ddof=0)

    # define displacement vector: P
    x = int_dis_x # obtain from PSO
    xx = 0
    xy = 0
    y = int_dis_y # obtain from PSO
    yx = 0
    yy = 0
    # warp function coefficient of deformed subset
    warp_aft_coef = np.array([(1+xx, xy, x),\
                              (yx, 1+yy, y),\
                              (0, 0, 1)], dtype=np.float64)
        
    """========== ICGN =========="""
    cnt = 0
    limit = 0.1
    point_ini = np.array((C1_B_x,C1_B_y), dtype=np.float64)
    img_1A = img_1A.astype(np.float64)
    img_1A_flat = img_1A.flatten(order='C') # C:n row major
    while limit > 0.001 and cnt < 20:
        target_matrix_g_flat = np.zeros(Size*Size, dtype=np.float64) # create new 1d array
        # ========== 
        m = cdll.LoadLibrary(f'{CF.DLL_DIR}/ICGN.dll')
        
        m.update_target_img_subset.argtypes = [
        POINTER(c_double), POINTER(c_double), POINTER(c_double), POINTER(c_double),
        c_int, c_int, c_int
        ]

        m.update_target_img_subset.restype = None

        img_1A_flat_ptr = img_1A_flat.ctypes.data_as(POINTER(c_double))
        target_matrix_g_flat_ptr = target_matrix_g_flat.ctypes.data_as(POINTER(c_double))
        point_ini_ptr = point_ini.ctypes.data_as(POINTER(c_double))
        warp_aft_coef_ptr = warp_aft_coef.ctypes.data_as(POINTER(c_double))
        # call dll
        m.update_target_img_subset(img_1A_flat_ptr,
                                    target_matrix_g_flat_ptr,
                                    point_ini_ptr,
                                    warp_aft_coef_ptr,
                                    width,
                                    height,
                                    Size)

        target_matrix_g = target_matrix_g_flat.reshape((Size, Size)) 
        # ========== 
        
        # print(f"limit: {limit}")
        # compute g_average
        g_average = np.mean(target_matrix_g)
        # compute delata_g (standard deviation)
        delta_g = np.std(target_matrix_g, ddof=0)

        corelation_sum = np.zeros(6, dtype=np.float64) 
        eps = 1e-12
        ratio = delta_f / (delta_g + eps) # prevent zero
        residual_F_G = (ref_matrix_f - f_average) - ratio*(target_matrix_g - g_average)
        corelation_sum = np.tensordot(J_1B1A, residual_F_G, axes=([0,1],[0,1]))

        corelation_sum = corelation_sum.reshape(6,1) 
        delta_P = (-H_inv_1B1A @ corelation_sum).flatten() # flatten turn 2d array to 1d array to get scalar
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
        cnt += 1
        # print(f"limit={limit}")
     
    X = warp_aft_coef[0][2]
    Y = warp_aft_coef[1][2]
    C1_A_x = X + C1_B_x
    C1_A_y = Y + C1_B_y
    return C1_A_x, C1_A_y, CoefValue