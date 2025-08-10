## ===== DIC (digital image correlation) ===== ##
from function.algorithm.interpolation import get_cubic_value
import numpy as np
from ctypes import cdll, c_int, c_double, POINTER
import cv2 as cv
import Config as CF
import Config_user as CF_user
def find_pt_info_1B2B(img_1B, img_2B, C1_B_x, C1_B_y,\
                   TEST_SUBSET_SIZE_1B2B,\
                   TEST_SUBSET_SIZE_2B2A,\
                    TEST_SCAN_SIZE_1B2B, H_inv_1B2B,\
                   J_1B2B, cubic_coef_2B, translate_1B2B, coef_max_range_1B2B_half):
    ## Initial setting ##
    Size = TEST_SUBSET_SIZE_1B2B
    # 子集合之半邊長
    Len = int(0.5*(Size-1)) 
    # half size of coef_max_range_1B2B (get the center of warp function)

    coef_max_range_1B2B = (coef_max_range_1B2B_half*2)+1
    img_bef = np.array(img_1B, dtype=np.int32)
    img_aft = np.array(img_2B, dtype=np.int32)

    # 建立位移暫存區
    Displacement = np.zeros((2,), dtype=np.int32) # 依序為 [u, v]
    # 係數index、CoefValue
    CoefValue = np.zeros((2,), dtype=np.float64)
    # 所選取目標點的位置 
    Object_point = np.array((C1_B_y, C1_B_x - translate_1B2B), dtype=np.int32)

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
    # 載入SO 動態連結檔案: test_2D_DIC_displacement.dll
    m = cdll.LoadLibrary(f'{CF.DLL_DIR}/PSO_ICGN_1B2B.dll')

    # 設定 dll 檔案中 SCAN 函數的參數資料型態:
    m.SCAN.argtypes = [POINTER(c_int), POINTER(c_int), POINTER(c_int),\
                       POINTER(c_double), POINTER(c_int), POINTER(c_int),\
                       POINTER(c_double)]

    # 設定 dll 檔案中 SCAN 函數的傳回值資料型態
    m.SCAN.restype = c_int

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
    Int_u = Displacement[0]
    Int_v = Displacement[1]

    """============================ Precomputation ==========================="""
    # define Incremental displacement vector: delta_P
    u_inc = 0.01
    ux_inc = 0.01
    uy_inc = 0.01
    v_inc = 0.01
    vx_inc = 0.01
    vy_inc = 0.01
    delta_P = np.array([u_inc, ux_inc, uy_inc, v_inc, vx_inc, vy_inc], dtype=float)

    # Incremental warp function coefficient
    warp_inc_coef = np.array([(1+ux_inc, uy_inc, u_inc),\
                              (vx_inc, 1+vy_inc, v_inc),\
                              (0, 0, 1)], dtype=float)
    
    # Reference subset (Simplied: delta_p = 0) !!!!!!!!
    Gvalue_ref_sub = img_bef_sub
    # Mean of Reference subset 
    f_average = np.mean(Gvalue_ref_sub)
    # Delta_f
    delta_f = np.sqrt(np.sum(np.square(Gvalue_ref_sub-f_average)))

    # define displacement vector: P
    u = Int_u # obtain by PSO
    ux = 0
    uy = 0
    v = Int_v # obtain by PSO
    vx = 0
    vy = 0
    # warp function coefficient of deformed subset
    warp_aft_coef = np.array([(1+ux, uy, u),\
                              (vx, 1+vy, v),\
                              (0, 0, 1)], dtype=float)
        
    """========================== Iteration ============================="""
    cnt = 0
    limit = np.sqrt(np.square(delta_P[0]) + np.square(delta_P[1]*Len)+\
                           np.square(delta_P[2]*Len) + np.square(delta_P[3])+\
                           np.square(delta_P[4]*Len) + np.square(delta_P[5]*Len))
    while limit > 0.0001 and cnt < 10:
        # Average gray value of deformed subset points(with interpolation)
        #print(f"cnt: {cnt}")
        Gvalue_g = np.zeros((Size,Size), dtype=float)
        for i in range(0,Size,1):
            for j in range(0,Size,1):
                # translate rigid subset to the position provided by PSO
                position = np.transpose(np.array([i-Len, j-Len, 1], dtype=float))
                position_warped = warp_aft_coef.dot(position) # relative cor to initial guess
                # convert subset cor to coefficient cor
                coef_idx_row = coef_max_range_1B2B_half + int(np.floor(position_warped[0])) # 
                coef_idx_col = coef_max_range_1B2B_half + int(np.floor(position_warped[1])) #
                # A: cubic coefficient in coef_idx_row, coef_idx_col
                # Find the coefficient in lookup table
                if coef_idx_row>(2*coef_max_range_1B2B) or coef_idx_row<0:
                    if CF_user.TEST_SHOW_DBG_EN == 1:
                        print(f"position: {position}")
                        print(f"position_warped: {position_warped}")
                        print(f"warp_aft_coef: {warp_aft_coef}")
                        print(f"coef_idx_col: {coef_idx_col}")
                        print(f"coef_idx_row is out of bound: coef_idx_row:{coef_idx_row}")
                        print(f"cnt={cnt}")
                        print(f"set coef_idx_row to: {coef_idx_row}")
                    coef_idx_row = (2*coef_max_range_1B2B)
                    
                if coef_idx_col>(2*coef_max_range_1B2B) or coef_idx_col<0:
                    if CF_user.TEST_SHOW_DBG_EN == 1:
                        print(f"position: {position}")
                        print(f"position_warped: {position_warped}")
                        print(f"warp_aft_coef: {warp_aft_coef}")
                        print(f"coef_idx_col is out of bound: coef_idx_col:{coef_idx_col}")
                        print(f"coef_idx_row: {coef_idx_row}")
                        print(f"cnt={cnt}")
                        print(f"set coef_idx_col to: {coef_idx_col}")
                    coef_idx_col = (2*coef_max_range_1B2B)

                A = cubic_coef_2B[coef_idx_row][coef_idx_col][:] # 加上Length後表示起點為子集合中心開始算
                A_re = np.reshape(A, (4,4), order='F')
                # Calculate interpolation and store in matrix Gvalue_g[][]
                Gvalue_g[i][j] = get_cubic_value(position_warped[0]-np.floor(position_warped[0]),\
                                                 position_warped[1]-np.floor(position_warped[1]), A_re)
        
        # compute g_average
        g_average = np.mean(Gvalue_g)

        # compute delata_g
        delta_g = np.sqrt(np.sum(np.square(Gvalue_g-g_average)))
        
        Corelation_sum = 0
        dF_dP = (Gvalue_ref_sub - f_average) - (delta_f/delta_g)*(Gvalue_g - g_average)
        for i in range(0,Size,1):
            for j in range(0,Size,1):
                Corelation_sum += np.transpose(J_1B2B[i][j][:])*dF_dP[i][j]

        delta_P = -H_inv_1B2B.dot(Corelation_sum)
        # Update limit (if limit is enough small, then quit)
        limit = np.sqrt(np.square(delta_P[0]) + np.square(delta_P[1]*Len)+\
                        np.square(delta_P[2]*Len) + np.square(delta_P[3])+\
                        np.square(delta_P[4]*Len) + np.square(delta_P[5]*Len))
        
        warp_inc_coef = np.array([(1+delta_P[1], delta_P[2], delta_P[0]),\
                                  (delta_P[4], 1+delta_P[5], delta_P[3]),\
                                  (0, 0, 1)], dtype=float)
        
        warp_inc_coef_inv = np.linalg.inv(warp_inc_coef)
        # 更新warp function
        warp_aft_coef = warp_aft_coef.dot(warp_inc_coef_inv)
        # 計次數
        cnt += 1
        
    # Subpixel displacement
    U = warp_aft_coef[0][2] # 垂直
    V = warp_aft_coef[1][2] # 水平
    C2_B_y = U + C1_B_y
    C2_B_x = V + C1_B_x - translate_1B2B
    
    """ Calculate new gray value and image gradient of C2_B image: for 2B2A """
    # 計算小數座標之灰階值，並以SOBEL計算影像梯度(image gradient)
    Size_TEMP = TEST_SUBSET_SIZE_2B2A + 2 # No value in boundary, so we allocated 2 more elements in x, y direction,...
    Len_TEMP = int(0.5*(Size_TEMP-1)) # ...make sure we have (Size)*(Size) size matrix.
    warp_aft_coef = np.array([(1, 0, U), (0, 1, V), (0, 0, 1)], dtype=float)
    Gvalue_TEMP = np.zeros(((Size_TEMP),(Size_TEMP)), dtype=float)
    for i in range(0,Size_TEMP,1): # Size需要多2個元素，因為稍後需計算新的Sobel影像梯度
        for j in range(0,Size_TEMP,1):
            position = np.transpose(np.array([i-Len_TEMP, j-Len_TEMP, 1], dtype=float))
            position_warped = warp_aft_coef.dot(position)
            ## 查表以尋找目標點之插值係數(位移記得取整數)
            # 檢查 position_warped 是否有效
            if not np.all(np.isfinite(position_warped)):
                print("[ERROR] position_warped is not finite!")
                continue
            coef_idx_row = coef_max_range_1B2B_half + int(np.floor(position_warped[0]))
            coef_idx_col = coef_max_range_1B2B_half + int(np.floor(position_warped[1]))
            # 邊界檢查 
            if coef_idx_row < 0 or coef_idx_col < 0 or coef_idx_row > (2*coef_max_range_1B2B) or coef_idx_col > (2*coef_max_range_1B2B):
                if CF_user.TEST_SHOW_DBG_EN == 1:
                    print(f"[ERROR] in calculate new gray valu for 2B2A: coef_idx_row, coef_idx_col over max_range:({(2*coef_max_range_1B2B)})!")
                    print(f"coef_idx_row={coef_idx_row}")
                    print(f"coef_idx_col={coef_idx_col}")
                    print(f"i={i}")
                    print(f"j={j}")
                continue
            A = cubic_coef_2B[coef_idx_row][coef_idx_col][:]
            A_re = np.reshape(A, (4,4), order='F')
            # 計算插值之灰階值，並且儲存至Gvalue_g
            Gvalue_TEMP[i][j] = get_cubic_value(position_warped[0]-np.floor(position_warped[0]),\
                                                position_warped[1]-np.floor(position_warped[1]), A_re)
            
    # new gray value matrix of C2B (with interpolation): 
    img_2B_sub = Gvalue_TEMP[1:Size_TEMP-1,1:Size_TEMP-1]
    # Image gradient of new C2B image
    IGrad_2B2A_u_temp = cv.Sobel(Gvalue_TEMP, cv.CV_64F, 0, 1)*0.125 # y方向
    IGrad_2B2A_v_temp = cv.Sobel(Gvalue_TEMP, cv.CV_64F, 1, 0)*0.125 # x方向
    Sobel_2B_u = IGrad_2B2A_u_temp[1:Size_TEMP-1,1:Size_TEMP-1]
    Sobel_2B_v = IGrad_2B2A_v_temp[1:Size_TEMP-1,1:Size_TEMP-1]
    return C2_B_x, C2_B_y, Sobel_2B_u, Sobel_2B_v, img_2B_sub





### ===== 1B1A =====
def find_pt_1B1A(img_1B, img_1A,\
                C1_B_x, C1_B_y,\
                TEST_SUBSET_SIZE_1B1A,\
                H_inv_1B1A, J_1B1A,\
                Cubic_coef_1B1A):
    ## Initial setting ##
    # 設定子矩陣大小(邊長) 需要是奇數!!
    Size = TEST_SUBSET_SIZE_1B1A
    Len = int(0.5*(Size-1))

    img_bef = img_1B.astype(int)
    img_aft = img_1A.astype(int)

    # 建立位移暫存區
    Displacement = np.zeros((2,), dtype=int) # 依序為 [u, v]
    # 係數index、CoefValue
    CoefValue = np.zeros((2,), dtype=float) # dtype無double因此使用float 
    # 所選取目標點的位置 
    Object_point = np.array((C1_B_y,C1_B_x), dtype=int)

    # 建構變形前後影像之子矩陣: img_bef_sub
    img_bef_sub = img_bef[C1_B_y-Len:C1_B_y+Len+1,\
                          C1_B_x-Len:C1_B_x+Len+1] 
                            
    # Reference subset (undeformed subset)
    Mean_bef = np.array(np.mean(img_bef_sub), dtype=float)
    img_bef_sub = img_bef_sub.astype(int)     # 將float轉int

    img_aft_sub = np.zeros((Size,Size), dtype=int)

    """ =============== Compute integer displacement ==============="""
    #============================ 使用C計算位移 ============================#
    # 載入 dll 動態連結檔案: test_2D_DIC_displacement.dll
    m = cdll.LoadLibrary(f'{CF.DLL_DIR}/PSO_ICGN_1B1A.dll')

    # 設定 dll 檔案中 SCAN 函數的參數資料型態:
    m.SCAN.argtypes = [POINTER(c_int), POINTER(c_int), POINTER(c_int),\
                       POINTER(c_double), POINTER(c_int), POINTER(c_int),\
                       POINTER(c_double)]

    # 設定 dll 檔案中 SCAN 函數的傳回值資料型態
    m.SCAN.restype = c_int

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
    delta_P = np.array([u_inc, ux_inc, uy_inc, v_inc, vx_inc, vy_inc], dtype=float)

    # warp function coefficient  (with increment)
    warp_inc_coef = np.array([(1+ux_inc, uy_inc, u_inc),\
                              (vx_inc, 1+vy_inc, v_inc),\
                              (0, 0, 1)], dtype=float)
        
    # Reference subset (Simplied: delta_p = 0) !!!!!!!!
    Gvalue_ref_sub = img_bef_sub 
    # Mean of Reference subset
    f_average = np.mean(Gvalue_ref_sub)
    # Delta_f
    delta_f = np.sqrt(np.sum(np.square(Gvalue_ref_sub-f_average)))

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
                              (0, 0, 1)], dtype=float)
        
    """========================== Iteration ============================="""
    # f:referenve subset, g:target subset
    cnt = 0
    limit = np.sqrt(np.square(delta_P[0]) + np.square(delta_P[1]*Len)+\
                    np.square(delta_P[2]*Len) + np.square(delta_P[3])+\
                    np.square(delta_P[4]*Len) + np.square(delta_P[5]*Len))
    while limit > 0.0001 and cnt < 30:
        # Average gray value of deformed subset points(with interpolation)
        Gvalue_g = np.zeros((Size,Size), dtype=float)
        
        #====================== Compute Gvalue_g in C =======================#
        # 載入SO 動態連結檔案:
        m = cdll.LoadLibrary(f'{CF.DLL_DIR}/iteration_1B1A.dll')
        # 設定 dll 檔案中 Bicubic 函數的參數資料型態:
        m.Gvalue_g.argtypes = [POINTER(c_double), POINTER(c_double), POINTER(c_double)]
        # 設定 dll 檔案中 Bicubic 函數的傳回值資料型態
        #m.Bicubic.restype = c_int
        # 取得陣列指標 3個
        Gvalue_g_Ptr = Gvalue_g.ctypes.data_as(POINTER(c_double))
        Cubic_coef_1B1A_Ptr = Cubic_coef_1B1A.ctypes.data_as(POINTER(c_double))
        warp_aft_coef_Ptr = warp_aft_coef.ctypes.data_as(POINTER(c_double))
        # 呼叫 dll 檔案中的 Gvalue_g 函數 
        m.Gvalue_g(Gvalue_g_Ptr, Cubic_coef_1B1A_Ptr, warp_aft_coef_Ptr)
        #=====================================================================#
        
        # compute g_average
        g_average = np.mean(Gvalue_g)

        # compute delata_g
        delta_g = np.sqrt(np.sum(np.square(Gvalue_g-g_average)))
        # construct Correlation_sum matrix
        Correlation_sum = np.zeros((6,), dtype=float)
        # dF_dP
        dF_dP = (Gvalue_ref_sub-f_average) - (delta_f/delta_g)*(Gvalue_g-g_average)
        
        ## ===== Compute Correlation_sum in C ===== ##
        # 載入SO 動態連結檔案:
        m = cdll.LoadLibrary(f'{CF.DLL_DIR}/iteration_1B1A.dll')
        # 設定 dll 檔案中 Correlation_sum 函數的參數資料型態:
        m.CorrSum.argtypes = [POINTER(c_double), POINTER(c_double), POINTER(c_double)]
        # 設定 dll 檔案中 Correlation_sum 函數的傳回值資料型態
        #m.SCAN.restype = c_int
        # 取得陣列指標 3個
        Correlation_sum_Ptr = Correlation_sum.ctypes.data_as(POINTER(c_double))
        dF_dP_Ptr = dF_dP.ctypes.data_as(POINTER(c_double))
        J_1B1A_Ptr = J_1B1A.ctypes.data_as(POINTER(c_double))
        # 呼叫 dll 檔案中的 Correlation_sum 函數 
        m.CorrSum(Correlation_sum_Ptr, dF_dP_Ptr, J_1B1A_Ptr)
        ## ===== ##
        
        # Compute new delta_P
        delta_P = -H_inv_1B1A.dot(Correlation_sum)
        # Compute size of limit (if enough small then break while loop)
        limit = np.sqrt(np.square(delta_P[0]) + np.square(delta_P[1]*Len)+\
                        np.square(delta_P[2]*Len) + np.square(delta_P[3])+\
                        np.square(delta_P[4]*Len) + np.square(delta_P[5]*Len))
        # New incremental warp function 
        warp_inc_coef = np.array([(1+delta_P[1], delta_P[2], delta_P[0]),\
                                  (delta_P[4], 1+delta_P[5], delta_P[3]),\
                                  (0, 0, 1)], dtype=float)
        # Inverse new incremental warp function
        warp_inc_coef_inv = np.linalg.inv(warp_inc_coef)
        # Update warp function
        warp_aft_coef = warp_aft_coef.dot(warp_inc_coef_inv)
        # count
        cnt += 1
     
    U = warp_aft_coef[0][2] # 垂直
    V = warp_aft_coef[1][2] # 水平
    C1_A_y = U + C1_B_y
    C1_A_x = V + C1_B_x
    return C1_A_x, C1_A_y, CoefValue




## ===== 2B2A =====
def find_pt_2B2A(img_2A,\
                C2_B_x, C2_B_y,\
                TEST_SUBSET_SIZE_2B2A,\
                H_inv_2B2A, J_2B2A,\
                Cubic_coef_2B2A,\
                img_bef_sub):
    ## Initial setting ##
    Size = TEST_SUBSET_SIZE_2B2A
    Len = int(0.5*(Size-1))

    img_aft = img_2A.astype(int)

    # 建立位移暫存區
    Displacement = np.zeros((2,), dtype=int) # 依序為 [u, v]
    # 係數index、CoefValue
    CoefValue = np.zeros((2,), dtype=float) # dtype無double因此使用float 
    # 所選取目標點的位置 (integer)
    Object_point = np.array((int(C2_B_y),int(C2_B_x)), dtype=int)
                            
    # Reference subset (undeformed subset)
    Mean_bef = np.array(np.mean(img_bef_sub), dtype=float)

    img_aft_sub = np.zeros((Size,Size), dtype=int)

    """ =============== Compute integer displacement ==============="""
    #============================ 使用C計算位移 ============================#
    # 載入SO 動態連結檔案: test_2D_DIC_displacement.dll
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
    delta_P = np.array([u_inc, ux_inc, uy_inc, v_inc, vx_inc, vy_inc], dtype=float)

    # warp function coefficient  (with increment)
    warp_inc_coef = np.array([(1+ux_inc, uy_inc, u_inc),\
                              (vx_inc, 1+vy_inc, v_inc),\
                              (0, 0, 1)], dtype=float)
        
    # Reference subset (Simplied: delta_p = 0) !!!!!!!!
    Gvalue_ref_sub = img_bef_sub 
    # Mean of Reference subset
    f_average = np.mean(Gvalue_ref_sub)
    # Delta_f
    delta_f = np.sqrt(np.sum(np.square(Gvalue_ref_sub-f_average)))

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
                              (0, 0, 1)], dtype=float)
        
    """========================== Iteration ============================="""
    # f:referenve subset, g:target subset
    cnt = 0
    limit = np.sqrt(np.square(delta_P[0]) + np.square(delta_P[1]*Len)+\
                    np.square(delta_P[2]*Len) + np.square(delta_P[3])+\
                    np.square(delta_P[4]*Len) + np.square(delta_P[5]*Len))
    while limit > 0.0001 and cnt < 30:
        # Average gray value of deformed subset points(with interpolation)
        Gvalue_g = np.zeros((Size,Size), dtype=float)
        
        #====================== Compute Gvalue_g in C =======================#
        # 載入 dll 動態連結檔案:
        m = cdll.LoadLibrary(f'{CF.DLL_DIR}/iteration_2B2A.dll')
        # 設定 dll 檔案中 Bicubic 函數的參數資料型態:
        m.Gvalue_g.argtypes = [POINTER(c_double), POINTER(c_double), POINTER(c_double)]
        # 設定 dll 檔案中 Bicubic 函數的傳回值資料型態
        #m.Bicubic.restype = c_int
        # 取得陣列指標 3個
        Gvalue_g_Ptr = Gvalue_g.ctypes.data_as(POINTER(c_double))
        Cubic_coef_2B2A_Ptr = Cubic_coef_2B2A.ctypes.data_as(POINTER(c_double))
        warp_aft_coef_Ptr = warp_aft_coef.ctypes.data_as(POINTER(c_double))
        # 呼叫 dll 檔案中的 Gvalue_g 函數 
        m.Gvalue_g(Gvalue_g_Ptr, Cubic_coef_2B2A_Ptr, warp_aft_coef_Ptr)
        #=====================================================================#
        
        # compute g_average
        g_average = np.mean(Gvalue_g)

        # compute delata_g
        delta_g = np.sqrt(np.sum(np.square(Gvalue_g-g_average)))
        # construct Correlation_sum matrix
        Correlation_sum = np.zeros((6,), dtype=float)
        # dF_dP
        dF_dP = (Gvalue_ref_sub-f_average) - (delta_f/delta_g)*(Gvalue_g-g_average)
        
        #==================== Compute Correlation_sum in C ======================#
        # 載入SO 動態連結檔案:
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
                                  (0, 0, 1)], dtype=float)
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