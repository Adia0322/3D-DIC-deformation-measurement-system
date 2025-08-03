import numpy as np
import Config as CF

def get_cubic_coef_1B2B(img, row, col):
    Cubic_X = np.array([(1, -1, 1, -1, -1, 1, -1, 1, 1, -1, 1, -1, -1, 1, -1, 1),\
                  (1, 0, 0, 0, -1, 0, 0, 0, 1, 0, 0, 0, -1, 0, 0, 0),\
                  (1, 1, 1, 1, -1, -1, -1, -1, 1, 1, 1, 1, -1, -1, -1, -1),\
                  (1, 2, 4, 8, -1, -2, -4, -8, 1, 2, 4, 8, -1, -2, -4, -8),\
                  (1, -1, 1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),\
                  (1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),\
                  (1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),\
                  (1, 2, 4, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),\
                  (1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1),\
                  (1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0),\
                  (1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1),\
                  (1, 2, 4, 8, 1, 2, 4, 8, 1, 2, 4, 8, 1, 2, 4, 8),\
                  (1, -1, 1, -1, 2, -2, 2, -2, 4, -4, 4, -4, 8, -8, 8, -8),\
                  (1, 0, 0, 0, 2, 0, 0, 0, 4, 0, 0, 0, 8, 0, 0, 0),\
                  (1, 1, 1, 1, 2, 2, 2, 2, 4, 4, 4, 4, 8, 8, 8, 8),\
                  (1, 2, 4, 8, 2, 4, 8, 16, 4, 8, 16, 32, 8, 16, 32, 64)], dtype=int)
    Cubic_Xinv = np.linalg.inv(Cubic_X)
    coef = np.zeros((row,col,16), dtype=float)
    for i in range(0,row,1):
        for j in range(0,col,1):
            gray_value = img[i:i+4, j:j+4]
            gray_value = np.reshape(gray_value, (16,1), order='F')
            coef[i, j, ...] = np.transpose(Cubic_Xinv.dot(gray_value))
    return coef, Cubic_Xinv

def get_cubic_coef_1B1A(Cubic_Xinv, Length, img):
    from ctypes import cdll, c_int, c_double, POINTER
    coef_1B1A = np.zeros((2*Length+1, 2*Length+1,16), dtype=float)
    Length = np.array([Length], dtype=int)
    img = img.astype('int')

    # 載入SO 動態連結檔案:
    m = cdll.LoadLibrary(f'{CF.SO_FILE_INTERPLATION_DIR}/CubicCoef_1B1A.so')
    # 設定 SO 檔案中 CubicCoef 函數的參數資料型態:
    m.CubicCoef.argtypes = [POINTER(c_double), POINTER(c_int),\
                            POINTER(c_int), POINTER(c_double)]
    # 設定 SO 檔案中 CubicCoef 函數的傳回值資料型態
    #m.CubicCoef.restype = c_int #似乎可以不設定
    # 取得陣列指標 4個
    Cubic_Xinv_Ptr = Cubic_Xinv.ctypes.data_as(POINTER(c_double))
    Length_Ptr = Length.ctypes.data_as(POINTER(c_int))
    img_Ptr = img.ctypes.data_as(POINTER(c_int))
    coef_1B1A_Ptr = coef_1B1A.ctypes.data_as(POINTER(c_double))
    # 呼叫 SO 檔案中的 CubicCoef 函數 
    m.CubicCoef(Cubic_Xinv_Ptr, Length_Ptr, img_Ptr, coef_1B1A_Ptr)
    #===========================================================#
    return coef_1B1A

def get_cubic_coef_2B2A(Cubic_Xinv, Length, img):
    from ctypes import cdll, c_int, c_double, POINTER
    import numpy as np
    Coef_2B2A = np.zeros((2*Length+1, 2*Length+1,16), dtype=float)
    Length = np.array([Length], dtype=int)
    img = img.astype('int')
    #============================ C ============================#
    # 載入SO 動態連結檔案:
    m = cdll.LoadLibrary(f'{CF.SO_FILE_INTERPLATION_DIR}./CubicCoef_2B2A.so')
    # 設定 SO 檔案中 CubicCoef 函數的參數資料型態:
    m.CubicCoef.argtypes = [POINTER(c_double), POINTER(c_int),\
                            POINTER(c_int), POINTER(c_double)]
    # 設定 SO 檔案中 CubicCoef 函數的傳回值資料型態
    #m.CubicCoef.restype = c_int #似乎可以不設定
    # 取得陣列指標 4個
    Cubic_Xinv_Ptr = Cubic_Xinv.ctypes.data_as(POINTER(c_double))
    Length_Ptr = Length.ctypes.data_as(POINTER(c_int))
    img_Ptr = img.ctypes.data_as(POINTER(c_int))
    Coef_2B2A_Ptr = Coef_2B2A.ctypes.data_as(POINTER(c_double))
    # 呼叫 SO 檔案中的 CubicCoef 函數 
    m.CubicCoef(Cubic_Xinv_Ptr, Length_Ptr, img_Ptr, Coef_2B2A_Ptr)
    #===========================================================#
    return Coef_2B2A

def get_cubic_value(u, v, coefficient):
    U = np.array([1, u, u*u, u*u*u], dtype=float)
    V = np.array([1, v, v*v, v*v*v], dtype=float)
    gray_value = U.dot(coefficient.dot(np.transpose(V))) # U*coefficient*V
    return gray_value