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
    cubic_Xinv = np.linalg.inv(Cubic_X)
    coef = np.zeros((row,col,16), dtype=float)
    for i in range(0,row,1):
        for j in range(0,col,1):
            gray_value = img[i:i+4, j:j+4]
            gray_value = np.reshape(gray_value, (16,1), order='F')
            coef[i, j, ...] = np.transpose(cubic_Xinv.dot(gray_value))
    return coef, cubic_Xinv

def get_cubic_coef_1B1A(cubic_Xinv, Length, img):
    import numpy as np
    from ctypes import cdll, c_int, c_double, POINTER

    coef_1B1A = np.zeros((2 * Length + 1, 2 * Length + 1, 16), dtype=np.float64)
    Length = int(Length)
    img = img.astype(np.int32)
    height, width = img.shape

    # 載入 DLL
    m = cdll.LoadLibrary(f'{CF.DLL_DIR}/cubic_coef_1B1A.dll')

    # 宣告 C 函式參數型態
    Matrix16x16 = (c_double * 16) * 16
    m.cubic_coef.argtypes = [
        Matrix16x16,         # cubic_Xinv 16x16 double
        c_int,               # Length
        c_int,               # width
        POINTER(c_int),      # img 一維 int array
        POINTER(c_double)    # Coef 一維 double array
    ]
    m.cubic_coef.restype = None

    # 轉換 numpy → ctypes 多維陣列
    c_cubic_Xinv = Matrix16x16(*[ (c_double * 16)(*map(float, row)) for row in cubic_Xinv ])
    c_img = img.ctypes.data_as(POINTER(c_int))
    c_coef = coef_1B1A.ctypes.data_as(POINTER(c_double))

    # 呼叫 C 函式
    m.cubic_coef(c_cubic_Xinv, Length, width, c_img, c_coef)

    return coef_1B1A


def get_cubic_coef_2B2A(cubic_Xinv, Length, img):
    import numpy as np
    from ctypes import cdll, c_int, c_double, POINTER

    coef_2B2A = np.zeros((2 * Length + 1, 2 * Length + 1, 16), dtype=np.float64)
    Length = int(Length)
    img = img.astype(np.int32)
    height, width = img.shape

    # 載入 DLL
    m = cdll.LoadLibrary(f'{CF.DLL_DIR}/cubic_coef_2B2A.dll')

    # 宣告 C 函式參數型態
    Matrix16x16 = (c_double * 16) * 16
    m.cubic_coef.argtypes = [
        Matrix16x16,         # cubic_Xinv 16x16 double
        c_int,               # Length
        c_int,               # width
        POINTER(c_int),      # img 一維 int array
        POINTER(c_double)    # Coef 一維 double array
    ]
    m.cubic_coef.restype = None

    # 轉換 numpy → ctypes 多維陣列
    c_cubic_Xinv = Matrix16x16(*[ (c_double * 16)(*map(float, row)) for row in cubic_Xinv ])
    c_img = img.ctypes.data_as(POINTER(c_int))
    c_coef = coef_2B2A.ctypes.data_as(POINTER(c_double))

    # 呼叫 C 函式
    m.cubic_coef(c_cubic_Xinv, Length, width, c_img, c_coef)

    return coef_2B2A


def get_cubic_value(u, v, coefficient):
    U = np.array([1, u, u*u, u*u*u], dtype=float)
    V = np.array([1, v, v*v, v*v*v], dtype=float)
    gray_value = U.dot(coefficient.dot(np.transpose(V))) # U*coefficient*V
    return gray_value


def get_cubic_value_spline(block4x4, x, y):
    import numpy as np
    from scipy.interpolate import CubicSpline
    # block4x4: 4x4 區塊
    # 先對每一列做 x 方向的三次樣條
    temp = np.array([CubicSpline(range(4), row)(x) for row in block4x4])
    # 再對結果做 y 方向的三次樣條
    value = CubicSpline(range(4), temp)(y)
    return value