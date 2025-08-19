
import Config as CF


# ## 雙線性
# def get_subpixel_value_cv(img, x, y):
#     import cv2 as cv
#     # getRectSubPix 取單個像素 (1x1 patch)
#     patch = cv.getRectSubPix(img, (1, 1), (x, y))  # (x, y) 為中心座標
#     return patch[0, 0]

## cubic



# ## 雙線性
# def get_subpixel_value_cv(img, x, y):
#     import cv2 as cv
#     # getRectSubPix 取單個像素 (1x1 patch)
#     patch = cv.getRectSubPix(img, (1, 1), (x, y))  # (x, y) 為中心座標
#     return patch[0, 0]

## cubic
def get_subpixel_value_cubic(img, x, y):
    # map_coordinates 需要 (row, col) 座標順序
    import numpy as np
    from scipy.ndimage import map_coordinates
    img = img.astype(np.float64)
    coords = np.array([[y], [x]], dtype=np.float64)
    val = map_coordinates(img, coords, order=3, mode='reflect')
    return val[0]



def get_subpixel_cv(img, x_coords, y_coords):
    import numpy as np
    import cv2 as cv
    val = cv.getRectSubPix(img.astype(np.float32), patchSize=(1,1), center=(x_coords, y_coords))
    return val[0,0]



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
    import numpy as np
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



def cubic_kernel(t, a=-0.5):
    """ Keys' bicubic kernel (a=-0.5 is Catmull-Rom spline) """
    t = abs(t)
    if t <= 1:
        return (a + 2) * (t**3) - (a + 3) * (t**2) + 1
    elif t < 2:
        return a * (t**3) - 5*a * (t**2) + 8*a*t - 4*a
    else:
        return 0.0

def get_cubic_coef(img, x, y, a=-0.5):
    import numpy as np
    """
    給定影像與浮點座標 (x, y)，回傳 4x4 的 bicubic 插值係數，
    同時計算對應像素值。
    
    img: 2D numpy array (灰階影像)
    x, y: float 座標
    a: cubic kernel 參數 (預設 -0.5)
    """
    x0 = int(np.floor(x))
    y0 = int(np.floor(y))

    # 計算水平方向權重
    wx = np.array([cubic_kernel(x - (x0 + i), a) for i in range(-1, 3)])
    wy = np.array([cubic_kernel(y - (y0 + j), a) for j in range(-1, 3)])

    # 係數矩陣 (外積)
    coef = np.outer(wy, wx)

    # 收集 16 個鄰近像素值
    values = np.zeros((4, 4))
    h, w = img.shape
    for j in range(-1, 3):
        for i in range(-1, 3):
            xx = np.clip(x0 + i, 0, w - 1)
            yy = np.clip(y0 + j, 0, h - 1)
            values[j + 1, i + 1] = img[yy, xx]

    # 插值結果
    interpolated_value = np.sum(coef * values)

    return interpolated_value

def get_cubic_X_inv():
    import numpy as np
    # Constant coefficient
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

    return Cubic_Xinv