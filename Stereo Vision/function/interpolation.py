import Config as CF
import numpy as np
from ctypes import cdll, c_int, c_double, POINTER

def bicubic(img, width, height, pt_x, pt_y):
    img_double = img.astype(np.float64)  
    m = cdll.LoadLibrary(f'{CF.DLL_DIR}/cubic_interp.dll')
    #  argtypes / restype
    m.get_bicubic_interp_value.argtypes = [
    POINTER(c_double), c_int, c_int, c_double, c_double
    ]
    m.get_bicubic_interp_value.restype = c_double

    # array pointer
    img_ptr = img_double.ctypes.data_as(POINTER(c_double))

    # call dll
    result = m.get_bicubic_interp_value(img_ptr, width, height, pt_x, pt_y)

    return result
