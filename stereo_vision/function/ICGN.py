
import Config as CF
import numpy as np
from ctypes import cdll, c_int, c_double, POINTER

# update target_img_subset(Size * Size). if not deformed, skip warp_coef
def update_target_img_subset(Size, img, point_ini, warp_coef=None):
    img = img.astype(np.float64)
    img_flat = img.flatten(order='C') # C:n row major
    height, width = img.shape
    if warp_coef is None:
        warp_coef = np.eye(3, dtype=np.float64)
    target_matrix_g_flat = np.zeros(Size*Size, dtype=np.float64) # create new 1d array
    # ========== 
    m = cdll.LoadLibrary(f'{CF.DLL_DIR}/ICGN.dll')
    
    m.update_target_img_subset.argtypes = [
    POINTER(c_double), POINTER(c_double), POINTER(c_double), POINTER(c_double),
    c_int, c_int, c_int
    ]

    m.update_target_img_subset.restype = None

    img_flat_ptr = img_flat.ctypes.data_as(POINTER(c_double))
    target_matrix_g_flat_ptr = target_matrix_g_flat.ctypes.data_as(POINTER(c_double))
    point_ini_ptr = point_ini.ctypes.data_as(POINTER(c_double))
    warp_coef_ptr = warp_coef.ctypes.data_as(POINTER(c_double))
    # call dll
    m.update_target_img_subset(img_flat_ptr,
                                target_matrix_g_flat_ptr,
                                point_ini_ptr,
                                warp_coef_ptr,
                                width,
                                height,
                                Size)

    target_matrix_g = target_matrix_g_flat.reshape((Size, Size))
    return target_matrix_g