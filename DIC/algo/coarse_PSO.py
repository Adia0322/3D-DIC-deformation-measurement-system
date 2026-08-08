from config.dic import DIC_config
import numpy as np
import ctypes
from DIC.common import DIC_search_pt_type
from ctypes import c_double

def run_PSO(dic_config: DIC_config, lib_PSO, PSO_proc):
    img_ref                                   = dic_config.dic_image.ref
    img_cur                                   = dic_config.dic_image.cur
    img_ref_x                                 = dic_config.img_ref_pt.pt_x
    img_ref_y                                 = dic_config.img_ref_pt.pt_y
    subset_side_len                           = dic_config.subset_ref_info.subset_side_len
    translation                               = dic_config.init_param.translate
    population                                = dic_config.coarse_method_cfg.population
    search_type                               = dic_config.search_type

    if img_ref.dtype != np.float64 or not img_ref.flags['C_CONTIGUOUS']:
            img_ref = np.ascontiguousarray(img_ref, dtype=np.float64)
    if img_cur.dtype != np.float64 or not img_cur.flags['C_CONTIGUOUS']:
            img_cur = np.ascontiguousarray(img_cur, dtype=np.float64)
    h, w = img_ref.shape

    img_ref_ptr = img_ref.ctypes.data_as(ctypes.POINTER(c_double))
    img_cur_ptr = img_cur.ctypes.data_as(ctypes.POINTER(c_double))

    img_ref_pt_x_guess = img_ref_x - translation if search_type == DIC_search_pt_type.initial else img_ref_x
    img_ref_pt_y_guess = img_ref_y

    PSO_proc._ref_pt_pos[0] = img_ref_y; PSO_proc._ref_pt_pos[1] = img_ref_x
    PSO_proc._cur_pt_pos[0] = img_ref_pt_y_guess; PSO_proc._cur_pt_pos[1] = img_ref_pt_x_guess

    lib_PSO.pso_optimize(
            img_ref_ptr, 
            img_cur_ptr, 
            w, 
            h, 
            population,
            subset_side_len,
            PSO_proc._ref_pt_ptr,
            PSO_proc._cur_pt_ptr,
            PSO_proc._result_ptr
    )
    local_dis_y, local_dis_x = PSO_proc._result_buf[0], PSO_proc._result_buf[1]

    return local_dis_x, local_dis_y
        