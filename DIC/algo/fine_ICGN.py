# ===== DIC (digital image correlation) ===== #
import numpy as np
import time
import ctypes
import config.base as CF
import config.user as CF_user
from config.dic import DIC_config
from DIC.common import DIC_search_pt_type

def inverse_warp_func(delta_P):
       a = 1 + delta_P[1]
       b = delta_P[2]
       tx = delta_P[0]

       c = delta_P[4]
       d = 1 + delta_P[5]
       ty = delta_P[3]

       det = a*d - b*c
       inv_det = 1.0 / det

       warp_inc_function_inv = np.array([
       [ d*inv_det,
       -b*inv_det,
       (b*ty - d*tx)*inv_det],

       [-c*inv_det,
       a*inv_det,
       (c*tx - a*ty)*inv_det],

       [0,0,1]
       ], dtype=np.float64)
       return warp_inc_function_inv     

# @profile
def run_fine_ICGN(dic_config: DIC_config, lib_ICGN, ICGN_proc, coarse_dis_x, coarse_dis_y):
       img_ref                                   = dic_config.dic_image.ref
       img_cur                                   = dic_config.dic_image.cur
       img_ref_x                                 = dic_config.img_ref_pt.pt_x
       img_ref_y                                 = dic_config.img_ref_pt.pt_y
       subset_side_len                           = dic_config.subset_ref_info.subset_side_len
       subset_ref_data                           = dic_config.subset_ref_info.subset_data
       H_inv_mat                                 = dic_config.img_grad.H_inv_mat
       J_mat                                     = dic_config.img_grad.J_mat
       translation                               = dic_config.init_param.translate
       search_type                               = dic_config.search_type

       img_ref_pt_x_guess = img_ref_x - translation if search_type == DIC_search_pt_type.initial else img_ref_x
       img_ref_pt_y_guess = img_ref_y

       ref_mat_f = subset_ref_data
       ref_mat_f_mean = np.mean(ref_mat_f)
       delta_f = np.std(ref_mat_f, ddof=0)

       # define displacement vector: P
       x      = coarse_dis_x
       xx     = 0
       xy     = 0
       y      = coarse_dis_y
       yx     = 0
       yy     = 0
       # warp function coefficient of deformed subset
       warp_function = np.array([(1+xx, xy, x),
                                 (yx, 1+yy, y),
                                 (0, 0, 1)], dtype=np.float64)

       cnt = 0
       eps = 1e-12
       limit = CF_user.DIC_ICGN_ACCURACY_INIT
       # [NOTICE] FROM NOW ON IS (X,Y), NOT (Y,X) anymore!!
       point_ini = np.array((img_ref_pt_x_guess, img_ref_pt_y_guess), dtype=np.float64)
       sub_len_h = int(0.5*(subset_side_len-1)) 
       start_ICGN = time.time()
       while limit > CF_user.DIC_ICGN_ACCURACY and cnt < CF_user.DIC_ICGN_MAX_ITER:
              target_mat_g                = ICGN_proc.update_target_img_subset(img_cur, point_ini, lib_ICGN, warp_function)
              target_mat_g_mean           = np.mean(target_mat_g)
              diff_g                      = target_mat_g - target_mat_g_mean
              delta_g                     = np.sqrt(np.mean(diff_g * diff_g))

              corelation_sum              = np.zeros(6, dtype=np.float64) 
              ratio                       = delta_f / (delta_g + eps) # eps: prevent zero
              residual_F_G                = (ref_mat_f - ref_mat_f_mean) - ratio * (target_mat_g - target_mat_g_mean)
              corelation_sum              = np.tensordot(J_mat, residual_F_G, axes=([0,1],[0,1]))

              corelation_sum              = corelation_sum.reshape(6,1) 
              delta_P                     = (-H_inv_mat @ corelation_sum).ravel() # flatten turn 2d array to 1d array to get scalar

              limit = np.sqrt(np.square(delta_P[0]) + np.square(delta_P[1] * sub_len_h)+
                              np.square(delta_P[2] * sub_len_h) + np.square(delta_P[3])+
                              np.square(delta_P[4] * sub_len_h) + np.square(delta_P[5] * sub_len_h))
              
              warp_inc_function_inv       = inverse_warp_func(delta_P)
              warp_function               = warp_function @ warp_inc_function_inv
              cnt += 1
       # print(f"cnt: {cnt}")       
       end_ICGN = time.time()
       time_ICGN = end_ICGN - start_ICGN
       # print(f"time_ICGN: {time_ICGN:.5f}")
       
       local_x = warp_function[0][2]
       local_y = warp_function[1][2]
       # print(f"(X,Y)=({X:.3f},{Y:.3f})")
       img_cur_y = local_y + img_ref_pt_y_guess
       img_cur_x = local_x + img_ref_pt_x_guess
       return img_cur_x, img_cur_y
