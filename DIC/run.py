from enum import IntEnum
import numpy as np
import config.user as CF_user
import DIC.common as dic_common
import DIC.algo.coarse_PSO as coarse_PSO
import DIC.algo.fine_ICGN as fine_ICGN

from copy import deepcopy
from config.dic import (
    DIC_config, DIC_Image, Img_Ref_Pt_Pos, 
    Stereo_DIC_Init_Param, Subset_Info, 
    Img_Grad_Info, Coarse_Search_Method, PSO_Config
)

def set_base_dic_cfg():
    base_cfg = DIC_config (
        coarse_method       = Coarse_Search_Method.PSO,
        coarse_method_cfg   = PSO_Config(CF_user.PSO_POPULATION),

        dic_image           = DIC_Image(ref = None, cur = None),
        img_ref_pt          = None,
        init_param          = Stereo_DIC_Init_Param(translate = None),
        subset_ref_info     = None,
        subset_cur_info     = Subset_Info(None, subset_side_len = None),
        img_grad            = None,

        search_type         = dic_common.DIC_search_pt_type.normal
    )
    return base_cfg

def build_dic_cfg_1B2B(session, pt_x, pt_y, sub_ref, H_inv, J, trans=0):
    cfg = deepcopy(set_base_dic_cfg())
    cfg.dic_image           = DIC_Image(ref = session.img_buf.img1_ref_rec_gray, cur = session.img_buf.img2_ref_rec_gray)
    cfg.img_ref_pt          = Img_Ref_Pt_Pos(pt_x, pt_y)
    cfg.init_param          = Stereo_DIC_Init_Param(translate = (trans))
    cfg.subset_ref_info     = Subset_Info(sub_ref, subset_side_len = CF_user.TEST_SUBSET_SIZE_1B2B)
    cfg.subset_cur_info     = Subset_Info(None, subset_side_len = CF_user.TEST_SUBSET_SIZE_1B2B)
    cfg.img_grad            = Img_Grad_Info(H_inv_mat=H_inv, J_mat=J)
    cfg.search_type         = dic_common.DIC_search_pt_type.initial
    return cfg

def build_dic_cfg_1B1A(session, pt_x, pt_y, sub_ref, H_inv, J, trans=0):
    cfg = deepcopy(set_base_dic_cfg())
    cfg.dic_image           = DIC_Image(ref = session.img_buf.img1_ref_rec_gray, cur = session.img_buf.img1_cur_rec_gray)
    cfg.img_ref_pt          = Img_Ref_Pt_Pos(pt_x, pt_y)
    cfg.init_param          = Stereo_DIC_Init_Param(translate = (trans))
    cfg.subset_ref_info     = Subset_Info(sub_ref, subset_side_len = CF_user.TEST_SUBSET_SIZE_1B1A)
    cfg.subset_cur_info     = Subset_Info(None, subset_side_len = CF_user.TEST_SUBSET_SIZE_1B1A)
    cfg.img_grad            = Img_Grad_Info(H_inv_mat=H_inv, J_mat=J)
    cfg.search_type         = dic_common.DIC_search_pt_type.normal
    return cfg

def build_dic_cfg_2B2A(session, pt_x, pt_y, sub_ref, H_inv, J, trans=0):
    cfg = deepcopy(set_base_dic_cfg())
    cfg.dic_image           = DIC_Image(ref = session.img_buf.img2_ref_rec_gray, cur = session.img_buf.img2_cur_rec_gray)
    cfg.img_ref_pt          = Img_Ref_Pt_Pos(pt_x, pt_y)
    cfg.init_param          = Stereo_DIC_Init_Param(translate = (trans))
    cfg.subset_ref_info     = Subset_Info(sub_ref, subset_side_len = CF_user.TEST_SUBSET_SIZE_2B2A)
    cfg.subset_cur_info     = Subset_Info(None, subset_side_len = CF_user.TEST_SUBSET_SIZE_2B2A)
    cfg.img_grad            = Img_Grad_Info(H_inv_mat=H_inv, J_mat=J)
    cfg.search_type         = dic_common.DIC_search_pt_type.normal
    return cfg

def data_init_1B():
    return

def data_init_2B():
    return

def update_displacement_result(session, row, col, C1_A_x, C1_A_y, C2_A_x):
    X_cur, Y_cur, Z_cur = session.disparity_to_3d_pt(
        C1_A_x,
        C1_A_y,
        C2_A_x
    )

    wc_cur = np.array((X_cur, Y_cur, Z_cur))
    wc_bef = session.dic_buf.WC_bef_zone[row][col]
    dis_3D_vec = wc_cur - wc_bef
    session.dic_buf.WC_aft_zone[row][col] = wc_cur
    session.result_buf.disM[row][col][:] = dis_3D_vec
    dis_in_1 = dis_3D_vec[0]
    dis_in_2 = dis_3D_vec[1]
    dis_out  = dis_3D_vec[2]

    # in-plane displacement magnitude
    dis_in_sum = np.sqrt(dis_in_1**2 + dis_in_2**2)

    session.result_buf.disM_out[row][col]  = dis_out
    session.result_buf.disM_in_1[row][col] = dis_in_1
    session.result_buf.disM_in_2[row][col] = dis_in_2

    return dis_out, dis_in_sum

def get_ref_sub(session, C1_B_x, C1_B_y):
    return session.icgn_proc_1B2B.update_target_img_subset(
        session.img_buf.img1_ref_rec_gray,
        np.array((C1_B_x, C1_B_y), dtype=np.float64),
        session.lib.ICGN,
        warp_coef=None
    )

def run_DIC_coarse(session, dic_config):
    return coarse_PSO.run_PSO(dic_config, session.lib.PSO, session.pso_proc)

def run_DIC_fine(session, dic_config, local_coarse_x, local_coarse_y):
    return fine_ICGN.run_fine_ICGN(dic_config, session.lib.ICGN, session.icgn_proc_1B2B, local_coarse_x, local_coarse_y)

def run_dic_1B2B(session, row, col):
    C1_B_y, C1_B_x                      = session.dic_buf.C1B_points[row][col]
    H_inv_1B1A                          = session.dic_buf.H_1B_inv_all[row][col][:][:]
    J_1B1A                              = session.dic_buf.J_1B_all[row][col][:][:][:]
    img_1B_sub                          = get_ref_sub(session, C1_B_x, C1_B_y)

    session.dic_buf.img_1B_sub_zone[row][col] = img_1B_sub

    dic_config = build_dic_cfg_1B2B(
        session,
        C1_B_x,
        C1_B_y,
        img_1B_sub,
        H_inv_1B1A,
        J_1B1A,
        trans=session.cfg.tranlation
    )
    local_coarse_x, local_coarse_y      = run_DIC_coarse(session, dic_config)
    return run_DIC_fine(session, dic_config, local_coarse_x, local_coarse_y)

def run_dic_1B1A(session, row, col):
    C1_B_y, C1_B_x                      = session.dic_buf.C1B_points[row][col]
    H_inv_1B1A                          = session.dic_buf.H_1B_inv_all[row][col][:][:]
    J_1B1A                              = session.dic_buf.J_1B_all[row][col][:][:][:]
    img_1B_sub                          = session.dic_buf.img_1B_sub_zone[row][col]
    dic_config                          = build_dic_cfg_1B1A(session, C1_B_x, C1_B_y, img_1B_sub, H_inv_1B1A, J_1B1A, trans=0)
    local_coarse_x, local_coarse_y      = run_DIC_coarse(session, dic_config)
    return run_DIC_fine(session, dic_config, local_coarse_x, local_coarse_y)

def run_dic_2B2A(session, row, col):
    C2_B_y, C2_B_x                      = session.dic_buf.C2B_points[row][col]
    H_inv_2B2A                          = session.dic_buf.H_2B_inv_all[row][col][:][:]
    J_2B2A                              = session.dic_buf.J_2B_all[row][col][:][:][:]
    img_2B_sub                          = session.dic_buf.img_2B_sub_zone[row][col]
    dic_config                          = build_dic_cfg_2B2A(session, C2_B_x, C2_B_y, img_2B_sub, H_inv_2B2A, J_2B2A, trans=0)
    local_coarse_x, local_coarse_y      = run_DIC_coarse(session, dic_config)
    return run_DIC_fine(session, dic_config, local_coarse_x, local_coarse_y)