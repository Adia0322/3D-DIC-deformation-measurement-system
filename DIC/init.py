import config_user as CF_user
import tools.math.src.hessian as lib_hessian
import cv2 as cv
import numpy as np

def pre_calculate_img_grad_1B(session, C1_B_x, C1_B_y, row, col):
    session.dic_buf.C1B_points[row][col][0] = C1_B_y
    session.dic_buf.C1B_points[row][col][1] = C1_B_x
    len_h = int(0.5*(CF_user.TEST_SUBSET_SIZE_1B2B-1))
    img_grad_1B_y = session.img_buf.img1_ref_sobel_y[C1_B_y - len_h:C1_B_y + len_h + 1,\
                                                     C1_B_x - len_h:C1_B_x + len_h + 1]
    img_grad_1B_x = session.img_buf.img1_ref_sobel_x[C1_B_y - len_h:C1_B_y + len_h + 1,\
                                                     C1_B_x - len_h:C1_B_x + len_h + 1]
    H_inv_1B, J_1B = lib_hessian.get_Hinv_jacobian(CF_user.TEST_SUBSET_SIZE_1B2B, img_grad_1B_x, img_grad_1B_y)
    session.dic_buf.H_1B_inv_all[row][col][:][:]   = H_inv_1B[:][:]
    session.dic_buf.J_1B_all[row][col][:][:][:]    = J_1B[:][:][:]
    return


def pre_calculate_img_grad_2B(session, C2_B_x, C2_B_y, row, col):
    img_2B_sub = session.icgn_proc_1B2B.update_target_img_subset(
        session.img_buf.img2_ref_rec_gray,
        np.array((C2_B_x, C2_B_y), dtype=np.float64),
        session.lib.ICGN,
        warp_coef=None
    )
    
    pad = 1  # Sobel need more 1 pixel to expand boarder
    img_2B_sub_pad = cv.copyMakeBorder(img_2B_sub, pad, pad, pad, pad, borderType=cv.BORDER_REFLECT)
    img_2B_sobel_y = cv.Sobel(img_2B_sub_pad, cv.CV_64F, 0, 1)*0.125
    img_2B_sobel_x = cv.Sobel(img_2B_sub_pad, cv.CV_64F, 1, 0)*0.125
    img_grad_2B2A_y = img_2B_sobel_y[pad:-pad, pad:-pad]
    img_grad_2B2A_x = img_2B_sobel_x[pad:-pad, pad:-pad]
    
    H_inv_2B, J_2B = lib_hessian.get_Hinv_jacobian(CF_user.TEST_SUBSET_SIZE_2B2A, img_grad_2B2A_x, img_grad_2B2A_y) 
    session.dic_buf.H_2B_inv_all[row][col][:][:]    = H_inv_2B
    session.dic_buf.J_2B_all[row][col][:][:][:]     = J_2B
    session.dic_buf.img_2B_sub_zone[row][col][:][:] = img_2B_sub
    return