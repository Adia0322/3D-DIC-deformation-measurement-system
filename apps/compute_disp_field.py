
print("\n<< Stereo_DIC >>")
import cv2 as cv
import time
import config.user as CF_user
from tools.vision.src.click_tool import get_click_point
import DIC.run as run_dic
import DIC.init as dic_init

from DIC.session import (
    DIC_user_config, create_session,
)

from config.user import (
    Test_Mode
)

mode = Test_Mode.IN_PLANE
pt_mat_len_half = int((CF_user.TEST_POINT_LEN - 1) / 2)
file_name = f"{CF_user.LOAD_MIN}_{CF_user.LOAD_MAX}kg_image1.jpg"
config = DIC_user_config()
session = create_session(config)
(
    session.load_stereo_images_ref(file_name)
            .pre_process()
            .get_img_sobel()
)

C1_B_x_ini, C1_B_y_ini = get_click_point(session.img_buf.img1_ref_rec_show, 'img1_ref_rec_show', 'set a reference point on img_1B')
C2_B_x_ini, C2_B_y_ini = get_click_point(session.img_buf.img2_ref_rec_show, 'img2_ref_rec_show', 'set a reference point on img_2B')
session.free_show_image()
session.get_reprojection_info()

session.cfg.tranlation = (C1_B_x_ini - C2_B_x_ini)

for row in range(-pt_mat_len_half, pt_mat_len_half + 1, 1):
    for col in range(-pt_mat_len_half, pt_mat_len_half + 1, 1):
        subset_y = row + pt_mat_len_half
        subset_x = col + pt_mat_len_half

        C1_B_x = int(CF_user.TEST_INTERVAL*col + C1_B_x_ini)
        C1_B_y = int(CF_user.TEST_INTERVAL*row + C1_B_y_ini)

        dic_init.pre_calculate_img_grad_1B(session, C1_B_x, C1_B_y, subset_y, subset_x)

        C2_B_x, C2_B_y = run_dic.run_dic_1B2B(session, subset_y, subset_x)

        dic_init.pre_calculate_img_grad_2B(session, C2_B_x, C2_B_y, subset_y, subset_x)

        X_ref, Y_ref, Z_ref = session.disparity_to_3d_pt(C1_B_x, C1_B_y, C2_B_x)
        session.dic_buf.C2B_points[subset_y][subset_x] = (C2_B_y, C2_B_x)
        session.dic_buf.WC_bef_zone[subset_y][subset_x] = (X_ref, Y_ref, Z_ref)
        
        session.img_buf.img1_ref_rec = cv.circle(session.img_buf.img1_ref_rec, (int(C1_B_x), int(C1_B_y)), 5, (0, 255, 255), 1)  
        session.img_buf.img2_ref_rec = cv.circle(session.img_buf.img2_ref_rec, (int(C2_B_x), int(C2_B_y)), 5, (0, 255, 255), 1)

cv.imshow('session.img_buf.img1_ref_rec', session.img_buf.img1_ref_rec)
cv.imshow('session.img_buf.img2_ref_rec', session.img_buf.img2_ref_rec)
cv.waitKey(0)
cv.destroyAllWindows()

dis_sum = 0
total_time = 0
for img_idx in range(1, CF_user.TEST_TARGET_IMG_PAIR_NUM + 1,1):
    loaded_file_name = f"{CF_user.LOAD_CUR}_{CF_user.LOAD_MAX}kg_image{img_idx}.jpg"
    print(f"loaded_file_name: {loaded_file_name}")
    (
        session.load_stereo_images_cur(loaded_file_name)
                .pre_process_cur()
    )
    for row in range(-pt_mat_len_half, pt_mat_len_half + 1, 1):
        for col in range(-pt_mat_len_half, pt_mat_len_half + 1, 1):
            subset_y = row + pt_mat_len_half
            subset_x = col + pt_mat_len_half

            start_DIC = time.time()
            start = time.time()

            C1_A_x, C1_A_y = run_dic.run_dic_1B1A(session, subset_y, subset_x)
            C2_A_x, C2_A_y = run_dic.run_dic_2B2A(session, subset_y, subset_x)

            end_DIC = time.time()
            time_dic = end_DIC - start_DIC
            # print(f"time_dic: {time_dic:.5f}")

            dis_out, dis_in_sum = run_dic.update_displacement_result(session, subset_y, subset_x, C1_A_x, C1_A_y, C2_A_x)
            
            end = time.time()
            increase_time = end - start
            # print(f"increase_time: {increase_time:.4f}")
            total_time += increase_time

            if CF_user.TEST_MODE == Test_Mode.IN_PLANE.value:
                # print(np.round(dis_in_sum, 6))
                dis_sum += dis_in_sum
            else:
                # print(np.round(dis_out, 6))
                dis_sum += dis_out

            session.img_buf.img1_cur_rec = cv.circle(session.img_buf.img1_cur_rec, (int(C1_A_x), int(C1_A_y)), 5, (0, 255, 255), 1)  
            session.img_buf.img2_cur_rec = cv.circle(session.img_buf.img2_cur_rec, (int(C2_A_x), int(C2_A_y)), 5,(0, 255, 255), 1)  
            

cv.imshow('img_1A_rec', session.img_buf.img1_cur_rec)
cv.imshow('img_2A_rec', session.img_buf.img2_cur_rec)
cv.waitKey(0)
cv.destroyAllWindows()

print(f"Average time per point: {total_time / (CF_user.TEST_POINT_ARRAY * CF_user.TEST_TARGET_IMG_PAIR_NUM):.6f}")
print(f"Average dis: {dis_sum / (CF_user.TEST_TARGET_IMG_PAIR_NUM * CF_user.TEST_POINT_ARRAY):.6f}")
print("End")

