import numpy as np
import os
import cv2 as cv
import config as CF
import config_user as CF_user
import ctypes
from ctypes import cdll, c_int, c_double, POINTER
import camera_calibration.image_calibration as img_cal
from tools.vision.src.processor import rotate_image, check_file_path, run_Gaussian_blur

class DIC_user_config:
    def __init__(self):
        self.pt_mat_side_len            = CF_user.TEST_POINT_ARRAY
        self.subset_len_1B2B            = CF_user.TEST_SUBSET_SIZE_1B2B
        self.subset_len_1B1A            = CF_user.TEST_SUBSET_SIZE_1B1A
        self.subset_len_2B2A            = CF_user.TEST_SUBSET_SIZE_2B2A
        self.tranlation                 = None

class Stereo_DIC_result:
    def __init__(self):
        self.X_after                    = None
        self.Y_after                    = None
        self.Z_after                    = None

class Stereo_DIC_session:
    def __init__(self, cfg: DIC_user_config):
        self.cfg                        = cfg
        self.img_buf                    = img_buffer()
        self.cal_info                   = calibration_info()
        self.dic_buf                    = DIC_buffer(cfg)
        self.lib                        = Library()
        self.result_buf                 = result_buffer(cfg)
        self.icgn_proc_1B2B             = ICGN_processor(cfg.subset_len_1B2B)
        self.icgn_proc_1B1A             = ICGN_processor(cfg.subset_len_1B1A)
        self.icgn_proc_2B2A             = ICGN_processor(cfg.subset_len_2B2A)
        self.pso_proc                   = PSO_processor()
        
    def get_img_dir(self, cam_idx):
        if (CF_user.TEST_MODE == 0):
            if cam_idx == 1:
                return CF.IMAGE_TARGET_IN_CAM1_DIR
            else:
                return CF.IMAGE_TARGET_IN_CAM2_DIR
        elif (CF_user.TEST_MODE == 1):
            if cam_idx == 1:
                return CF.IMAGE_TARGET_OUT_CAM1_DIR
            else:
                return CF.IMAGE_TARGET_OUT_CAM2_DIR
            
    def check_img_path(self, path, file_name):
        if check_file_path(path) != 0:
            print(f"{file_name}: {path} not found!")
            exit(1)

    def get_ref_file_path(self, file_name):
        self.img_buf.img1_ref_path = os.path.join(self.get_img_dir(cam_idx=1), file_name)
        self.img_buf.img2_ref_path = os.path.join(self.get_img_dir(cam_idx=2), file_name)
        self.check_img_path(self.img_buf.img1_ref_path, "img1_ref_path")
        self.check_img_path(self.img_buf.img2_ref_path, "img2_ref_path")

    def get_cur_file_path(self, file_name):
        self.img_buf.img1_cur_path = os.path.join(self.get_img_dir(cam_idx=1), file_name)
        self.img_buf.img2_cur_path = os.path.join(self.get_img_dir(cam_idx=2), file_name)
        self.check_img_path(self.img_buf.img1_cur_path, "img1_cur_path")
        self.check_img_path(self.img_buf.img2_cur_path, "img2_cur_path")
    
    def read_img_ref(self):
        self.img_buf.img1_ref               = cv.imread(str(self.img_buf.img1_ref_path))
        self.img_buf.img2_ref               = cv.imread(str(self.img_buf.img2_ref_path))
        if self.img_buf.img1_ref_path is None: print(f"[ERROR] fail to read img1_ref_path: {self.img_buf.img1_ref_path}")
        if self.img_buf.img2_ref_path is None: print(f"[ERROR] fail to read img2_ref_path: {self.img_buf.img2_ref_path}")
    
    def read_img_cur(self):
        self.img_buf.img1_cur               = cv.imread(str(self.img_buf.img1_cur_path))
        self.img_buf.img2_cur               = cv.imread(str(self.img_buf.img2_cur_path))
        if self.img_buf.img1_cur_path is None: print(f"[ERROR] fail to read img1_cur_path: {self.img_buf.img1_cur_path}")
        if self.img_buf.img2_cur_path is None: print(f"[ERROR] fail to read img2_cur_path: {self.img_buf.img2_cur_path}")

    def load_stereo_images_ref(self, file_name):
        self.get_ref_file_path(file_name)
        self.read_img_ref()
        return self
    
    def load_stereo_images_cur(self, file_name):
        self.get_cur_file_path(file_name)
        self.read_img_cur()
        return self

    def pre_process(self):
        self.img_buf.img1_ref = rotate_image(self.img_buf.img1_ref, angle=90, is_enable=CF_user.TEST_ROTATE_IMG_EN)
        self.img_buf.img2_ref = rotate_image(self.img_buf.img2_ref, angle=-90, is_enable=CF_user.TEST_ROTATE_IMG_EN)

        self.img_buf.img1_ref_rec, self.img_buf.img2_ref_rec = img_cal.undistortRectify(self.img_buf.img1_ref, self.img_buf.img2_ref)
        
        self.img_buf.img1_ref_rec_show = np.copy(self.img_buf.img1_ref_rec)
        self.img_buf.img2_ref_rec_show = np.copy(self.img_buf.img2_ref_rec)
        self.img_buf.img1_ref_rec = run_Gaussian_blur(self.img_buf.img1_ref_rec, is_enable=CF_user.TEST_GAUSSIANBLUR_EN)
        self.img_buf.img2_ref_rec = run_Gaussian_blur(self.img_buf.img2_ref_rec, is_enable=CF_user.TEST_GAUSSIANBLUR_EN)
        self.img_buf.img1_ref_rec_gray = cv.cvtColor(self.img_buf.img1_ref_rec, cv.COLOR_BGR2GRAY)
        self.img_buf.img2_ref_rec_gray = cv.cvtColor(self.img_buf.img2_ref_rec, cv.COLOR_BGR2GRAY)
        self.img_buf.img1_ref_rec_gray = self.img_buf.img1_ref_rec_gray.astype(np.double)
        self.img_buf.img2_ref_rec_gray = self.img_buf.img2_ref_rec_gray.astype(np.double)
        return self
    
    def pre_process_cur(self):
        self.img_buf.img1_cur = rotate_image(self.img_buf.img1_cur, angle=90, is_enable=CF_user.TEST_ROTATE_IMG_EN)
        self.img_buf.img2_cur = rotate_image(self.img_buf.img2_cur, angle=-90, is_enable=CF_user.TEST_ROTATE_IMG_EN)

        self.img_buf.img1_cur_rec, self.img_buf.img2_cur_rec = img_cal.undistortRectify(self.img_buf.img1_cur, self.img_buf.img2_cur)
        
        self.img_buf.img1_cur_rec_show      = np.copy(self.img_buf.img1_cur_rec)
        self.img_buf.img2_cur_rec_show      = np.copy(self.img_buf.img2_cur_rec)
        self.img_buf.img1_cur_rec           = run_Gaussian_blur(self.img_buf.img1_cur_rec, is_enable=CF_user.TEST_GAUSSIANBLUR_EN)
        self.img_buf.img2_cur_rec           = run_Gaussian_blur(self.img_buf.img2_cur_rec, is_enable=CF_user.TEST_GAUSSIANBLUR_EN)
        self.img_buf.img1_cur_rec_gray      = cv.cvtColor(self.img_buf.img1_cur_rec, cv.COLOR_BGR2GRAY)
        self.img_buf.img2_cur_rec_gray      = cv.cvtColor(self.img_buf.img2_cur_rec, cv.COLOR_BGR2GRAY)
        self.img_buf.img1_cur_rec_gray      = self.img_buf.img1_cur_rec_gray.astype(np.double)
        self.img_buf.img2_cur_rec_gray      = self.img_buf.img2_cur_rec_gray.astype(np.double)
        return self
    
    def prepare_display_windows(self, message='set a reference point on img_1B'):
        return self

    def get_img_sobel(self):
        self.img_buf.img1_ref_sobel_y       = cv.Sobel(self.img_buf.img1_ref_rec_gray, cv.CV_64F, 0, 1)*0.125
        self.img_buf.img1_ref_sobel_x       = cv.Sobel(self.img_buf.img1_ref_rec_gray, cv.CV_64F, 1, 0)*0.125
        # self.img_buf.img2_ref_sobel_y     = cv.Sobel(self.img_buf.img2_ref_rec_gray, cv.CV_64F, 0, 1)*0.125
        # self.img_buf.img2_ref_sobel_x     = cv.Sobel(self.img_buf.img2_ref_rec_gray, cv.CV_64F, 1, 0)*0.125
        return self
    
    def get_reprojection_info(self):
        cv_file = cv.FileStorage()
        cv_file.open(CF.STEREO_MAP_PATH, cv.FileStorage_READ)
        Q = cv_file.getNode('Q').mat()
        cv_file.release()
        self.cal_info.focal                 = Q[2][3]       # focal (unit:pixel)
        self.cal_info.baseline              = 1/Q[3][2]     # baseline (unit:mm)
        self.cal_info.principal_x           = -Q[0][3]      # The pt center coor of the camera
        self.cal_info.principal_y           = -Q[1][3]      # The pt center coor of the camera

    def build_dic_cfg():
        return

    def disparity_to_3d_pt(self, cam1_x, cam1_y, cam2_x, pt_type='pt_ref'):
        disparity = cam1_x - cam2_x # get disparity: xl-xr (unit:pixel)
        disparity_inv = np.divide(1, disparity)
        X = (cam1_x - self.cal_info.principal_x) * self.cal_info.baseline * disparity_inv
        Y = (cam1_y - self.cal_info.principal_y) * self.cal_info.baseline * disparity_inv
        Z = self.cal_info.focal * self.cal_info.baseline * disparity_inv
        return X, Y, Z 

    def save_result(self, C2_B_x, C2_B_y, X_ref, Y_ref, Z_ref):
        return


    def free_show_image(self):
        self.img_buf.img1_ref_rec_show      = None
        self.img_buf.img2_ref_rec_show      = None
        return
    
    def select_target_pt():
        return
    
    def check_ini_pt():
        return
    
def create_session(cfg: DIC_user_config):
    return Stereo_DIC_session(cfg)

class img_buffer:
    def __init__(self):
        self.force_direct                   = None
        self.img1_ref_path                  = None
        self.img2_ref_path                  = None
        self.img1_ref                       = None
        self.img2_ref                       = None
        self.img1_ref_rec                   = None
        self.img2_ref_rec                   = None
        self.img1_ref_rec_show              = None
        self.img2_ref_rec_show              = None
        self.img1_ref_rec_gray              = None
        self.img2_ref_rec_gray              = None

        self.img1_cur_path                  = None
        self.img2_cur_path                  = None
        self.img1_cur                       = None
        self.img2_cur                       = None
        self.img1_cur_rec                   = None
        self.img2_cur_rec                   = None
        self.img1_cur_rec_show              = None
        self.img2_cur_rec_show              = None
        self.img1_cur_rec_gray              = None
        self.img2_cur_rec_gray              = None

        self.img1_ref_sobel_y               = None
        self.img1_ref_sobel_x               = None
        self.img2_cur_sobel_y               = None
        self.img2_cur_sobel_x               = None

        self.img1_cur_pyramid               = None
        self.img2_cur_pyramid               = None

class calibration_info:
    def __init__(self):
        self.baseline                       = None
        self.focal                          = None
        self.principal_x                    = None
        self.principal_y                    = None

class DIC_buffer:
    def __init__(self, cfg: DIC_user_config):
        self.img1_ref_sobel_y               = None
        self.img1_ref_sobel_x               = None
        self.C1B_points                     = np.zeros((cfg.pt_mat_side_len, cfg.pt_mat_side_len, 2), dtype=int)
        self.C2B_points                     = np.zeros((cfg.pt_mat_side_len, cfg.pt_mat_side_len, 2), dtype=np.double)
        self.WC_bef_zone                    = np.zeros((cfg.pt_mat_side_len, cfg.pt_mat_side_len, 3), dtype=np.double)
        self.WC_aft_zone                    = np.zeros((cfg.pt_mat_side_len, cfg.pt_mat_side_len, 3), dtype=np.double)
        self.H_1B_inv_all                   = np.zeros((cfg.pt_mat_side_len, cfg.pt_mat_side_len, 6, 6), dtype=np.double)
        self.H_2B_inv_all                   = np.zeros((cfg.pt_mat_side_len, cfg.pt_mat_side_len, 6, 6), dtype=np.double)
        self.J_1B_all                       = np.zeros((cfg.pt_mat_side_len, cfg.pt_mat_side_len, cfg.subset_len_1B1A, cfg.subset_len_1B1A, 6), dtype=np.double)
        self.J_2B_all                       = np.zeros((cfg.pt_mat_side_len, cfg.pt_mat_side_len, cfg.subset_len_2B2A, cfg.subset_len_2B2A, 6), dtype=np.double)
        self.img_1B_sub_zone                = np.zeros((cfg.pt_mat_side_len, cfg.pt_mat_side_len, cfg.subset_len_1B1A, cfg.subset_len_1B1A), dtype=np.double)
        self.img_2B_sub_zone                = np.zeros((cfg.pt_mat_side_len, cfg.pt_mat_side_len, cfg.subset_len_2B2A, cfg.subset_len_2B2A), dtype=np.double)

class Library:
    def __init__(self):
        self.ICGN                           = cdll.LoadLibrary(f'{CF.BUILD_DIR}/ICGN.dll')
        self.PSO                            = cdll.LoadLibrary(f'{CF.BUILD_DIR}/PSO.dll')
        self.interp                         = cdll.LoadLibrary(f'{CF.BUILD_DIR}/cubic_interp.dll')
        self._set_type_ICGN()
        self._set_type_PSO()
        self._set_type_interp()

    def _set_type_ICGN(self):
        self.ICGN.update_target_img_subset_core.argtypes = [
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double),
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int
        ]
        self.ICGN.update_target_img_subset_core.restype = None
        return
    
    def _set_type_PSO(self):
        self.PSO.pso_optimize.argtypes = [
            ctypes.POINTER(ctypes.c_double),    # ref_img
            ctypes.POINTER(ctypes.c_double),    # cur_img
            ctypes.c_int,                       # width
            ctypes.c_int,                       # height
            ctypes.c_int,                       # population
            ctypes.c_int,                       # subset_side_len
            ctypes.POINTER(ctypes.c_double),    # img_ref_pt
            ctypes.POINTER(ctypes.c_double),    # img_cur_pt
            ctypes.POINTER(ctypes.c_double)     # result_buffer
        ]
        self.PSO.pso_optimize.restype = None
        return
    
    def _set_type_interp(self):
        self.interp.get_bicubic_interp_value.argtypes = [
            ctypes.POINTER(ctypes.c_double),
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_double,
            ctypes.c_double
        ]
        self.interp.get_bicubic_interp_value.restype = ctypes.c_double
        return

class result_buffer:
    def __init__(self, cfg: DIC_user_config):
        self.disM                           = np.zeros((cfg.pt_mat_side_len, cfg.pt_mat_side_len, 3), dtype=np.double)
        self.disM_out                       = np.zeros((cfg.pt_mat_side_len, cfg.pt_mat_side_len), dtype=np.double)
        self.disM_in_1                      = np.zeros((cfg.pt_mat_side_len, cfg.pt_mat_side_len), dtype=np.double)
        self.disM_in_2                      = np.zeros((cfg.pt_mat_side_len, cfg.pt_mat_side_len), dtype=np.double)
        self.stress_in                      = None
        self.stress_out                     = None

class ICGN_processor:
    def __init__(self, subset_size_len):
        self.subset_size_len = subset_size_len
        n = subset_size_len
        self._target_subset_flat            = np.zeros(n * n, dtype=np.float64)
        self._target_subset_flat_ptr        = self._target_subset_flat.ctypes.data_as(POINTER(c_double))
        self._target_subset_2d              = self._target_subset_flat.reshape((n, n))   # view: zero copy
        self._warp_eye                      = np.eye(3, dtype=np.float64)
        self._warp_inc                      = np.zeros((3, 3), dtype=np.float64)
        self._warp_inc[2]                   = [0, 0, 1]  # fix the last row
        self._delta_P                       = np.zeros(6, dtype=np.float64)
        self._corr_sum                      = np.zeros((6, 1), dtype=np.float64)

    def update_target_img_subset(self, img, point_ini, lib_ICGN, warp_coef=None):
        img = np.asarray(img, dtype=np.float64)
        if not img.flags['C_CONTIGUOUS']:
            img = np.ascontiguousarray(img)
        img_flat = img.ravel()   # ravel (no copy）vs. flatten（copy）

        wc = self._warp_eye if warp_coef is None else warp_coef
        height, width = img.shape

        lib_ICGN.update_target_img_subset_core(
            img_flat.ctypes.data_as(POINTER(c_double)),
            self._target_subset_flat.ctypes.data_as(POINTER(c_double)),
            point_ini.ctypes.data_as(POINTER(c_double)),
            wc.ctypes.data_as(POINTER(c_double)),
            width,
            height,
            self.subset_size_len
        )
        return self._target_subset_2d

class PSO_processor:
    def __init__(self):
        self._result_buf     = np.zeros(3,  dtype=np.float64)
        self._ref_pt_pos     = np.zeros(2,  dtype=np.float64)
        self._cur_pt_pos     = np.zeros(2,  dtype=np.float64)
        self._result_ptr     = self._result_buf.ctypes.data_as(POINTER(c_double))
        self._ref_pt_ptr     = self._ref_pt_pos.ctypes.data_as(POINTER(c_double))
        self._cur_pt_ptr     = self._cur_pt_pos.ctypes.data_as(POINTER(c_double))

    def run_PSO(self, lib_PSO):
        return

class system_config:
    def __init__(self):
        self._img_ref                       = None
        self._img_cur                       = None
        self._result_buffer                 = None
        self._img_ref_pt_pos                = None
        self._img_cur_pt_pos                = None
        self._result_buffer                 = None

