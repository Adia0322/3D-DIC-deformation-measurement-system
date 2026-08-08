## PATH
import os
import config_user as CF_user
current_file_path               = os.path.abspath(__file__)
WORKSPACE                       = os.path.dirname(current_file_path)
BUILD_DIR                       = os.path.join(WORKSPACE, "build")
IMAGE_DIR                       = os.path.join(WORKSPACE, "data")
IMAGE_CAL_DIR                   = os.path.join(IMAGE_DIR, "calibration")
IMAGE_CAL_LEFT_DIR              = os.path.join(IMAGE_DIR, "calibration", "StereoLeft")
IMAGE_CAL_RIGHT_DIR             = os.path.join(IMAGE_DIR, "calibration", "StereoRight")
IMAGE_TARGET_DIR                = os.path.join(IMAGE_DIR, CF_user.TEST_IMG_DIR)
IMAGE_TARGET_IN_DIR             = os.path.join(IMAGE_DIR, CF_user.TEST_IMG_DIR, "in")
IMAGE_TARGET_OUT_DIR            = os.path.join(IMAGE_DIR, CF_user.TEST_IMG_DIR, "out")
IMAGE_TARGET_IN_CAM1_DIR        = os.path.join(IMAGE_DIR, CF_user.TEST_IMG_DIR, "in", "cam1")
IMAGE_TARGET_IN_CAM2_DIR        = os.path.join(IMAGE_DIR, CF_user.TEST_IMG_DIR, "in", "cam2")
IMAGE_TARGET_OUT_CAM1_DIR       = os.path.join(IMAGE_DIR, CF_user.TEST_IMG_DIR, "out", "cam1")
IMAGE_TARGET_OUT_CAM2_DIR       = os.path.join(IMAGE_DIR, CF_user.TEST_IMG_DIR, "out", "cam2")
STEREO_MAP_PATH                 = f"{BUILD_DIR}/stereoMap.xml"



