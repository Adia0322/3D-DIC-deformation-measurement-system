
from enum import Enum
## camera parameters
# focal
CAM1_FOCAL                      = 70
CAM2_FOCAL                      = 70
# camera index
CAM1_ID                         = 1
CAM2_ID                         = 0
CAM_BUFFER_SIZE_EN              = 0
CAM_AUTO_FOCAL_EN               = 0
CAM_AUTO_WB_EN                  = 0

# image
TEST_MAX_IMG_CNT                = 11
TEST_TARGET_IMG_PAIR_NUM        = 1

## test mode (0: in-plane, 1:out-of-plane)
TEST_MODE                       = 0
TEST_SHOW_DBG_EN                = 0
TEST_ROTATE_IMG_EN              = 0
TEST_REC_IMG_EN                 = 1
TEST_IMG_DIR                    = 'Target'
TEST_POINT_LEN                  = 5
TEST_POINT_ARRAY                = TEST_POINT_LEN * TEST_POINT_LEN
TEST_INTERVAL                   = 10
TEST_SUBSET_SIZE_1B2B           = 31
TEST_SUBSET_SIZE_1B1A           = 31
TEST_SUBSET_SIZE_2B2A           = 31
TEST_SCAN_SIZE_1B2B             = 31
TEST_SCAN_SIZE_1B1A             = 31
TEST_SCAN_SIZE_2B2A             = 31
TEST_GAUSSIANBLUR_EN            = 0

# PSO parms
PSO_population                  = 30

## CAPTURE IMAGE
LOAD_MIN                        = 0
LOAD_CUR                        = 5
LOAD_MAX                        = 5

## camera calibration
CAL_CHESSBOARD_SIZE             = (9,6)
CAL_IMAGE_RES                   = (640,480)
CAL_ITERATION_TIMES             = 100
CAL_ACCURACY                    = 0.001
CAL_SQUARE_SIZE                 = 8 # unit: mm

# DIC
DIC_ICGN_ACCURACY_INIT          = 1
DIC_ICGN_ACCURACY               = 0.001
DIC_ICGN_MAX_ITER               = 10


class Test_Mode(Enum):
    in_plane            = 0
    out_of_plane        = 1




