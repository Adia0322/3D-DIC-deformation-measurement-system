from enum import Enum, IntEnum
class Test_Mode(IntEnum):
    IN_PLANE                    = 0
    OUT_OF_PLANE                = 1

class ImageRes(Enum):
    res_640_480                 = (640, 480)
    res_1280_720                = (1280, 720)
    res_1920_1080               = (1920, 1080)
    res_3840_2160               = (3840, 2160)

TEST_DBG_TRUN_ON                = 1
TEST_DBG_TRUN_OFF               = 0
TEST_ROTATE_IMG_ON              = 1
TEST_ROTATE_IMG_OFF             = 0
TEST_REC_TRUN_ON                = 1
TEST_REC_TRUN_OFF               = 0
TEST_GAUSSIANBLUR_ON            = 1
TEST_GAUSSIANBLUR_OFF           = 0

# camera parameters
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

TEST_MODE                       = Test_Mode.IN_PLANE
TEST_SHOW_DBG_EN                = TEST_DBG_TRUN_OFF
TEST_ROTATE_IMG_EN              = TEST_ROTATE_IMG_OFF
TEST_REC_IMG_EN                 = TEST_REC_TRUN_ON
TEST_GAUSSIANBLUR_EN            = TEST_GAUSSIANBLUR_OFF
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

# pso parms
PSO_POPULATION                  = 30

# capture images
LOAD_MIN                        = 0
LOAD_CUR                        = 5
LOAD_MAX                        = 5

# camera calibration
CAL_CHESSBOARD_SIZE             = (9,6)
CAL_IMAGE_RES                   = ImageRes.res_640_480.value
CAL_ITERATION_TIMES             = 100
CAL_ACCURACY                    = 0.001
CAL_SQUARE_SIZE                 = 8 # unit: mm

# DIC
DIC_ICGN_ACCURACY_INIT          = 1
DIC_ICGN_ACCURACY               = 0.001
DIC_ICGN_MAX_ITER               = 10