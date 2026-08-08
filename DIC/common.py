from enum import IntEnum
import numpy as np
from copy import deepcopy
from config.dic import (
    DIC_config, DIC_Image, Img_Ref_Pt_Pos, 
    Stereo_DIC_Init_Param, Subset_Info, 
    Img_Grad_Info, Coarse_Search_Method, PSO_Config
)

class DIC_search_pt_type(IntEnum):
    initial         = 0
    normal          = 1


# def get_subset(img, pt_x, pt_y, subset_side_len):
    
#     return


