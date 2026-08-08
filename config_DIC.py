from dataclasses import dataclass
from enum import Enum
import numpy as np

# a:b (b is a data type hint for a)
@dataclass
class DIC_Image:
    ref: np.ndarray
    cur: np.ndarray

@dataclass
class Img_Grad_Info:
    H_inv_mat: np.ndarray
    J_mat: np.ndarray

@dataclass
class Stereo_DIC_Init_Param:
    translate: int = 0

@dataclass
class PSO_Config:
    population: int = 40

@dataclass
class Img_Ref_Pt_Pos:
    pt_x: float
    pt_y: float

@dataclass
class Subset_Info:
    subset_data: np.ndarray = None
    subset_side_len: int = 31

class Coarse_Search_Method(Enum):
    PSO = 1
    GA  = 2
    SA  = 3

# summary
@dataclass
class DIC_config:
    coarse_method:          Coarse_Search_Method
    coarse_method_cfg:      PSO_Config
    dic_image:              DIC_Image
    img_ref_pt:             Img_Ref_Pt_Pos
    init_param:             Stereo_DIC_Init_Param
    subset_ref_info:        Subset_Info
    subset_cur_info:        Subset_Info
    img_grad:               Img_Grad_Info
    search_type:            int = 1