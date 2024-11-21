from dataclasses import dataclass
from PIL import Image
import numpy as np

def convert_array(s):
    # remove brackets and split on comma
    s = s.strip('[]').split(',')
    # convert each element to a float and return as a numpy array
    return np.array([float(x) for x in s])

@dataclass
class ImageContainer_all:
    img: Image.Image
    img_name: str
    scene_id: str
    cam_K: np.ndarray
    crops: list
    mask_crops: list
    ori_crops: list
    descs: list
    x_offsets: list
    y_offsets: list
    obj_names: list
    obj_ids: list
    model_infos: list
    t_gts: list
    R_gts: list
    masks: list

@dataclass
class ImageContainer_masks:
    img: Image.Image
    img_name: str
    scene_id: str
    cam_K: np.ndarray
    crops: list
    descs: list
    x_offsets: list
    y_offsets: list
    obj_names: list
    obj_ids: list
    model_infos: list
    t_gts: list
    R_gts: list
    masks: list

@dataclass
class ImageContainer_Mask:
    img: Image.Image
    masked_img: list
    img_name: str
    scene_id: str
    cam_K: np.ndarray
    crops: list
    mask_crops: list
    ori_crops: list
    descs: list
    x_offsets: list
    y_offsets: list
    obj_names: list
    obj_ids: list
    model_infos: list
    t_gts: list
    R_gts: list
    masks: list
    masks_cv2: list
    
@dataclass
class ImageContainer_multiple_masks:
    img: Image.Image
    img_name: str
    scene_id: str
    cam_K: np.ndarray
    crops: list
    descs: list
    x_offsets: list
    y_offsets: list
    obj_names: list
    obj_ids: list
    model_infos: list
    t_gts: list
    R_gts: list
    masks: list
    
    