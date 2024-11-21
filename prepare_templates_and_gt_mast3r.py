import argparse
import os
import json
import numpy as np
import torch
from src.pose_extractor import PoseViTExtractor
from src.ply_file_to_3d_coord_model import convert_unique
from rendering.renderer_xyz import Renderer
from rendering.model import Model3D
from tqdm import tqdm
import cv2
from PIL import Image
from pose_utils import img_utils
from rendering.utils import get_rendering, get_sympose
import sys

# Append paths for dependencies
sys.path.append('../mast3r/')
sys.path.append('.')

from mast3r.model import AsymmetricMASt3R
from mast3r.fast_nn import fast_reciprocal_NNs
from dust3r.inference import inference
from dust3r.utils.image import load_images_crop

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test pose estimation inference on test set')
    parser.add_argument('--config_file',
                        default="./database/zs6d_configs/template_gt_preparation_configs/cfg_template_gt_generation_lmo.json")

    args = parser.parse_args()

    with open(os.path.join(args.config_file), 'r') as f:
        config = json.load(f)

    with open(os.path.join(config['path_models_info_json']), 'r') as f:
        models_info = json.load(f)

    obj_poses = np.load(config['path_template_poses'])

    # Creating the output folder for the cropped templates and descriptors
    if not os.path.exists(config['path_output_templates_and_descs_folder']):
        os.makedirs(config['path_output_templates_and_descs_folder'])

    # Creating the models_xyz folder
    if not os.path.exists(config['path_output_models_xyz']):
        os.makedirs(config['path_output_models_xyz'])

    # Preparing the object models in xyz format:
    print("Loading and preparing the object meshes:")
    norm_factors = {}
    for obj_model_name in tqdm(os.listdir(config['path_object_models_folder'])):
        if obj_model_name.endswith(".ply"):
            obj_id = int(obj_model_name.split("_")[-1].split(".ply")[0])
            input_model_path = os.path.join(config['path_object_models_folder'], obj_model_name)
            output_model_path = os.path.join(config['path_output_models_xyz'], obj_model_name)
            # if not os.path.exists(output_model_path):
            x_abs, y_abs, z_abs, x_ct, y_ct, z_ct = convert_unique(input_model_path, output_model_path)

            norm_factors[obj_id] = {'x_scale': float(x_abs),
                                    'y_scale': float(y_abs),
                                    'z_scale': float(z_abs),
                                    'x_ct': float(x_ct),
                                    'y_ct': float(y_ct),
                                    'z_ct': float(z_ct)}

    with open(os.path.join(config['path_output_models_xyz'], "norm_factor.json"), "w") as f:
        json.dump(norm_factors, f)

    # Load feature extractor
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    matcher = AsymmetricMASt3R.from_pretrained("naver/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric").to(device)
    print("Loading Matcher is done!")

    cam_K = np.array(config['cam_K']).reshape((3, 3))
    #  Vispy’s gloo. Texture2D, the shape parameter follows the (height, width) convention
    #  this function requires the input to be (w, h)
    ren = Renderer((config['template_resolution'][1], config['template_resolution'][0]), cam_K)
    template_labels_gt = dict()

    with torch.no_grad():
        for template_name in tqdm(os.listdir(config['path_templates_folder'])):
            path_template_folder = os.path.join(config['path_templates_folder'], template_name)

            if os.path.isdir(
                    path_template_folder) and template_name != "models" and template_name != "models_proc" and template_name != "object_poses":
                path_to_template_desc = os.path.join(config['path_output_templates_and_descs_folder'],
                                                     template_name)

                if not os.path.exists(path_to_template_desc):
                    os.makedirs(path_to_template_desc)

                # obj_id = template_name.split("_")[-1]
                obj_id = template_name.lstrip('0')

                model_info = models_info[str(obj_id)]

                obj_model = Model3D()
                model_path = os.path.join(config['path_output_models_xyz'], f"obj_{int(obj_id):06d}.ply")

                # Some objects are scaled inconsistently within the dataset, these exceptions are handled here:
                obj_scale = config['obj_models_scale']
                obj_model.load(model_path, scale=obj_scale)

                files = os.listdir(path_template_folder)
                filtered_files = list(filter(lambda x: not x.endswith('_depth.png'), files))
                filtered_files = sorted(filtered_files)
                tmp_list = []

                for i, file in enumerate(filtered_files):
                    # Load the image with alpha channel
                    image = cv2.imread(os.path.join(path_template_folder, file), cv2.IMREAD_UNCHANGED)

                    # Check if image has four channels
                    if image.shape[2] == 4:
                        # Split into channels
                        b, g, r, a = cv2.split(image)
                        # Combine the first three channels to form the RGB image
                        rgb_image = cv2.merge([r, g, b])
                        # The alpha channel is your mask
                        mask = a

                    x, y, w, h = cv2.boundingRect(mask)
                    crop_size = max(w, h)
                    img_crop, crop_x, crop_y = img_utils.make_quadratic_crop_ratio(rgb_image, [x, y, w, h],
                                                                                   patch_size=16, final_ratio=1.0)
                    img_crop = Image.fromarray(img_crop)

                    mask_crop, _, _ = img_utils.make_quadratic_crop_ratio(mask, [x, y, w, h], patch_size=16,
                                                                          final_ratio=1.0)
                    mask_crop = Image.fromarray(mask_crop)

                    load_size = 224  # h, w for torch

                    # Preprocess images and masks
                    image1_batch, image1_pil, _ = img_utils.preprocess_norm(img_crop,
                                                                            model_type='mast3r',
                                                                            load_size=load_size)
                    # image2_batch, image2_pil, _ = img_utils.preprocess_norm(img_crop, model_type='mast3r',
                    #                                                         load_size=load_size)
                    #
                    # images = load_images_crop([image1_pil, image2_pil], (h_0, w_0), verbose=False)

                    # # Load masks
                    # mask1, _, _ = img_utils.preprocess(mask_crop,
                    #                                    load_size=(images[0]['true_shape'][0][0],
                    #                                               images[0]['true_shape'][0][1]))
                    #
                    # mask2, _, _ = img_utils.preprocess(mask_crop,
                    #                                    load_size=(images[0]['true_shape'][0][0],
                    #                                               images[0]['true_shape'][0][1]))

                    # Perform inference
                    B = image1_batch.shape[0]
                    # Recover true_shape when available, otherwise assume that the img shape is the true one
                    shape1 = torch.tensor(image1_batch.shape[-2:])[None].repeat(B, 1)

                    desc, pos, _ = matcher._encode_image(image1_batch.to(device), shape1.to(device))

                    # Process descriptors
                    desc = desc.squeeze(0).detach().cpu().numpy()

                    R = obj_poses[i][:3, :3]
                    t = obj_poses[i].T[-1, :3]
                    sym_continues = [0, 0, 0, 0, 0, 0]
                    keys = model_info.keys()

                    if ('symmetries_continuous' in keys):
                        sym_continues[:3] = model_info['symmetries_continuous'][0]['axis']
                        sym_continues[3:] = model_info['symmetries_continuous'][0]['offset']

                    rot_pose, rotation_lock = get_sympose(R, sym_continues)

                    img_uv, depth_rend, bbox_template = get_rendering(obj_model, rot_pose, t / 1000., ren)
                    img_uv_ori = img_uv.astype(np.uint8)

                    img_uv, _, _ = img_utils.make_quadratic_crop_ratio(img_uv_ori, [x, y, w, h], final_ratio=1.0)
                    img_uv_crop = Image.fromarray(img_uv)

                    # Storing template information:
                    tmp_dict = {"img_id": str(i),
                                "img_name": os.path.join(os.path.join(path_template_folder, file)),
                                "mask_name": os.path.join(os.path.join(path_template_folder, f"mask_{file}")),
                                "obj_id": str(obj_id),
                                "bbox_obj": [x, y, w, h],
                                "cam_R_m2c": R.tolist(),
                                "cam_t_m2c": t.tolist(),
                                "model_path": os.path.join(config['path_object_models_folder'],
                                                           f"obj_{int(obj_id):06d}.ply"),
                                "model_info": models_info[str(obj_id)],
                                "cam_K": cam_K.tolist(),
                                # "img_crop": os.path.join(path_to_template_desc, f"{file[:-4]}.jpg"),
                                "img_crop": os.path.join(path_to_template_desc, file),
                                "mask_crop": os.path.join(path_to_template_desc, f"mask_{file}"),
                                "img_uv_crop": os.path.join(path_to_template_desc, f"uv_{file}"),
                                "img_uv": os.path.join(path_to_template_desc, f"uv_full_{file}"),
                                "img_desc": os.path.join(path_to_template_desc, f"{file.split('.')[0]}.npy"),
                                "uv_crop": os.path.join(path_to_template_desc, f"{file.split('.')[0]}_uv.npy"),
                                "uv": os.path.join(path_to_template_desc, f"{file.split('.')[0]}_uv_full.npy"),
                                }

                    tmp_list.append(tmp_dict)

                    # Saving all template crops and descriptors:
                    np.save(tmp_dict['uv_crop'], img_uv)
                    np.save(tmp_dict['uv'], img_uv_ori)
                    np.save(tmp_dict['img_desc'], desc)

                    img_crop.save(tmp_dict['img_crop'])
                    img_uv_crop.save(tmp_dict['img_uv_crop'])
                    Image.fromarray(img_uv_ori).save(tmp_dict['img_uv'])
                    mask_crop.save(tmp_dict['mask_crop'])

                template_labels_gt[str(obj_id)] = tmp_list

    with open(config['output_template_gt_file'], 'w') as f:
        json.dump(template_labels_gt, f)
