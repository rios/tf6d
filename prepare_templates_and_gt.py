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
import trimesh
import io

if __name__ == "__main__":
    dataset_name = 'lmo'
    parser = argparse.ArgumentParser(description='Test pose estimation inference on test set')
    parser.add_argument('--config_file',
                        default="./database/zs6d_configs/template_gt_preparation_configs/cfg_template_gt_generation_%s.json" % (
                            dataset_name))

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

            # mesh = trimesh.load(input_model_path)
            # # Convert the mesh to PLY format in-memory
            # ply_data = io.BytesIO()
            # mesh.export(ply_data, file_type='ply')
            #
            # # Reset the stream position to the beginning
            # ply_data.seek(0)

            x_abs, y_abs, z_abs, x_ct, y_ct, z_ct = convert_unique(input_model_path, output_model_path)
            # x_abs, y_abs, z_abs, x_ct, y_ct, z_ct = convert_unique(ply_data, output_model_path)

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
    # extractor = PoseViTExtractor(model_type='dino_vits8', stride=8, device=device)
    extractor = PoseViTExtractor(model_type='dinov2_vits14', stride=14, device=device)

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
                if dataset_name == 'lmoWonder3d':
                    model_path = os.path.join(config['path_output_models_xyz'], f"obj_{int(obj_id):06d}.obj")
                else:
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
                    rgba_image = cv2.imread(os.path.join(path_template_folder, file), cv2.IMREAD_UNCHANGED)

                    # Check if the image has an alpha channel
                    if dataset_name in ['tless', 'itodd']:
                        # Extract RGB and Alpha channels
                        rgb = rgba_image[:, :, :3]
                        alpha = rgba_image[:, :, 3] / 255.0  # Normalize alpha to [0, 1]

                        # Define a black background for blending
                        black_background = np.zeros_like(rgb, dtype=np.uint8)

                        # Blend the RGB image with the black background using the alpha channel
                        rgb_image = (rgb * alpha[:, :, np.newaxis] + black_background * (
                                1 - alpha[:, :, np.newaxis])).astype(np.uint8)

                        # Create a binary mask from the alpha channel
                        mask = cv2.threshold((alpha * 255).astype(np.uint8), 0, 255, cv2.THRESH_BINARY)[1]
                    else:
                        # Split into channels
                        b, g, r, a = cv2.split(rgba_image)
                        # Combine the first three channels to form the RGB image
                        rgb_image = cv2.merge([r, g, b])
                        # The alpha channel is your mask
                        mask = a

                    x, y, w, h = cv2.boundingRect(mask)
                    crop_size = max(w, h)
                    img_crop, crop_x, crop_y = img_utils.make_quadratic_crop_ratio(rgb_image, [x, y, w, h],
                                                                                   final_ratio=1.0)
                    img_crop = Image.fromarray(img_crop)
                    img_prep, _, _ = extractor.preprocess(img_crop, load_size=(224, 224))

                    mask_crop, _, _ = img_utils.make_quadratic_crop_ratio(mask, [x, y, w, h], final_ratio=1.0)
                    mask_crop = Image.fromarray(mask_crop)
                    _, _, _ = extractor.preprocess_mask(mask_crop, load_size=(224, 224))

                    desc = extractor.extract_descriptors(img_prep.to(device), layer=9, facet='token', bin=False,
                                                         include_cls=True)
                    desc = desc.squeeze(0).squeeze(0).detach().cpu().numpy()

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
