import argparse
import json
import os
import torch
import numpy as np
from tqdm import tqdm
from PIL import Image
import cv2
import csv
import logging
import time
import matplotlib.pyplot as plt
import sys

from src.pose_extractor import PoseViTExtractor
from pose_utils.data_utils import ImageContainer_all
import pose_utils.img_utils as img_utils
import pose_utils.utils as utils
import pose_utils.vis_utils as vis_utils
import pose_utils.eval_utils as eval_utils
from src.correspondences import find_correspondences, _log_bin
from torch.nn import functional as nn_F

# Adding paths for model imports
sys.path.append('../mast3r/')
from mast3r.fast_nn import fast_reciprocal_NNs

# Setup logging configuration
logging.basicConfig(level=logging.INFO,
                    filename="pose_estimation_new.log",
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def show_correspondences(image1: Image.Image, image2: Image.Image, points1: list, points2: list):
    """Display correspondences between two images using matplotlib."""
    img1 = np.array(image1)
    img2 = np.array(image2)

    # Convert points to tuples and swap positions
    # points1 = [[y, x] for x, y in points1]
    # points2 = [[y, x] for x, y in points2]

    plt.figure(figsize=(15, 10))
    concatenated_image = np.concatenate((img1, img2), axis=1)
    plt.imshow(concatenated_image)

    offset = img1.shape[1]
    for (x1, y1), (x2, y2) in zip(points1, points2):
        plt.scatter(x1, y1, color='red', marker='o')
        plt.scatter(x2 + offset, y2, color='blue', marker='o')
        plt.plot([x1, x2 + offset], [y1, y2], color='green', linewidth=1)

    plt.axis('off')
    plt.show()
    plt.savefig('demo.png')
    plt.close()


def load_config(config_file):
    """Load configuration from a JSON file."""
    with open(config_file, 'r') as f:
        return json.load(f)


def load_ground_truth_data(config):
    """Load ground truth data from specified paths in the config."""
    templates_gt = load_config(config['templates_gt_path'])
    data_gt = load_config(config['gt_path'])
    norm_factors = load_config(config['norm_factor_path'])
    return templates_gt, data_gt, norm_factors


def create_results_directory(result_path):
    """Create a directory for results if it doesn't exist."""
    if not os.path.exists(result_path):
        os.makedirs(result_path)


def prepare_csv_file(csv_file):
    """Create a new CSV file and write the headers."""
    headers = ['scene_id', 'im_id', 'obj_id', 'score', 'R', 't', 'time']
    with open(csv_file, mode='w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(headers)


def process_templates(templates_gt, config):
    """Load templates into memory for later processing."""
    templates_desc = {}
    templates_gt_new = {}

    for obj_id, template_labels in tqdm(templates_gt.items()):
        try:
            templates_desc[obj_id] = torch.cat(
                [torch.from_numpy(np.load(template_label['img_desc'])).unsqueeze(0)
                 for i, template_label in enumerate(template_labels) if
                 i % config['template_subset'] == 0], dim=0)

            templates_gt_new[obj_id] = [template_label for i, template_label in enumerate(template_labels) if
                                        i % config['template_subset'] == 0]
        except Exception as e:
            logger.error(f"Error processing templates for object {obj_id}: {e}")

    return templates_desc, templates_gt_new


def process_image(data_gt, templates_desc, templates_gt_new, config, csv_file, debug_img_path, device, extractor,
                  feature_extractor):
    """Process input images and estimate poses."""
    for all_id, img_labels in tqdm(data_gt.items()):
        scene_id, img_id = all_id.split("_")[0], all_id.split("_")[-1]

        img_path = os.path.join(config['dataset_path'], img_labels[0]['img_name'].split("./")[-1])
        img_name = img_path.split("/")[-1].split(".png")[0]

        # Load image and camera parameters
        img = Image.open(img_path)
        cam_K = np.array(img_labels[0]['cam_K']).reshape((3, 3))

        img_data = ImageContainer_all(img=img,
                                      img_name=img_name,
                                      scene_id=scene_id,
                                      cam_K=cam_K,
                                      crops=[],
                                      ori_crops=[],
                                      mask_crops=[],
                                      descs=[],
                                      x_offsets=[],
                                      y_offsets=[],
                                      obj_names=[],
                                      obj_ids=[],
                                      model_infos=[],
                                      t_gts=[],
                                      R_gts=[],
                                      masks=[])

        for obj_index, img_label in enumerate(img_labels):
            bbox_gt = img_label[config['bbox_type']]
            if bbox_gt[2] == 0 or bbox_gt[3] == 0 or bbox_gt == [-1, -1, -1, -1]:
                continue

            img_data.t_gts.append(np.array(img_label['cam_t_m2c']) * config['scale_factor'])
            img_data.R_gts.append(np.array(img_label['cam_R_m2c']).reshape((3, 3)))
            img_data.obj_ids.append(str(img_label['obj_id']))
            img_data.model_infos.append(img_label['model_info'])

            try:
                mask = img_utils.rle_to_mask(img_label['mask_sam']).astype(np.uint8)
                mask_3_channel = np.stack([mask] * 3, axis=-1)

                bbox = img_utils.get_bounding_box_from_mask(mask)
                img_crop, y_offset, x_offset = img_utils.make_quadratic_crop(np.array(img), bbox)

                mask_crop, _, _ = img_utils.make_quadratic_crop(mask, bbox)
                img_data.ori_crops.append(Image.fromarray(img_crop))
                img_crop = cv2.bitwise_and(img_crop, img_crop, mask=mask_crop)

                img_data.crops.append(Image.fromarray(img_crop))
                img_data.mask_crops.append(mask_crop)

                img_prep, img_crop, _ = extractor.preprocess(Image.fromarray(img_crop), load_size=(224, 224))

                with torch.no_grad():
                    desc = extractor.extract_descriptors(img_prep.to(device), layer=9, facet='token', bin=False,
                                                         include_cls=True)
                    img_data.descs.append(desc.squeeze(0).squeeze(0).detach().cpu())

                img_data.y_offsets.append(y_offset)
                img_data.x_offsets.append(x_offset)
                img_data.masks.append(mask_3_channel)

            except Exception as e:
                logger.warning(
                    f"Loading mask and extracting descriptor failed for img {img_path} and object_id {obj_index}: {e}")
                img_data.crops.append(None)
                img_data.descs.append(None)
                img_data.y_offsets.append(None)
                img_data.x_offsets.append(None)
                img_data.masks.append(None)

        # Process each crop and estimate poses
        for i in range(len(img_data.crops)):
            start_time = time.time()
            object_id = img_data.obj_ids[i]
            if object_id not in ['3']:
                continue

            if img_data.crops[i] is not None:
                # try:
                #     matched_templates = utils.find_template_cpu(img_data.descs[i],
                #                                                 templates_desc[object_id],
                #                                                 num_results=config['num_matched_templates'],
                #                                                 use_cls=False)
                # except Exception as e:
                #     logger.error(
                #         f"Template matching failed for {img_data.img_name} and object_id {img_data.obj_ids[i]}: {e}")
                #
                min_err = np.inf
                pose_est = False

                # for matched_template in matched_templates:
                #     template = Image.open(templates_gt_new[object_id][matched_template[1]]['img_crop'])

                matched_templates = templates_gt_new[object_id]
                # Iterate through matched templates for pose estimation
                for matched_template in matched_templates:
                    template = Image.open(matched_template['img_crop'])

                    test_temp = Image.open(matched_templates[0]['img_crop'])

                    try:
                        with torch.no_grad():
                            load_size = (test_temp.size[1], test_temp.size[0])

                            img_prep, img_crop, _ = extractor.preprocess(test_temp, load_size=load_size)

                            template_prep, template_crop, _ = extractor.preprocess(template, load_size=load_size)

                            desc1 = feature_extractor(img_prep.to(device))
                            desc2 = feature_extractor(template_prep.to(device))

                            # # Upsampling
                            # scale_factor = feature_extractor.patch_size
                            scale_factor = 1

                            w = int(test_temp.size[0] / scale_factor)
                            h = int(test_temp.size[1] / scale_factor)

                            desc1 = nn_F.interpolate(desc1, size=(h, w), mode="bicubic")
                            desc2 = nn_F.interpolate(desc2, size=(h, w), mode="bicubic")

                            desc1 = desc1.squeeze(0).permute(1, 2, 0)
                            desc2 = desc2.squeeze(0).permute(1, 2, 0)

                            mask1, _, _ = extractor.preprocess_mask(Image.open(matched_templates[0]['mask_crop']),
                                                                    load_size=(h, w))
                            mask2, _, _ = extractor.preprocess_mask(
                                Image.open(matched_template['mask_crop']),
                                load_size=(h, w))

                            # Mask out features
                            desc1 = desc1 * mask1.permute(1, 2, 0).to(device)
                            desc2 = desc2 * mask2.permute(1, 2, 0).to(device)

                            # Find 2D-2D matches between the two images
                            matches_im0, matches_im1 = fast_reciprocal_NNs(desc1, desc2, subsample_or_initxy1=8,
                                                                           device=device, dist='dot',
                                                                           block_size=2 ** 13)

                            # Ignore small border around the edge
                            valid_matches = (matches_im0[:, 0] >= 3) & (matches_im0[:, 0] < w - 3) & \
                                            (matches_im0[:, 1] >= 3) & (matches_im0[:, 1] < h - 3) & \
                                            (matches_im1[:, 0] >= 3) & (matches_im1[:, 0] < w - 3) & \
                                            (matches_im1[:, 1] >= 3) & (matches_im1[:, 1] < h - 3)

                            matches_im0, matches_im1 = matches_im0[valid_matches], matches_im1[valid_matches]

                            # Scale back the points
                            points1 = np.round(matches_im0 / np.array(
                                [w / test_temp.size[0], h / test_temp.size[1]])).astype(int)
                            points2 = np.round(matches_im1 / np.array(
                                [w / test_temp.size[0], h / test_temp.size[1]])).astype(int)

                            img_crop = Image.eval(img_crop, lambda p: 255 if p == 0 else p)
                            template_crop = Image.eval(template_crop, lambda p: 255 if p == 0 else p)

                            show_correspondences(img_crop, template_crop, points1, points2)
                            # # Converting points to tuples and swapping positions
                            points1 = [[y, x] for x, y in points1]
                            points2 = [[y, x] for x, y in points2]

                            # Upsampling
                            # scale_factor = feature_extractor.patch_size
                            # w = int(img_data.crops[i].size[0] / scale_factor)
                            # h = int(img_data.crops[i].size[1] / scale_factor)
                            #
                            # desc1 = nn_F.interpolate(desc1, size=(h, w), mode="bicubic")
                            # desc2 = nn_F.interpolate(desc2, size=(h, w), mode="bicubic")
                            #
                            # mask1, _, _ = extractor.preprocess_mask(Image.fromarray(img_data.mask_crops[i]),
                            #                                         load_size=(h, w))
                            # mask2, _, _ = extractor.preprocess_mask(
                            #     Image.open(templates_gt_new[object_id][matched_template[1]]['mask_crop']),
                            #     load_size=(h, w))
                            #
                            # # Mask out features
                            # desc1 = desc1 * mask1.to(device)
                            # desc2 = desc2 * mask2.to(device)

                            # num_pathes = desc1.shape[-2:]
                            # desc1 = desc1.permute(0, 2, 3, 1).flatten(start_dim=-3, end_dim=-2).unsqueeze(1)
                            # # Bx1xtx(dxh)
                            # # binning
                            # desc1 = _log_bin(desc1, hierarchy=2, device=device)
                            #
                            # desc2 = desc2.permute(0, 2, 3, 1).flatten(start_dim=-3, end_dim=-2).unsqueeze(1)
                            # # binning
                            # desc2 = _log_bin(desc2, hierarchy=2, device=device)
                            #
                            # points1, points2 = find_correspondences(
                            #     desc1, desc2,
                            #     num_patches=num_pathes,
                            #     num_pairs=200,
                            #     use_kmeans=True,
                            #     patch_size=feature_extractor.patch_size,
                            #     # patch_size=scale_factor,
                            #     use_best_buddies=True
                            # )
                            #
                            # show_correspondences(img_crop, template_crop, points1, points2)

                    except Exception as e:
                        logger.error(
                            f"Local correspondence matching failed for {img_data.img_name} and object_id {img_data.obj_ids[i]}: {e}")

                    try:
                        img_uv = np.load(matched_template['uv_crop']).astype(np.uint8)
                        load_size = (img_data.crops[i].size[1], img_data.crops[i].size[0])  # h, w for torch
                        _, img_uv, _ = extractor.preprocess(Image.fromarray(img_uv), load_size=load_size)

                        R_est, t_est, quality = utils.get_pose_from_correspondences(points1,
                                                                           points2,
                                                                           img_data.y_offsets[i],
                                                                           img_data.x_offsets[i],
                                                                           np.array(img_uv),
                                                                           img_data.cam_K,
                                                                           norm_factors[str(img_data.obj_ids[i])],
                                                                           config['scale_factor'])


                    except Exception as e:
                        logger.error(
                            f"Not enough correspondences could be extracted for {img_data.img_name} and object_id {object_id}: {e}")
                        R_est, t_est, quality = None, None, None

                    if R_est is None:
                        R_est = np.array(matched_template['cam_R_m2c']).reshape((3, 3))
                        t_est = np.array([0., 0., 0.])
                        quality = -np.inf

                    end_time = time.time()
                    # err, acc = eval_utils.calculate_score(R_est, img_data.R_gts[i], int(img_data.obj_ids[i]), 0)
                    err = -quality

                    if err < min_err:
                        min_err = err
                        R_best = R_est
                        t_best = t_est
                        pose_est = True

                if not pose_est:
                    R_best = np.eye(3)  # Identity matrix
                    t_best = np.zeros(3)
                    logger.warning(f"No pose could be determined for {img_data.img_name} and object_id {object_id}")
                    score = 0.0
                else:
                    score = 0.0

            else:
                R_best = np.eye(3)
                t_best = np.zeros(3)
                logger.warning(
                    f"No Pose could be determined for {img_data.img_name} and object_id {object_id} because no object crop available")
                score = 0.0

            # Prepare for writing results
            R_best_str = " ".join(map(str, R_best.flatten()))
            t_best_str = " ".join(map(str, t_best * 1000))
            elapsed_time = end_time - start_time

            # Write the detections to the CSV file
            check = np.eye(3)
            if not np.allclose(R_best, check):
                with open(csv_file, mode='a', newline='') as csvfile:
                    csv_writer = csv.writer(csvfile)
                    csv_writer.writerow(
                        [img_data.scene_id, img_data.img_name, object_id, score, R_best_str, t_best_str, elapsed_time])

            # Create and save debug images if enabled
            if config['debug_imgs'] and i % config['debug_imgs'] == 0:
                dbg_img = vis_utils.create_debug_image(R_best, t_best, img_data.R_gts[i], img_data.t_gts[i],
                                                       np.asarray(img_data.img),
                                                       img_data.cam_K,
                                                       img_data.model_infos[i],
                                                       config['scale_factor'],
                                                       image_shape=(config['image_resolution'][0],
                                                                    config['image_resolution'][1]),
                                                       colEst=(0, 255, 0))
                dbg_img = cv2.cvtColor(dbg_img, cv2.COLOR_BGR2RGB)
                dbg_img_mask = cv2.hconcat([dbg_img, img_data.masks[i]]) if img_data.masks[i] is not None else dbg_img
                cv2.imwrite(os.path.join(debug_img_path, f"{img_data.img_name}_{img_data.obj_ids[i]}.png"),
                            dbg_img_mask)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test pose estimation inference on test set')
    parser.add_argument('--config_file', default="./database/zs6d_configs/bop_eval_configs/cfg_ycbv_inference_bop_zero.json")
    args = parser.parse_args()

    config = load_config(args.config_file)
    templates_gt, data_gt, norm_factors = load_ground_truth_data(config)

    result_path = './logs/results_fastsam'
    csv_file = os.path.join(result_path, config['results_file'])
    create_results_directory(result_path)
    prepare_csv_file(csv_file)

    if config['debug_imgs']:
        debug_img_path = os.path.join("./logs/debug_imgs_fastsam_pandaOct", config['results_file'].split(".csv")[0])
        create_results_directory(debug_img_path)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    extractor = PoseViTExtractor(model_type='dinov2_vits14', stride=14, device=device)

    """Initialize and return the feature extractor."""
    # index = [4, 11, 17, 23]
    from models.dino import DINO
    feature_extractor = DINO(dino_name='dinov2', model_name='vits14', output='dense', layer=9)
    feature_extractor = feature_extractor.to(device)
    #
    # from models.stablediffusion import DIFT
    # feature_extractor = DIFT(time_step=1, layer=1, output="dense")
    # feature_extractor = feature_extractor.to(device)

    # from models.sam import SAM
    # feature_extractor = SAM(arch='vit_l', layer=9, output="dense")
    # feature_extractor = feature_extractor.to(device)

    # from models.convnext import ConvNext
    # feature_extractor = ConvNext(arch='convnext_base_w', checkpoint='laion2b_s13b_b82k', layer=3, output="dense")
    # feature_extractor = feature_extractor.to(device)

    # extractor = PoseViTExtractor(model_type='dino_vits8', stride=8, device=device)
    # extractor_point = PoseViTExtractor(model_type='dino_vits8', stride=8, device=device)

    # Process templates
    templates_desc, templates_gt_new = process_templates(templates_gt, config)

    # Process each image within the process_image function
    process_image(data_gt, templates_desc, templates_gt_new, config, csv_file, debug_img_path, device, extractor,
                  feature_extractor)
