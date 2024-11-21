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

# Adding paths for model import
sys.path.append('../mast3r/')

from mast3r.model import AsymmetricMASt3R
from mast3r.fast_nn import fast_reciprocal_NNs

import mast3r.utils.path_to_dust3r
from dust3r.inference import inference
from dust3r.utils.image import load_images_crop, load_images
import matplotlib.pyplot as plt

# Setup logging configuration
logging.basicConfig(level=logging.INFO,
                    filename="pose_estimation_new.log",
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Replace <#threads> with the number of threads you want to use
num_threads = '32'

# Setting the environment variables
os.environ["OPENBLAS_NUM_THREADS"] = num_threads
os.environ["GOTO_NUM_THREADS"] = num_threads
os.environ["OMP_NUM_THREADS"] = num_threads
os.environ["MKL_NUM_THREADS"] = num_threads


def show_correspondences(image1: Image.Image, image2: Image.Image, points1: list, points2: list):
    """Display correspondences between two images using matplotlib."""
    img1 = np.array(image1)
    img2 = np.array(image2)

    plt.figure(figsize=(15, 10))
    concatenated_image = np.concatenate((img1, img2), axis=1)
    plt.imshow(concatenated_image)

    offset = img1.shape[1]
    for (x1, y1), (x2, y2) in zip(points1, points2):
        plt.scatter(x1, y1, color='red', marker='o')
        plt.scatter(x2 + offset, y2, color='blue', marker='o')
        plt.plot([x1, x2 + offset], [y1, y2], color='yellow', linewidth=1)

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


def process_image(data_gt, templates_desc, templates_gt_new, config, csv_file, debug_img_path, device, model):
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
                img_crop, y_offset, x_offset = img_utils.make_quadratic_crop(np.array(img), bbox, patch_size=16)

                mask_crop, _, _ = img_utils.make_quadratic_crop(mask, bbox, patch_size=16)
                img_data.ori_crops.append(Image.fromarray(img_crop))
                img_crop = cv2.bitwise_and(img_crop, img_crop, mask=mask_crop)

                img_data.crops.append(Image.fromarray(img_crop))
                img_data.mask_crops.append(mask_crop)

                img_prep, img_crop, _ = img_utils.preprocess_norm(Image.fromarray(img_crop), load_size=(224, 224))

                with torch.no_grad():
                    # Perform inference
                    B = img_prep.shape[0]
                    # Recover true_shape when available, otherwise assume that the img shape is the true one
                    shape1 = torch.tensor(img_prep.shape[-2:])[None].repeat(B, 1)
                    desc, pos, _ = model._encode_image(img_prep.to(device), shape1.to(device))
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
            # if object_id not in ['16']:
            #     continue

            if img_data.crops[i] is not None:
                try:
                    matched_templates = utils.find_template_cpu(img_data.descs[i],
                                                                templates_desc[object_id],
                                                                num_results=config['num_matched_templates'],
                                                                use_cls=True)
                except Exception as e:
                    logger.error(
                        f"Template matching failed for {img_data.img_name} and object_id {img_data.obj_ids[i]}: {e}")

                min_err = np.inf
                pose_est = False
                for matched_template in matched_templates:
                    template = Image.open(templates_gt_new[object_id][matched_template[1]]['img_crop'])

                    try:
                        with torch.no_grad():
                            load_size = (img_data.crops[i].size[1], img_data.crops[i].size[0])  # h, w for torch
                            image1_batch, image1_pil, _ = img_utils.preprocess_norm(img_data.crops[i],
                                                                                    model_type='mast3r',
                                                                                    load_size=load_size)
                            image2_batch, image2_pil, _ = img_utils.preprocess_norm(template, model_type='mast3r',
                                                                                    load_size=load_size)

                            images = load_images_crop([image1_pil, image2_pil], 224, verbose=False)

                            h = images[0]['true_shape'][0][0]
                            w = images[0]['true_shape'][0][1]

                            mask1, _, _ = img_utils.preprocess(Image.fromarray(img_data.mask_crops[i]),
                                                               load_size=(h, w))
                            mask2, _, _ = img_utils.preprocess(
                                Image.open(templates_gt_new[object_id][matched_template[1]]['mask_crop']),
                                load_size=(h, w))

                            output = inference([tuple(images)], model, device, batch_size=1, verbose=False)

                            # at this stage, you have the raw dust3r predictions
                            view1, pred1 = output['view1'], output['pred1']
                            view2, pred2 = output['view2'], output['pred2']

                            desc1, desc2 = pred1['desc'].squeeze(0).detach(), pred2['desc'].squeeze(0).detach()

                            # mask out features!!!
                            desc1 = desc1 * mask1.permute(1, 2, 0)
                            desc2 = desc2 * mask2.permute(1, 2, 0)

                            conf_list = [pred1['desc_conf'].squeeze(0).cpu().numpy(),
                                         pred2['desc_conf'].squeeze(0).cpu().numpy()]

                            # find 2D-2D matches between the two images
                            matches_im0, matches_im1 = fast_reciprocal_NNs(desc1, desc2, subsample_or_initxy1=2,
                                                                           device=device, dist='dot',
                                                                           block_size=2 ** 13)

                            # ignore small border around the edge
                            H0, W0 = view1['true_shape'][0]
                            valid_matches_im0 = (matches_im0[:, 0] >= 3) & (matches_im0[:, 0] < int(W0) - 3) & (
                                    matches_im0[:, 1] >= 3) & (matches_im0[:, 1] < int(H0) - 3)

                            H1, W1 = view2['true_shape'][0]
                            valid_matches_im1 = (matches_im1[:, 0] >= 3) & (matches_im1[:, 0] < int(W1) - 3) & (
                                    matches_im1[:, 1] >= 3) & (matches_im1[:, 1] < int(H1) - 3)

                            valid_matches = valid_matches_im0 & valid_matches_im1
                            matches_im0, matches_im1 = matches_im0[valid_matches], matches_im1[valid_matches]

                            matches_confs = np.minimum(
                                conf_list[0][matches_im0[:, 1], matches_im0[:, 0]],
                                conf_list[1][matches_im1[:, 1], matches_im1[:, 0]]
                            )

                            # Apply confidence threshold
                            # conf_thr = 0.05
                            # if len(matches_confs) > 0:
                            #     mask = matches_confs >= conf_thr
                            #     matches_im0 = matches_im0[mask]
                            #     matches_im1 = matches_im1[mask]
                            #     matches_confs = matches_confs[mask]

                            # num_matches = matches_im0.shape[0]
                            # n_viz = min(30, num_matches)
                            #
                            # match_idx_to_viz = np.round(np.linspace(0, num_matches - 1, n_viz)).astype(int)
                            # matches_im0, matches_im1 = matches_im0[match_idx_to_viz], matches_im1[match_idx_to_viz]

                            w_ori = img_data.crops[i].size[0]
                            h_ori = img_data.crops[i].size[1]

                            points1 = np.round(matches_im0 / np.array([W0 / w_ori, H0 / h_ori])).astype(int)
                            points2 = np.round(matches_im1 / np.array([W0 / w_ori, H0 / h_ori])).astype(int)

                            # show_correspondences(image1_pil, image2_pil, points1, points2)

                            # Convert points to tuples and swap positions
                            points1 = [[y, x] for x, y in points1]
                            points2 = [[y, x] for x, y in points2]

                    except Exception as e:
                        logger.error(
                            f"Local correspondence matching failed for {img_data.img_name} and object_id {img_data.obj_ids[i]}: {e}")

                    try:
                        img_uv = np.load(templates_gt_new[object_id][matched_template[1]]['uv_crop']).astype(np.uint8)
                        load_size = (img_data.crops[i].size[1], img_data.crops[i].size[0])  # h, w for torch
                        _, img_uv, _ = img_utils.preprocess(Image.fromarray(img_uv), load_size=load_size)

                        R_est, t_est, quality = utils.get_pose_from_correspondences(points1,
                                                                                    points2,
                                                                                    img_data.y_offsets[i],
                                                                                    img_data.x_offsets[i],
                                                                                    np.array(img_uv),
                                                                                    img_data.cam_K,
                                                                                    norm_factors[
                                                                                        str(img_data.obj_ids[i])],
                                                                                    config['scale_factor'])
                    except Exception as e:
                        logger.error(
                            f"Not enough correspondences could be extracted for {img_data.img_name} and object_id {object_id}: {e}")
                        R_est = None
                        t_est = None
                        quality = None

                    if R_est is None:
                        R_est = np.array(templates_gt_new[object_id][matched_template[1]]['cam_R_m2c']).reshape((3, 3))
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
    parser.add_argument('--config_file',
                        default="./database/zs6d_configs/bop_eval_configs/cfg_lmo_inference_bop_zero.json")
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
    """Initialize and return the feature extractor."""
    ###########
    model_name = "naver/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric"
    # you can put the path to a local checkpoint in model_name if needed
    model = AsymmetricMASt3R.from_pretrained(model_name).to(device)
    ##########

    # Process templates
    templates_desc, templates_gt_new = process_templates(templates_gt, config)

    # Process each image within the process_image function
    process_image(data_gt, templates_desc, templates_gt_new, config, csv_file, debug_img_path, device, model)
