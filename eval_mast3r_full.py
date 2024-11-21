import argparse
import csv
import json
import logging
import os
import time
import sys
import cv2
import numpy as np
from PIL import Image
import torch
from pyglet import image
from tqdm import tqdm
from pose_utils.data_utils import  ImageContainer_Mask
import pose_utils.img_utils as img_utils
import pose_utils.utils as utils
import pose_utils.vis_utils as vis_utils
import pose_utils.eval_utils as eval_utils
import matplotlib.pyplot as plt

# Append paths for dependencies
sys.path.append('../mast3r/')
sys.path.append('.')

from mast3r.model import AsymmetricMASt3R
from mast3r.fast_nn import fast_reciprocal_NNs
from dust3r.inference import inference
from dust3r.utils.image import load_images_crop

# Setup logging configuration
logging.basicConfig(level=logging.INFO,
                    filename="pose_estimation_matching.log",
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


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


def load_json_file(file_path):
    """Load a JSON file and return its content."""
    with open(file_path, 'r') as f:
        return json.load(f)


def prepare_csv_file(csv_file_path, headers):
    """Create a CSV file and write headers to it."""
    with open(csv_file_path, mode='w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(headers)


def create_directory_if_not_exists(directory):
    """Create a directory if it does not exist."""
    if not os.path.exists(directory):
        os.makedirs(directory)


def process_templates(templates_gt, config):
    """Process template images and descriptors."""
    templates_desc = {}
    templates_gt_new = {}

    for obj_id, template_labels in tqdm(templates_gt.items()):
        try:
            # Load and concatenate descriptors for the current object
            templates_desc[obj_id] = torch.cat([
                torch.from_numpy(np.load(template_label['img_desc'])).unsqueeze(0)
                for i, template_label in enumerate(template_labels) if i % config['template_subset'] == 0
            ], dim=0)

            # Store filtered template labels
            templates_gt_new[obj_id] = [
                template_label for i, template_label in enumerate(template_labels) if i % config['template_subset'] == 0
            ]
        except Exception as e:
            print(f"Error processing templates for object {obj_id}: {e}")

    print("Preparing templates finished!")
    return templates_desc, templates_gt_new


def process_images(data_gt, templates_gt_new, config, matcher, norm_factors, csv_file):
    """Process input images and estimate poses."""
    for all_id, img_labels in tqdm(data_gt.items()):
        scene_id, img_id = all_id.split("_")[0], all_id.split("_")[-1]

        img_path = os.path.join(config['dataset_path'], img_labels[0]['img_name'].split("./")[-1])
        img_name = img_path.split("/")[-1].split(".png")[0]

        # Load image and camera parameters
        img = Image.open(img_path)
        cam_K = np.array(img_labels[0]['cam_K']).reshape((3, 3))

        img_data = ImageContainer_Mask(img=img,
                                      masked_img=[],
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
                                      masks=[],
                                      masks_cv2=[]
                                      )

        # Process each object in the image
        for obj_index, img_label in enumerate(img_labels):
            bbox_gt = img_label[config['bbox_type']]
            if bbox_gt[2] == 0 or bbox_gt[3] == 0 or bbox_gt == [-1, -1, -1, -1]:
                continue

            # Load ground truth pose and object information
            img_data.t_gts.append(np.array(img_label['cam_t_m2c']) * config['scale_factor'])
            img_data.R_gts.append(np.array(img_label['cam_R_m2c']).reshape((3, 3)))
            img_data.obj_ids.append(str(img_label['obj_id']))
            img_data.model_infos.append(img_label['model_info'])

            try:
                # Convert RLE mask to binary mask
                mask = img_utils.rle_to_mask(img_label['mask_sam']).astype(np.uint8)
                mask_3_channel = np.stack([mask] * 3, axis=-1)

                # Create bounding box and crop the image
                bbox = img_utils.get_bounding_box_from_mask(mask)
                img_crop, y_offset, x_offset = img_utils.make_quadratic_crop(np.array(img), bbox)
                mask_crop, _, _ = img_utils.make_quadratic_crop(mask, bbox)

                img_data.ori_crops.append(Image.fromarray(img_crop))
                img_crop = cv2.bitwise_and(img_crop, img_crop, mask=mask_crop)
                img_data.crops.append(Image.fromarray(img_crop))
                img_data.mask_crops.append(mask_crop)

                img_data.y_offsets.append(y_offset)
                img_data.x_offsets.append(x_offset)
                img_data.masks.append(mask)
                img_data.masks_cv2.append(mask_3_channel)

                # masked image
                masked_img = cv2.bitwise_and(np.array(img), np.array(img), mask=mask)
                img_data.masked_img.append(Image.fromarray(masked_img))

            except Exception as e:
                logger.warning(
                    f"Loading mask and extracting descriptor failed for img {img_path} and object_id {obj_index}: {e}")
                img_data.crops.append(None)
                img_data.y_offsets.append(None)
                img_data.x_offsets.append(None)
                img_data.masks.append(None)

        # Estimate pose for each crop
        for i in range(len(img_data.masked_img)):
            start_time = time.time()
            object_id = img_data.obj_ids[i]

            if img_data.masked_img[i] is not None:
                min_err = np.inf
                pose_est = False
                matched_templates = templates_gt_new[object_id]

                # Iterate through matched templates for pose estimation
                for matched_template in matched_templates:
                    # template = Image.open(matched_template['img_crop'])
                    template_image = Image.open(matched_template['img_name'])
                    template = template_image.convert('RGB')

                    # Create a template background
                    template_mask = template_image.getchannel("A")

                    try:
                        with torch.no_grad():
                            h, w = img_data.masked_img[i].size[1], img_data.masked_img[i].size[0]
                            load_size = (h, w)  # h, w for torch

                            # Preprocess images and masks
                            image1_batch, image1_pil, _ = img_utils.preprocess_norm(img_data.masked_img[i],
                                                                                    model_type='mast3r',
                                                                                    load_size=load_size)
                            image2_batch, image2_pil, _ = img_utils.preprocess_norm(template, model_type='mast3r',
                                                                                    load_size=load_size)

                            images = load_images_crop([image1_pil, image2_pil], 640, verbose=False)

                            # Load masks
                            mask1, _, _ = img_utils.preprocess(Image.fromarray(img_data.masks[i]),
                                                               load_size=(images[0]['true_shape'][0][0],
                                                                          images[0]['true_shape'][0][1]))

                            mask2, _, _ = img_utils.preprocess(template_mask,
                                                               load_size=(images[0]['true_shape'][0][0],
                                                                          images[0]['true_shape'][0][1]))

                            # Perform inference
                            output = inference([tuple(images)], matcher, device, batch_size=1, verbose=False)
                            view1, pred1 = output['view1'], output['pred1']
                            view2, pred2 = output['view2'], output['pred2']

                            # Process descriptors
                            desc1 = pred1['desc'].squeeze(0).detach() * mask1.permute(1, 2, 0)
                            desc2 = pred2['desc'].squeeze(0).detach() * mask2.permute(1, 2, 0)
                            conf_list = [pred1['desc_conf'].squeeze(0).cpu().numpy(),
                                         pred2['desc_conf'].squeeze(0).cpu().numpy()]

                            # Find 2D-2D matches
                            matches_im0, matches_im1 = fast_reciprocal_NNs(desc1, desc2, subsample_or_initxy1=8,
                                                                           device=device, dist='dot',
                                                                           block_size=2 ** 13)
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
                            conf_thr = 0.05
                            if len(matches_confs) > 0:
                                mask = matches_confs >= conf_thr
                                matches_im0 = matches_im0[mask]
                                matches_im1 = matches_im1[mask]
                                matches_confs = matches_confs[mask]

                            # num_matches = matches_im0.shape[0]
                            # n_viz = min(30, num_matches)
                            #
                            # match_idx_to_viz = np.round(np.linspace(0, num_matches - 1, n_viz)).astype(int)
                            # matches_im0, matches_im1 = matches_im0[match_idx_to_viz], matches_im1[match_idx_to_viz]

                            w_ori = img_data.masked_img[i].size[0]
                            h_ori = img_data.masked_img[i].size[1]

                            points1 = np.round(matches_im0 / np.array([W0 / w_ori, H0 / h_ori])).astype(int)
                            points2 = np.round(matches_im1 / np.array([W0 / w_ori, H0 / h_ori])).astype(int)

                            # show matched points
                            # show_correspondences(image1_pil, image2_pil, points1, points2)

                            # Convert points to tuples and swap positions
                            points1 = [[y, x] for x, y in points1]
                            points2 = [[y, x] for x, y in points2]


                    except Exception as e:
                        logger.error(
                            f"Local correspondence matching failed for {img_data.img_name} and object_id {img_data.obj_ids[i]}: {e}")

                    try:
                        # img_uv = np.load(matched_template['uv_crop']).astype(np.uint8)
                        img_uv = np.load(matched_template['uv']).astype(np.uint8)
                        # load_size = (img_data.masked_img[i].size[1], img_data.masked_img[i].size[0])
                        # _, img_uv, _ = img_utils.preprocess(Image.fromarray(img_uv), load_size=load_size)

                        # Estimate pose from correspondences
                        R_est, t_est, quality = utils.get_pose_from_correspondences(
                            points1,
                            points2,
                            img_data.y_offsets[i] * 0,
                            img_data.x_offsets[i] * 0,
                            np.array(img_uv),
                            img_data.cam_K,
                            norm_factors[str(img_data.obj_ids[i])],
                            config['scale_factor']
                        )
                    except Exception as e:
                        logger.error(
                            f"Not enough correspondences could be extracted for {img_data.img_name} and object_id {object_id}: {e}")
                        R_est = None
                        t_est = None
                        quality = None

                    # Fallback for pose estimation
                    if R_est is None:
                        R_est = np.array(matched_template['cam_R_m2c']).reshape((3, 3))
                        t_est = np.array([0., 0., 0.])
                        quality = -np.inf

                    end_time = time.time()
                    # err, acc = eval_utils.calculate_score(R_est, img_data.R_gts[i], int(img_data.obj_ids[i]), 0)
                    err = -quality
                    if err < min_err:
                        min_err = err
                        R_best, t_best = R_est, t_est
                        pose_est = True

                # Handle case with no pose estimation
                if not pose_est:
                    R_best = np.eye(3)  # Identity matrix
                    t_best = np.zeros(3)
                    logger.warning(f"No pose could be determined for {img_data.img_name} and object_id {object_id}")
                    score = 0.0
                else:
                    score = 0.0  # Placeholder for score; replace with actual scoring logic if needed

            else:
                R_best = np.eye(3)  # Identity matrix
                t_best = np.zeros(3)
                logger.warning(
                    f"No pose could be determined for {img_data.img_name} and object_id {object_id} due to missing crop")
                score = 0.0

            # Prepare for writing results
            R_best_str = " ".join(map(str, R_best.flatten()))
            t_best_str = " ".join(map(str, t_best * 1000))
            elapsed_time = end_time - start_time

            # Write results to CSV file
            if not np.allclose(R_best, np.eye(3)):
                with open(csv_file, mode='a', newline='') as csvfile:
                    csv_writer = csv.writer(csvfile)
                    csv_writer.writerow(
                        [img_data.scene_id, img_data.img_name, object_id, score, R_best_str, t_best_str, elapsed_time])

            # Debug image creation if enabled
            if config['debug_imgs'] and i % config['debug_imgs'] == 0:
                dbg_img = vis_utils.create_debug_image(R_best, t_best, img_data.R_gts[i], img_data.t_gts[i],
                                                       np.asarray(img_data.img), img_data.cam_K,
                                                       img_data.model_infos[i], config['scale_factor'],
                                                       image_shape=(config['image_resolution'][0],
                                                                    config['image_resolution'][1]),
                                                       colEst=(0, 255, 0))

                dbg_img = cv2.cvtColor(dbg_img, cv2.COLOR_BGR2RGB)
                dbg_img_mask = cv2.hconcat([dbg_img, img_data.masks_cv2[i]]) if img_data.masks_cv2[i] is not None else dbg_img
                cv2.imwrite(os.path.join(debug_img_path, f"{img_data.img_name}_{img_data.obj_ids[i]}.png"),
                            dbg_img_mask)


if __name__ == "__main__":
    # Argument parsing
    parser = argparse.ArgumentParser(description='Test pose estimation inference on test set')
    parser.add_argument('--config_file',
                        default="./database/zs6d_configs/bop_eval_configs/cfg_lmo_inference_bop.json")
    args = parser.parse_args()

    # Load configuration and paths
    config = load_json_file(args.config_file)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    matcher = AsymmetricMASt3R.from_pretrained("naver/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric").to(device)
    print("Loading Matcher is done!")

    # Load ground truth files
    templates_gt = load_json_file(config['templates_gt_path'])
    data_gt = load_json_file(config['gt_path'])  # load fastSAM
    norm_factors = load_json_file(config['norm_factor_path'])

    # Set up results CSV file
    csv_file = os.path.join('./logs/results_fastsam', config['results_file'])
    create_directory_if_not_exists('./logs/results_fastsam')

    headers = ['scene_id', 'im_id', 'obj_id', 'score', 'R', 't', 'time']
    prepare_csv_file(csv_file, headers)

    # Prepare debug image path if needed
    if config['debug_imgs']:
        debug_img_path = os.path.join("./logs/debug_imgs_fastsam_pandaWD", config['results_file'].split(".csv")[0])
        os.makedirs(debug_img_path, exist_ok=True)

    # Load templates and process images
    templates_desc, templates_gt_new = process_templates(templates_gt, config)
    print("Processing input images:")
    process_images(data_gt, templates_gt_new, config, matcher, norm_factors, csv_file)
