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
from tqdm import tqdm
from pose_utils.data_utils import ImageContainer_all
import pose_utils.img_utils as img_utils
import pose_utils.utils as utils
import pose_utils.vis_utils as vis_utils
import pose_utils.eval_utils as eval_utils
from matching import get_matcher
import matplotlib.pyplot as plt

# Replace <#threads> with the number of threads you want to use
num_threads = '16'

# Setting the environment variables
os.environ["OPENBLAS_NUM_THREADS"] = num_threads
os.environ["GOTO_NUM_THREADS"] = num_threads
os.environ["OMP_NUM_THREADS"] = num_threads
os.environ["MKL_NUM_THREADS"] = num_threads

# Setup logging configuration
logging.basicConfig(level=logging.INFO,
                    filename="pose_estimation_matching.log",
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add path for image matching models
sys.path.append('../image-matching-models/')
sys.path.append('../image-matching-models/matching/third_party')


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

def create_directory_if_not_exists(directory):
    """Create a directory if it does not exist."""
    if not os.path.exists(directory):
        os.makedirs(directory)

def load_json_file(file_path):
    """Load a JSON file and return its content."""
    with open(file_path, 'r') as f:
        return json.load(f)

def prepare_csv_file(csv_file_path, headers):
    """Create a CSV file and write headers to it."""
    with open(csv_file_path, mode='w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(headers)

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

                img_prep, img_crop, _ = img_utils.preprocess(Image.fromarray(img_crop), load_size=(420, 420))
                img_data.y_offsets.append(y_offset)
                img_data.x_offsets.append(x_offset)
                img_data.masks.append(mask_3_channel)

            except Exception as e:
                logger.warning(
                    f"Loading mask and extracting descriptor failed for img {img_path} and object_id {obj_index}: {e}")
                img_data.crops.append(None)
                img_data.y_offsets.append(None)
                img_data.x_offsets.append(None)
                img_data.masks.append(None)

        # Estimate pose for each crop
        for i in range(len(img_data.crops)):
            start_time = time.time()
            object_id = img_data.obj_ids[i]

            # if object_id not in ['16']:
            #     continue

            if img_data.crops[i] is not None:
                min_err = np.inf
                pose_est = False
                matched_templates = templates_gt_new[object_id]

                # Iterate through matched templates for pose estimation
                for matched_template in matched_templates:
                    template = Image.open(matched_template['img_crop'])
                    try:
                        with torch.no_grad():
                            # h, w = img_data.crops[i].size[1], img_data.crops[i].size[0]
                            h, w = 224, 224
                            load_size = (h, w)  # h, w for torch

                            # Preprocess images and masks
                            image1_batch, image1_pil, _ = img_utils.preprocess(img_data.crops[i], load_size=load_size)
                            image2_batch, image2_pil, _ = img_utils.preprocess(template, load_size=load_size)
                            mask1, _, _ = img_utils.preprocess(Image.fromarray(img_data.mask_crops[i]),
                                                               load_size=load_size)
                            mask2, _, _ = img_utils.preprocess(Image.open(matched_template['mask_crop']),
                                                               load_size=load_size)

                            # Perform matching
                            result = matcher(image1_batch.squeeze(0), image2_batch.squeeze(0))
                            matches_im0, matches_im1 = result['matched_kpts0'], result['matched_kpts1']

                           # masked scores
                            masked_score_1 = mask1.squeeze()[matches_im0[:, 1], matches_im0[:, 0]]
                            masked_score_2 = mask2.squeeze()[matches_im1[:, 1], matches_im1[:, 0]]

                            masked_scores = np.minimum(masked_score_1, masked_score_2)
                            # # Apply Masks
                            if len(masked_scores) > 0:
                                matches_im0 = matches_im0[masked_scores==1]
                                matches_im1 = matches_im1[masked_scores==1]
                                # match_scores = match_scores[masked_scores==1]

                            # Scale points back to original size
                            w_ori, h_ori = img_data.crops[i].size
                            points1 = np.round(matches_im0 / np.array([w / w_ori, h / h_ori])).astype(int)
                            points2 = np.round(matches_im1 / np.array([w / w_ori, h / h_ori])).astype(int)

                            # Show correspondences
                            _, image1_pil, _ = img_utils.preprocess(img_data.crops[i], load_size=(h_ori, w_ori))
                            _, image2_pil, _ = img_utils.preprocess(template, load_size=(h_ori, w_ori))
                            # show_correspondences(image1_pil, image2_pil, points1, points2)

                            points1 = [[y, x] for x, y in points1]  # swap coordinates
                            points2 = [[y, x] for x, y in points2]


                    except Exception as e:
                        logger.error(
                            f"Local correspondence matching failed for {img_data.img_name} and object_id {img_data.obj_ids[i]}: {e}")

                    try:
                        img_uv = np.load(matched_template['uv_crop']).astype(np.uint8)
                        load_size = (img_data.crops[i].size[1], img_data.crops[i].size[0])
                        _, img_uv, _ = img_utils.preprocess(Image.fromarray(img_uv), load_size=load_size)

                        # Estimate pose from correspondences
                        R_est, t_est, quality = utils.get_pose_from_correspondences(
                            points1,
                            points2,
                            img_data.y_offsets[i],
                            img_data.x_offsets[i],
                            np.array(img_uv),
                            img_data.cam_K,
                            norm_factors[str(img_data.obj_ids[i])],
                            config['scale_factor']
                        )
                    except Exception as e:
                        logger.error(
                            f"Not enough correspondences could be extracted for {img_data.img_name} and object_id {object_id}: {e}")
                        R_est, t_est, quality = None, None, None


                    # Fallback for pose estimation
                    if R_est is None:
                        R_est = np.array(matched_template['cam_R_m2c']).reshape((3, 3))
                        t_est = np.array([0., 0., 0.])
                        quality = -np.inf

                    end_time = time.time()
                    err, acc = eval_utils.calculate_score(R_est, img_data.R_gts[i], int(img_data.obj_ids[i]), 0)
                    # err = -quality

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
                dbg_img_mask = cv2.hconcat([dbg_img, img_data.masks[i]]) if img_data.masks[i] is not None else dbg_img
                cv2.imwrite(os.path.join(debug_img_path, f"{img_data.img_name}_{img_data.obj_ids[i]}.png"),
                            dbg_img_mask)


if __name__ == "__main__":
    # Argument parsing
    parser = argparse.ArgumentParser(description='Test pose estimation inference on test set')
    parser.add_argument('--config_file', default="./database/zs6d_configs/bop_eval_configs/cfg_ycbv_inference_bop_zero.json")
    args = parser.parse_args()

    # Load configuration and paths
    config = load_json_file(args.config_file)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    matcher = get_matcher(['mast3r', 'se2loftr'], device=device)
    print("Loading Matcher is done!")

    # Load ground truth files
    templates_gt = load_json_file(config['templates_gt_path'])
    data_gt = load_json_file(config['gt_path'])
    norm_factors = load_json_file(config['norm_factor_path'])

    # Set up results CSV file
    csv_file = os.path.join('./logs/results_fastsam_final/', config['results_file'])

    # Set up results CSV file
    create_directory_if_not_exists('./logs/results_fastsam_final')

    # Set up results CSV file for all results
    # csv_file_all = os.path.join('./logs/results_fastsam_final', 'All' + config['results_file'])
    # create_directory_if_not_exists('./logs/results_fastsam_final')

    headers = ['scene_id', 'im_id', 'obj_id', 'score', 'R', 't', 'time']
    prepare_csv_file(csv_file, headers)

    # Prepare debug image path if needed
    if config['debug_imgs']:
        debug_img_path = os.path.join("./logs/debug_imgs_fastsam_panda", config['results_file'].split(".csv")[0])
        os.makedirs(debug_img_path, exist_ok=True)

    # Load templates and process images
    templates_desc, templates_gt_new = process_templates(templates_gt, config)
    print("Processing input images:")
    process_images(data_gt, templates_gt_new, config, matcher, norm_factors, csv_file)
