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
import matplotlib.pyplot as plt
import itertools

# Append paths for dependencies
sys.path.append('../mast3r/')
sys.path.append('.')

from mast3r.model import AsymmetricMASt3R
from mast3r.fast_nn import fast_reciprocal_NNs
from dust3r.inference import inference
from dust3r.utils.image import load_images_crop, load_images_crop_resize

# Replace <#threads> with the number of threads you want to use
num_threads = '64'

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


def process_images(data_gt, templates_gt_new, config, matcher, norm_factors, csv_file, csv_file_all):
    """Process input images and estimate poses in a batch."""
    instance_id_counter = itertools.count(start=1)

    for all_id, img_labels in tqdm(data_gt.items()):
        scene_id, img_id = all_id.split("_")[0], all_id.split("_")[-1]
        img_path = os.path.join(config['dataset_path'], img_labels[0]['img_name'].split("./")[-1])
        img_name = img_path.split("/")[-1].split(".png")[0]

        # Load image and camera parameters
        img = Image.open(img_path)
        cam_K = np.array(img_labels[0]['cam_K']).reshape((3, 3))

        # Initialize ImageContainer_all for this image
        img_data = ImageContainer_all(
            img=img, img_name=img_name, scene_id=scene_id, cam_K=cam_K,
            crops=[], ori_crops=[], mask_crops=[], descs=[],
            x_offsets=[], y_offsets=[], obj_names=[], obj_ids=[],
            model_infos=[], t_gts=[], R_gts=[], masks=[]
        )

        # Process each object in the image
        for obj_index, img_label in enumerate(img_labels):
            bbox_gt = img_label[config['bbox_type']]
            if bbox_gt[2] == 0 or bbox_gt[3] == 0 or bbox_gt == [-1, -1, -1, -1]:
                continue

            img_data.t_gts.append(np.array(img_label['cam_t_m2c']) * config['scale_factor'])
            img_data.R_gts.append(np.array(img_label['cam_R_m2c']).reshape((3, 3)))
            img_data.obj_ids.append(str(img_label['obj_id']))
            img_data.model_infos.append(img_label['model_info'])

            try:
                # Load and preprocess mask
                mask = img_utils.rle_to_mask(img_label['mask_sam']).astype(np.uint8)
                mask_3_channel = np.stack([mask] * 3, axis=-1)
                bbox = img_utils.get_bounding_box_from_mask(mask)

                img_crop, y_offset, x_offset = img_utils.make_quadratic_crop_ratio(
                    np.array(img), bbox, patch_size=16, final_ratio=1.0
                )
                mask_crop, _, _ = img_utils.make_quadratic_crop_ratio(mask, bbox, patch_size=16, final_ratio=1.0)

                img_data.ori_crops.append(Image.fromarray(img_crop))
                img_crop = cv2.bitwise_and(img_crop, img_crop, mask=mask_crop)
                img_data.crops.append(Image.fromarray(img_crop))
                img_data.mask_crops.append(mask_crop)
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

        # Process each crop for pose estimation
        for i in range(len(img_data.crops)):
            start_time = time.time()
            object_id = img_data.obj_ids[i]
            instance_id = next(instance_id_counter)

            if img_data.crops[i] is not None:
                matched_templates = templates_gt_new[object_id]
                # h, w for torch
                load_size = (img_data.crops[i].size[1], img_data.crops[i].size[0])
                resize_size = (320, 320)

                # Prepare templates and masks in batch
                templates, masks = [], []
                for template_info in matched_templates:
                    templates.append(Image.open(template_info['img_crop']))
                    masks.append(Image.open(template_info['mask_crop']))

                mask1, mask1_pil, _ = img_utils.preprocess(Image.fromarray(img_data.mask_crops[i]),
                                                           load_size=resize_size)

                try:
                    with torch.no_grad():
                        # Preprocess main image and templates in batch
                        image1_batch, image1_pil, _ = img_utils.preprocess_norm(
                            img_data.ori_crops[i], model_type='mast3r', load_size=load_size
                        )
                        processed_templates = [
                            img_utils.preprocess_norm(template, model_type='mast3r', load_size=load_size)[1] for
                            template in templates]
                        images = [tuple(load_images_crop_resize([image1_pil, template], resize_size, verbose=False)) for
                                  template in processed_templates]

                        # Inference on batch
                        output = inference(images, matcher, device, batch_size=len(matched_templates),
                                           verbose=False)

                        min_err, R_best, t_best, best_score = np.inf, np.eye(3), np.array([0, 0, 0]), 0.0

                        for idx in range(len(matched_templates)):
                            try:
                                # The decs and view of each pair
                                desc1 = output['pred1']['desc'][idx]
                                desc2 = output['pred2']['desc'][idx]

                                H0, W0 = output['view1']['true_shape'][idx]
                                H1, W1 = output['view2']['true_shape'][idx]

                                # Process descriptors
                                desc1 = torch.nn.functional.normalize(desc1.squeeze(0).detach(), p=2, dim=-1)
                                desc2 = torch.nn.functional.normalize(desc2.squeeze(0).detach(), p=2, dim=-1)

                                # Find matches and filter valid points
                                # Find 2D-2D matches
                                matches_im0, matches_im1 = fast_reciprocal_NNs(desc1, desc2, subsample_or_initxy1=8,
                                                                               device=device, dist='dot')

                                valid_matches_im0 = (matches_im0[:, 0] >= 3) & (matches_im0[:, 0] < int(W0) - 3) & (
                                        matches_im0[:, 1] >= 3) & (matches_im0[:, 1] < int(H0) - 3)

                                valid_matches_im1 = (matches_im1[:, 0] >= 3) & (matches_im1[:, 0] < int(W1) - 3) & (
                                        matches_im1[:, 1] >= 3) & (matches_im1[:, 1] < int(H1) - 3)

                                valid_matches = valid_matches_im0 & valid_matches_im1
                                matches_im0, matches_im1 = matches_im0[valid_matches], matches_im1[valid_matches]

                                # Calculate match scores
                                f1 = desc1[matches_im0[:, 1], matches_im0[:, 0]]
                                f2 = desc2[matches_im1[:, 1], matches_im1[:, 0]]
                                match_scores = (f1 * f2).sum(dim=-1)

                                # Apply mask scores
                                mask2 = img_utils.preprocess(masks[idx], load_size=resize_size)[0]

                                # Process descriptors and convert mask to bool
                                mask1 = (mask1 > 0.5).float().permute(1, 2, 0)
                                mask2 = (mask2 > 0.5).float().permute(1, 2, 0)

                                masked_score_1 = mask1.squeeze()[matches_im0[:, 1], matches_im0[:, 0]]
                                masked_score_2 = mask2.squeeze()[matches_im1[:, 1], matches_im1[:, 0]]
                                masked_scores = torch.minimum(masked_score_1, masked_score_2)
                                valid_mask = masked_scores == 1
                                matches_im0, matches_im1, match_scores = matches_im0[valid_mask], matches_im1[
                                    valid_mask], \
                                    match_scores[valid_mask]

                                # Pose estimation
                                points1 = np.round(
                                    matches_im0 / np.array([W0 / load_size[1], H0 / load_size[0]])).astype(int)
                                points2 = np.round(
                                    matches_im1 / np.array([W0 / load_size[1], H0 / load_size[0]])).astype(int)

                                # Fundamental Matrix
                                # fundamental_matrix, mask = cv2.findFundamentalMat(np.array(points1), np.array(points2),
                                #                                                   cv2.FM_RANSAC, ransacReprojThreshold=3,
                                #                                                   confidence=0.95, maxIters=2000)
                                # # # Select inlier points (points that were used to compute the fundamental matrix)
                                # points1 = points1[mask.ravel() == 1]
                                # points2 = points2[mask.ravel() == 1]

                                # Convert points to tuples and swap positions
                                points1 = [[y, x] for x, y in points1]
                                points2 = [[y, x] for x, y in points2]

                            except Exception as e:
                                logger.error(
                                    f"Local correspondence matching failed for {img_data.img_name} and object_id {img_data.obj_ids[i]}: {e}")

                            try:
                                # uv map
                                img_uv = np.load(matched_templates[idx]['uv_crop']).astype(np.uint8)
                                _, img_uv, _ = img_utils.preprocess(Image.fromarray(img_uv), load_size=load_size)

                                R_est, t_est, quality, inliers = utils.get_pose_from_correspondences(
                                    points1, points2, img_data.y_offsets[i], img_data.x_offsets[i], np.array(img_uv),
                                    img_data.cam_K, norm_factors[str(object_id)], config['scale_factor'],
                                    pnp_refine_lm=True
                                )
                                match_scores = match_scores[inliers].numpy().sum()

                            except Exception as e:
                                logger.error(
                                    f"Not enough correspondences could be extracted for {img_data.img_name} and object_id {object_id}: {e}")
                                R_est, t_est, quality = None, None, None
                                match_scores = -np.inf

                            if R_est is None:
                                R_est = np.array(matched_templates[idx]['cam_R_m2c']).reshape((3, 3))
                                t_est = np.array(matched_templates[idx]['cam_t_m2c'])
                                quality = -np.inf

                            # Calculate error and update if best
                            err = -match_scores
                            R_str = " ".join(map(str, R_est.flatten()))
                            t_str = " ".join(map(str, t_est * 1000))
                            elapsed_time = time.time() - start_time
                            GT_err, _ = eval_utils.calculate_score(R_est, img_data.R_gts[i], int(img_data.obj_ids[i]), 0)
                            # err = GT_err

                            # Write to csv_file_all for every template
                            with open(csv_file_all, mode='a', newline='') as csvfile_all:
                                csv_writer_all = csv.writer(csvfile_all)
                                csv_writer_all.writerow(
                                    [img_data.scene_id, img_data.img_name, object_id,
                                     GT_err,
                                     match_scores, quality,
                                     R_str, t_str,
                                     elapsed_time, instance_id]
                                )

                            if err < min_err:
                                min_err = err
                                R_best, t_best, best_score = R_est, t_est, quality

                        # Write only the best result to csv_file
                        R_best_str = " ".join(map(str, R_best.flatten()))
                        t_best_str = " ".join(map(str, t_best * 1000))
                        with open(csv_file, mode='a', newline='') as csvfile:
                            csv_writer = csv.writer(csvfile)
                            csv_writer.writerow(
                                [img_data.scene_id, img_data.img_name, object_id, best_score, R_best_str, t_best_str,
                                 elapsed_time]
                            )

                        # Save debug image if configured
                        if config['debug_imgs'] and i % config['debug_imgs'] == 0:
                            dbg_img = vis_utils.create_debug_image(
                                R_best, t_best, img_data.R_gts[i], img_data.t_gts[i],
                                np.asarray(img_data.img), img_data.cam_K, img_data.model_infos[i],
                                config['scale_factor'],
                                image_shape=(config['image_resolution'][0], config['image_resolution'][1]),
                                colEst=(0, 255, 0)
                            )
                            dbg_img = cv2.cvtColor(dbg_img, cv2.COLOR_BGR2RGB)
                            dbg_img_mask = cv2.hconcat([dbg_img, img_data.masks[i]]) if img_data.masks[
                                                                                            i] is not None else dbg_img
                            cv2.imwrite(os.path.join(debug_img_path, f"{img_data.img_name}_{img_data.obj_ids[i]}.png"),
                                        dbg_img_mask)

                except Exception as e:
                    logger.error(f"Error processing batch for img {img_data.img_name} and object_id {object_id}: {e}")


if __name__ == "__main__":
    # Argument parsing
    dataset_name = 'icbin'
    parser = argparse.ArgumentParser(description='Test pose estimation inference on test set')
    parser.add_argument('--config_file',
                        default="./database/zs6d_configs/bop_eval_configs/cfg_%s_inference_bop_zero.json" % (
                            dataset_name))
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
    csv_file = os.path.join('./logs/results_fastsam_final', config['results_file'])
    create_directory_if_not_exists('./logs/results_fastsam_final')

    # Set up results CSV file for all results
    csv_file_all = os.path.join('./logs/results_fastsam_final', 'All' + config['results_file'])
    create_directory_if_not_exists('./logs/results_fastsam_final')

    headers = ['scene_id', 'im_id', 'obj_id', 'score', 'R', 't', 'time']
    prepare_csv_file(csv_file, headers)

    headers_all = ['scene_id', 'im_id', 'obj_id', 'GT_error', 'score', 'pnp_inliners', 'R', 't', 'time', 'instance_id']
    prepare_csv_file(csv_file_all, headers_all)

    # Prepare debug image path if needed
    if config['debug_imgs']:
        debug_img_path = os.path.join("./logs/debug_imgs_fastsam_panda/" + dataset_name,
                                      config['results_file'].split(".csv")[0])
        os.makedirs(debug_img_path, exist_ok=True)

    # Load templates and process images
    templates_desc, templates_gt_new = process_templates(templates_gt, config)
    print("Processing input images:")
    process_images(data_gt, templates_gt_new, config, matcher, norm_factors, csv_file, csv_file_all)
