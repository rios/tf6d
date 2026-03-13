import json
import os
import cv2
import numpy as np
from pose_utils import utils
from pose_utils.img_utils import rle_to_mask
import glob
import imageio
from PIL import Image
from PIL import ImageDraw


def draw_and_calculate_iou(img, bbox, gT_bbox):
    # Convert bounding boxes from (x, y, w, h) to (x1, y1, x2, y2)
    bbox_converted = (bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3])  # Predicted bounding box
    gT_bbox_converted = (
        gT_bbox[0], gT_bbox[1], gT_bbox[0] + gT_bbox[2], gT_bbox[1] + gT_bbox[3])  # Ground truth bounding box

    # Draw both bounding boxes on the image
    draw = ImageDraw.Draw(img)
    draw.rectangle(bbox_converted, outline="red", width=3)  # Predicted bbox in red
    draw.rectangle(gT_bbox_converted, outline="blue", width=3)  # Ground truth bbox in blue
    img.show()  # Display or save the image if needed


def mask_iou(mask1: np.ndarray, mask2: np.ndarray):
    inter = np.logical_and(mask1, mask2)
    union = np.logical_or(mask1, mask2)
    union_count = float(union.sum())
    if union_count > 0:
        return inter.sum() / union_count
    else:
        return 0.0


def calculate_iou(bbox1, bbox2):
    # Convert (x, y, w, h) to (x1, y1, x2, y2)
    x1_1, y1_1, x2_1, y2_1 = bbox1[0], bbox1[1], bbox1[0] + bbox1[2], bbox1[1] + bbox1[3]
    x1_2, y1_2, x2_2, y2_2 = bbox2[0], bbox2[1], bbox2[0] + bbox2[2], bbox2[1] + bbox2[3]

    # Calculate intersection
    x1_inter = max(x1_1, x1_2)
    y1_inter = max(y1_1, y1_2)
    x2_inter = min(x2_1, x2_2)
    y2_inter = min(y2_1, y2_2)

    # Compute area of intersection
    inter_area = max(0, x2_inter - x1_inter) * max(0, y2_inter - y1_inter)

    # Compute areas of each bounding box
    area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
    area2 = (x2_2 - x1_2) * (y2_2 - y1_2)

    # Compute IoU
    iou = inter_area / (area1 + area2 - inter_area)
    return iou


def main():
    # Dataset setup
    dataset_name = 'itodd'
    dataset_path = './data/bop/'
    cad_name = 'models_cad' if dataset_name == 'tless' else 'models'

    if dataset_name == 'tless':
        test_set_name = 'test_primesense'
    elif dataset_name == 'hb':
        test_set_name = 'test_primesense'
    else:
        test_set_name = 'test'

    # Load data
    test_list, cnos_dets = utils.load_test_list_and_cnos_detections(
        dataset_path, dataset_name,
        './data/bop/default_detections/core19_model_based_unseen/cnos-fastsam/cnos-fastsam_itodd-test_df32d45b-301c-4fc9-8769-797904dd9325.json',
        # './data/bop/default_detections/core19_model_based_unseen/cnos-fastsam/cnos-fastsam_hb-test_db836947-020a-45bd-8ec5-c95560b68011.json',
        max_det_per_object_id=15,
    )
    # camera_info = utils.load_json(os.path.join(dataset_path, dataset_name, 'camera_primesense.json'))
    # camera_info = utils.load_json(os.path.join(dataset_path, dataset_name, 'camera_uw.json'))
    # camera_info = utils.load_json(os.path.join(dataset_path, dataset_name, 'camera.json'))
    # camera_info = utils.load_json(os.path.join(dataset_path, dataset_name, 'camera.json'))
    models_info = utils.load_json(os.path.join(dataset_path, dataset_name, cad_name, 'models_info.json'))

    # Initialize results storage
    fastsam_gt = {}

    # Iterate over test scenes
    for scene_im, scene_im_info in test_list.items():
        scene_id, image_id = scene_im.split('_')[0], scene_im.split('_')[1].lstrip('0') or '0'

        # # Load scene-related data
        base_path = os.path.join(dataset_path, dataset_name, test_set_name, scene_id)
        scene_camera = utils.load_json(os.path.join(base_path, 'scene_camera.json'))
        # scene_gt_info = utils.load_json(os.path.join(base_path, 'scene_gt_info.json'))
        # scene_gt = utils.load_json(os.path.join(base_path, 'scene_gt.json'))
        #
        # # Merge ground truth information
        # scene_list = {k: [{**v1, **v2} for v1, v2 in zip(scene_gt_info[k], scene_gt[k])] for k in scene_gt_info}
        # gt_im_info = scene_list[image_id]
        # gt_mapping = {obj_data['obj_id']: obj_data for obj_data in gt_im_info}

        # Process each detection
        detection_im_info = cnos_dets[scene_im]
        im_info = f"{scene_id}_{image_id}"
        fastsam_gt[im_info] = []
        path_prefix = f'./data/bop/{dataset_name}/{test_set_name}/{scene_id}/mask_visib/{int(image_id):06d}'
        matching_files = glob.glob(f'{path_prefix}*.png')

        # masks
        masks = []
        for file_path in matching_files:
            gt_mask = np.array(Image.open(file_path)) / 255.0
            masks.append(gt_mask)

        # for anno_id, gt in enumerate(gt_im_info):
        #     gt_mask_name = f'./data/bop/{dataset_name}/' + f'/{test_set_name}/{scene_id}/mask_visib/{int(image_id):06d}_{int(anno_id):06d}.png'
        #     gt_mask = np.array(Image.open(gt_mask_name)) / 255.0
        #
        #     iou = mask_iou(mask, np.array(gt_mask))
        #     # bbox_gt = img_utils.get_bounding_box_from_mask(gt_mask)
        #     # bbox_iou = calculate_iou(bbox, bbox_gt)
        #
        #     if iou > best_anno_iou:
        #         best_anno_iou = iou
        #         best_anno_id = anno_id

        for detection in detection_im_info:
            obj_id = detection['category_id']

            # Convert segmentation mask from RLE and apply morphological opening
            mask = rle_to_mask(detection['segmentation']).astype(np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))

            # Skip if the detection covers more than half of the image
            if mask.sum() > (mask.shape[0] * mask.shape[1] * 255 // 3):
                print(f"Skipped Masked Object {obj_id} in image {image_id}")
                continue

            best_anno_iou = 0
            best_anno_id = 0

            for anno_id, gt_mask in enumerate(masks):
                iou = mask_iou(mask, np.array(gt_mask))

                if iou > best_anno_iou:
                    best_anno_iou = iou
                    best_anno_id = anno_id

            if best_anno_iou < 0.1:
                print("IoU:", best_anno_iou)
                continue

            if dataset_name == 'hb':
                # Collect related information
                img_name = f'./{test_set_name}/{scene_id}/rgb/{int(image_id):06d}.png'
                model_path = f'./models/obj_{obj_id:06d}.ply'
                depth_name = f'./{test_set_name}/{scene_id}/depth/{int(image_id):06d}.png'
                model_info = models_info[str(obj_id)]
                # obj_gt = gt_mapping[obj_id]
            else:
                # Collect related information
                img_name = f'./{test_set_name}/{scene_id}/gray/{int(image_id):06d}.tif'
                model_path = f'./models/obj_{obj_id:06d}.ply'
                depth_name = f'./{test_set_name}/{scene_id}/depth/{int(image_id):06d}.tif'
                model_info = models_info[str(obj_id)]
                # obj_gt = gt_mapping[obj_id]

            # Compile all detection-related data
            object_info = {
                'scene_id': scene_id,
                "obj_id": obj_id,
                'img_name': img_name,
                'model_path': model_path,
                'model_info': model_info,
                'sam_bbox': detection['bbox'],
                'sam_score': detection['score'],
                'mask_sam': detection['segmentation'],
                # **obj_gt,  # Unpack object ground truth data
                **scene_camera[image_id],  # Unpack camera data
            }
            fastsam_gt[im_info].append(object_info)

    # After processing all scenes and detections
    # Remove entries with empty lists as values
    fastsam_gt = {k: v for k, v in fastsam_gt.items() if v}

    # Save results to a JSON file
    output_path = f'database/gts/test_gts/{dataset_name}_bop_test_gt_fastsam_filter.json'
    with open(output_path, 'w') as f:
        json.dump(fastsam_gt, f)
    print(f"Results saved to {output_path}")


if __name__ == "__main__":
    main()
