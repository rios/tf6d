import json
import os
import cv2
import numpy as np
from pose_utils import utils
from pose_utils.img_utils import rle_to_mask


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
        # './data/bop/default_detections/core19_model_based_unseen/cnos-fastsam/cnos-fastsam_hb-test_db836947-020a-45bd-8ec5-c95560b68011.json'
        # './data/bop/default_detections/core19_model_based_unseen/cnos-fastsam/cnos-fastsam_tudl-test_c48a2a95-1b41-4a51-9920-a667cb3d7149.json',
        # './data/bop/default_detections/core19_model_based_unseen/cnos-fastsam/cnos-fastsam_lmo-test_3cb298ea-e2eb-4713-ae9e-5a7134c5da0f.json',
        # './data/bop/default_detections/core19_model_based_unseen/cnos-fastsam/cnos-fastsam_ycbv-test_f4f2127c-6f59-447c-95b3-28e1e591f1a1.json',
        # './data/bop/default_detections/core19_model_based_unseen/cnos-fastsam/cnos-fastsam_tless-test_8ca61cb0-4472-4f11-bce7-1362a12d396f.json',
        #  max_det_per_object_id=16,
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

        for detection in detection_im_info:
            obj_id = detection['category_id']

            # Convert segmentation mask from RLE and apply morphological opening
            mask = rle_to_mask(detection['segmentation']).astype(np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))

            # Skip if the detection covers more than half of the image
            if mask.sum() > (mask.shape[0] * mask.shape[1] * 255 // 3):
                print(f"Skipped Masked Object {obj_id} in image {image_id}")
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
    output_path = f'database/gts/test_gts/{dataset_name}_bop_test_gt_fastsam.json'
    with open(output_path, 'w') as f:
        json.dump(fastsam_gt, f)
    print(f"Results saved to {output_path}")


if __name__ == "__main__":
    main()
