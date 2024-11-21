import torch
import cv2
from typing import List, Tuple
import numpy as np
import copy
from tqdm import tqdm
import json
from src.correspondences import _log_bin
import poselib
from scipy.optimize import least_squares


def find_template(desc_input, desc_templates, num_results):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = "cpu"
    desc_input = desc_input.to(device)
    desc_templates = desc_templates.to(device)

    similarities = [(1 - torch.nn.functional.cosine_similarity(desc_input.flatten(), desc_template.flatten(),
                                                               dim=0).detach().cpu(), i)
                    for i, desc_template in enumerate(desc_templates)
                    if desc_input.shape == desc_template.shape]

    sorted_sims = sorted(similarities, key=lambda x: x[0], reverse=True)

    result = [(sim[0], sim[1]) for sim in sorted_sims[:num_results]]

    desc_input = desc_input.detach().to("cpu")
    desc_templates = desc_templates.detach().to("cpu")

    # # Clear GPU memory
    torch.cuda.empty_cache()

    return result


def find_template_cpu_with_stats(desc_input, desc_templates, use_cls=True):
    if use_cls:
        # Flatten and normalize the desc_input
        desc_input_flat = desc_input.ravel()
        desc_input_norm = np.linalg.norm(desc_input_flat)

        # Precompute flattening and norms for all templates
        templates_flat = [template.ravel() for template in desc_templates]
        templates_norm = [np.linalg.norm(template_flat) for template_flat in templates_flat]
    else:
        # Flatten and normalize the desc_input
        desc_input_flat = desc_input[1:, :].ravel()
        desc_input_norm = np.linalg.norm(desc_input_flat)

        # Precompute flattening and norms for all templates
        templates_flat = [template[1:, :].ravel() for template in desc_templates]
        templates_norm = [np.linalg.norm(template_flat) for template_flat in templates_flat]

    # Compute cosine similarities in a vectorized manner
    similarities = [np.dot(desc_input_flat, template_flat) / (desc_input_norm * template_norm)
                    for template_flat, template_norm in zip(templates_flat, templates_norm)]

    # Calculate max and mean similarity
    max_similarity = max(similarities)
    mean_similarity = np.mean(similarities)

    # Return max and mean similarity
    return max_similarity, mean_similarity


def find_template_cpu(desc_input, desc_templates, num_results, use_cls=True):
    if use_cls:
        # Flatten and normalize the desc_input
        desc_input_flat = desc_input.ravel()
        desc_input_norm = np.linalg.norm(desc_input_flat)

        # Precompute flattening and norms for all templates
        templates_flat = [template.ravel() for template in desc_templates]
        templates_norm = [np.linalg.norm(template_flat) for template_flat in templates_flat]
    else:
        # Flatten and normalize the desc_input
        desc_input_flat = desc_input[1:, :].ravel()
        desc_input_norm = np.linalg.norm(desc_input_flat)

        # Precompute flattening and norms for all templates
        templates_flat = [template[1:, :].ravel() for template in desc_templates]
        templates_norm = [np.linalg.norm(template_flat) for template_flat in templates_flat]

    # Compute cosine similarities in a vectorized manner
    similarities = [(np.dot(desc_input_flat, template_flat) / (desc_input_norm * template_norm), i)
                    for i, (template_flat, template_norm) in enumerate(zip(templates_flat, templates_norm))]

    # Sort the results
    sorted_sims = sorted(similarities, key=lambda x: x[0], reverse=True)

    # Return the top num_results
    return sorted_sims[:num_results]


def find_template_cpu_matrix(desc_input, desc_templates, num_results):
    # Flatten and normalize the desc_input
    desc_input_flat = desc_input.ravel()
    desc_input_norm = np.linalg.norm(desc_input_flat)

    # Convert list of templates to a 3D NumPy array and flatten along the last two dimensions
    templates_array = np.array(desc_templates).reshape(len(desc_templates), -1)
    templates_norms = np.linalg.norm(templates_array, axis=1)

    # Compute cosine similarities using matrix operations
    similarities = np.dot(templates_array, desc_input_flat) / (templates_norms * desc_input_norm)

    # Get the indices of the top num_results similarities
    top_indices = np.argsort(similarities)[-num_results:][::-1]

    # Return the top similarities and their indices
    return [(similarities[i], i) for i in top_indices]


def find_template_patch(
        descriptors1: torch.Tensor,
        descriptors2: torch.Tensor,
        num_pairs: int = 10,
        top_k: int = 5,
        use_best_buddies: bool = True,
        use_bin: bool = True
) -> List[Tuple[float, int]]:
    """
    Find the most similar K samples between descriptors1 and a set of samples in descriptors2,
    using the sum of the top `num_pairs` similarities for each sample.

    Parameters:
        descriptors1 (torch.Tensor): Descriptors from the first image (shape: 1, 256, num_patches).
        descriptors2 (torch.Tensor): Descriptors from multiple samples (shape: N, 256, num_patches).
        num_patches (Tuple[int, int]): The size of the token grid in the feature map.
        num_pairs (int): Number of correspondence pairs to consider for summing similarities.
        top_k (int): Number of most similar samples to return.
        patch_size (int): Size of each patch.
        use_best_buddies (bool): Whether to compute mutual best buddies (default: True).
        seed (int): Random seed for reproducibility.

    Returns:
        List of tuples where each tuple contains:
            - The similarity score of the sample.
            - The index of the sample in descriptors2.
    """
    # Set random seed for reproducibility

    # Ensure descriptors are on the same device
    descriptors1 = descriptors1.unsqueeze(0)
    descriptors1 = descriptors1[:, 1:, :]
    descriptors2 = descriptors2[:, 1:, :]

    if use_bin:
        descriptors1 = descriptors1.unsqueeze(1)  # Bx1xtx(dxh)
        descriptors1 = _log_bin(descriptors1, hierarchy=1).to(descriptors1.device)  # Bx1x(t)x(dxh)

        descriptors2 = descriptors2.unsqueeze(1)  # Bx1xtx(dxh)
        descriptors2 = _log_bin(descriptors2, hierarchy=1).to(descriptors1.device)  # Bx1x(t)x(dxh)

        descriptors2 = descriptors2.squeeze(1)
        descriptors1 = descriptors1.squeeze(1)

    descriptors1 = descriptors1.permute(0, 2, 1)
    descriptors2 = descriptors2.permute(0, 2, 1)

    num_patches = (int(np.sqrt(descriptors1.shape[-1])), int(np.sqrt(descriptors1.shape[-1])))

    # Normalize descriptors
    descriptors1 = descriptors1 / descriptors1.norm(dim=1, keepdim=True)  # Shape: (1, D, num_patches)
    descriptors2 = descriptors2 / descriptors2.norm(dim=1, keepdim=True)  # Shape: (N, D, num_patches)

    # Compute pairwise patch-level similarities
    similarities = torch.einsum('bci,ncj->bnij', descriptors1, descriptors2).squeeze(
        0)  # Shape: (N, num_patches, num_patches)

    # Use best buddies if needed
    if use_best_buddies:
        sim_1, nn_1 = torch.max(similarities, dim=-1)  # Shape: (N, num_patches)
        sim_2, nn_2 = torch.max(similarities, dim=-2)  # Shape: (N, num_patches)
        image_idxs = torch.arange(num_patches[0] * num_patches[1]).to(descriptors1.device)
        bbs_mask = nn_2.gather(1, nn_1) == image_idxs[None]  # Ensure mutual nearest neighbors
    else:
        bbs_mask = torch.ones(similarities.shape[:2], dtype=torch.bool)
        sim_1 = similarities.max(dim=-1).values  # Get the best similarity for each patch

    # Apply the bbs_mask to filter out non-best buddy matches
    sim_1_masked = sim_1 * bbs_mask

    # Find the top `num_pairs` for each sample and sum them (vectorized)
    top_similarities, _ = torch.topk(sim_1_masked, num_pairs, dim=-1)  # Shape: (N, num_pairs)
    summed_similarities = torch.sum(top_similarities, dim=-1)  # Sum of top `num_pairs`, Shape: (N,)

    # Sort and find the top K most similar samples based on summed similarities
    top_k_similarities, top_k_indices = torch.topk(summed_similarities, top_k, largest=True)

    # Create list of (similarity, index) tuples
    result = [(top_k_similarities[i].item(), top_k_indices[i].item()) for i in range(top_k)]

    return result


def _transform_to_xyz(r, g, b, x_ct, y_ct, z_ct, x_scale, y_scale, z_scale):
    x = r / 255.
    x = x * 2 - 1
    x = x * x_scale + x_ct
    y = g / 255.
    y = y * 2 - 1
    y = y * y_scale + y_ct
    z = b / 255.
    z = z * 2 - 1
    z = z * z_scale + z_ct

    return x, y, z


def transform_2D_3D(points, img_uv, norm_factor):
    x_ct = norm_factor["x_ct"]
    y_ct = norm_factor["y_ct"]
    z_ct = norm_factor["z_ct"]
    x_scale = norm_factor["x_scale"]
    y_scale = norm_factor["y_scale"]
    z_scale = norm_factor["z_scale"]

    points_3D = []

    for point in points:
        r, g, b = img_uv[point[0], point[1]]
        x, y, z = _transform_to_xyz(r, g, b, x_ct, y_ct, z_ct, x_scale, y_scale, z_scale)
        points_3D.append([x, y, z])

    return points_3D

def get_pose_from_correspondences_mask(points1, points2, y_offset, x_offset, img_uv, cam_K, norm_factor, scale_factor,
                                  weights, resize_factor=1.0,  pnp_refine_lm=False):
    # filter valid points
    valid_points1 = []
    valid_points2 = []
    valid_weights = []

    for point1, point2, weight in zip(points1, points2, np.array(weights)):
        if np.any(img_uv[round(point2[0]), round(point2[1])] != [0, 0, 0]):
            valid_points1.append(point1)
            valid_points2.append(point2)
            valid_weights.append(weight)

    # Check if enough correspondences for PnPRansac
    if len(valid_points1) < 6:
        return None, None

    points2_3D = transform_2D_3D(valid_points2, img_uv, norm_factor)

    valid_points1 = np.array(valid_points1).astype(np.float64) / resize_factor

    valid_points1[:, 0] += y_offset
    valid_points1[:, 1] += x_offset

    valid_points1[:, [0, 1]] = valid_points1[:, [1, 0]]

    pnp_inlier_thresh = 8
    pnp_required_ransac_conf = 0.99
    pnp_ransac_iter = 400

    try:
        pose_est_success, rvec, tvec, inliers = cv2.solvePnPRansac(np.array(points2_3D).astype(np.float64),
                                                                   valid_points1, cam_K,
                                                                   distCoeffs=None, iterationsCount=pnp_ransac_iter,
                                                                   reprojectionError=pnp_inlier_thresh,
                                                                   confidence=pnp_required_ransac_conf,
                                                                   flags=cv2.SOLVEPNP_ITERATIVE,
                                                                   )

    except:
        print("Solving PnP failed!")
        pose_est_success = False
        return None, None, None

    else:
        # Optional LM refinement on inliers.
        if pose_est_success and pnp_refine_lm:
            mask = inliers[weights[inliers] ==1]

            rvec, tvec = cv2.solvePnPRefineLM(
                objectPoints=np.array(points2_3D).astype(np.float64)[mask],
                imagePoints=valid_points1[mask],
                cameraMatrix=cam_K,
                distCoeffs=None,
                rvec=rvec,
                tvec=tvec,
            )

        R_est = cv2.Rodrigues(rvec)[0]
        quality = 0.0
        if pose_est_success:
            quality = float(len(inliers))

        t_est = np.squeeze(tvec) * scale_factor

    return R_est, t_est, quality


def get_pose_from_weighted_correspondences(points1, points2, y_offset, x_offset, img_uv, cam_K, norm_factor,
                                           scale_factor,
                                           weights, resize_factor=1.0):
    # filter valid points
    valid_points1 = []
    valid_points2 = []
    valid_weights = []
    for point1, point2, weight in zip(points1, points2, weights):
        if 0 <= round(point2[0]) < img_uv.shape[0] and 0 <= round(point2[1]) < img_uv.shape[1]:
            if np.any(img_uv[round(point2[0]), round(point2[1])] != [0, 0, 0]):
                valid_points1.append(point1)
                valid_points2.append(point2)
                valid_weights.append(weight)

    # Check if enough correspondences for PnP
    if len(valid_points1) < 6:
        return None, None, 0

    points2_3D = transform_2D_3D(valid_points2, img_uv, norm_factor)
    valid_points1 = np.array(valid_points1).astype(np.float64) / resize_factor

    # Apply offsets to the 2D image points
    valid_points1[:, 0] += y_offset
    valid_points1[:, 1] += x_offset

    # Swap x and y axes to match the expected format
    valid_points1[:, [0, 1]] = valid_points1[:, [1, 0]]

    try:
        # Call weighted solvePnP
        rvec, tvec, inliners = weighted_solvePnP(np.array(points2_3D), valid_points1, cam_K,
                                                 np.array(valid_weights))

        R_est = cv2.Rodrigues(rvec)[0]
        t_est = np.squeeze(tvec) * scale_factor
        quality = inliners  # Use the final weighted reprojection error as quality

    except Exception as e:
        print(f"Weighted SolvePnP failed: {e}")
        return None, None, 0

    return R_est, t_est, quality


def smooth_l1_loss(differences):
    # Apply Smooth L1 Loss formula
    abs_diff = np.abs(differences)
    quadratic = np.minimum(abs_diff, 1)
    linear = abs_diff - quadratic
    smooth_l1 = 0.5 * quadratic ** 2 + linear
    return smooth_l1


def weighted_solvePnP(object_points, image_points, camera_matrix, weights, distCoeffs=None, rvec=None, tvec=None):
    # Normalize weights
    weights = np.array(weights)

    # Reprojection error calculation function
    def reprojection_error(params):
        # Extract rvec and tvec from the parameter vector
        rvec = params[:3].reshape(3, 1)
        tvec = params[3:].reshape(3, 1)

        # Project 3D points to 2D using current rvec and tvec
        projected_points, _ = cv2.projectPoints(object_points, rvec, tvec, camera_matrix, distCoeffs)
        projected_points = projected_points.squeeze()  # Flatten to shape (N, 2)

        # Compute the reprojection error as the Euclidean distance (L2 norm) between projected and actual image points
        errors = np.linalg.norm(projected_points - image_points, axis=1)  # Shape (N,)

        # Apply weights to the reprojection errors
        weighted_errors = errors * weights  # Shape (N,)

        # Return the weighted errors (no need to flatten here since the output is already 1D)
        return weighted_errors

    # If no initial guess for rvec and tvec, use cv2.solvePnP
    if rvec is None or tvec is None:

        pnp_inlier_thresh = 8
        pnp_required_ransac_conf = 0.99
        pnp_ransac_iter = 400

        success, rvec, tvec, inliners = cv2.solvePnPRansac(object_points, image_points, camera_matrix,
                                                           distCoeffs=distCoeffs, iterationsCount=pnp_ransac_iter,
                                                           reprojectionError=pnp_inlier_thresh,
                                                           confidence=pnp_required_ransac_conf,
                                                           flags=cv2.SOLVEPNP_ITERATIVE)
        # success, rvec, tvec = cv2.solvePnP(object_points, image_points, camera_matrix, distCoeffs)
        if not success:
            raise ValueError("Initial solvePnP failed.")

    # Flatten rvec and tvec into a single parameter vector
    params_init = np.hstack([rvec.flatten(), tvec.flatten()])

    # Optimize the pose using least squares, minimizing the reprojection error
    # Call least_squares to optimize pose
    result = least_squares(
        reprojection_error,
        params_init,
        method='lm'
    )

    # Extract optimized rvec and tvec from result
    rvec_opt = result.x[:3].reshape(3, 1)
    tvec_opt = result.x[3:].reshape(3, 1)

    # Compute the final weighted reprojection error (counting errors below a threshold)
    reshaped_errors = reprojection_error(result.x)  # Recompute errors with optimized rvec, tvec
    final_error = np.sum(reshaped_errors < 8)  # Count how many errors are below the threshold

    return rvec_opt, tvec_opt, final_error


def weighted_solve_pnp_ransac(object_points, image_points, camera_matrix, dist_coeffs, weights, iterations=100,
                              reprojection_error=8.0):
    best_inliers = []
    best_inlier_count = 0

    num_points = len(image_points)

    for _ in range(iterations):
        sample_indices = np.random.choice(num_points, 4, p=weights, replace=False)
        sampled_object_points = object_points[sample_indices]
        sampled_image_points = image_points[sample_indices]

        success, rvec, tvec = cv2.solvePnP(sampled_object_points, sampled_image_points, camera_matrix, dist_coeffs)

        if not success:
            continue

        projected_image_points, _ = cv2.projectPoints(object_points, rvec, tvec, camera_matrix, dist_coeffs)
        error = np.linalg.norm(projected_image_points - image_points, axis=2).reshape(-1)
        weighted_error = error * weights
        inliers = np.where(weighted_error < reprojection_error)[0]

        inlier_count = len(inliers)

        if inlier_count > best_inlier_count:
            best_inlier_count = inlier_count
            best_inliers = inliers
            best_rvec = rvec
            best_tvec = tvec

    refined_rvec, refined_tvec = cv2.solvePnP(object_points[best_inliers], image_points[best_inliers], camera_matrix,
                                              dist_coeffs)[1:3]

    return refined_rvec, refined_tvec, best_inliers


def get_pose_from_correspondences(points1, points2, y_offset, x_offset, img_uv, cam_K, norm_factor, scale_factor,
                                  resize_factor=1.0, pnp_refine_lm=False):
    # filter valid points
    valid_points1 = []
    valid_points2 = []
    for point1, point2 in zip(points1, points2):
        if np.any(img_uv[round(point2[0]), round(point2[1])] != [0, 0, 0]):
            valid_points1.append(point1)
            valid_points2.append(point2)

    # Check if enough correspondences for PnPRansac
    if len(valid_points1) < 6:
        return None, None, None, None

    points2_3D = transform_2D_3D(valid_points2, img_uv, norm_factor)

    valid_points1 = np.array(valid_points1).astype(np.float64) / resize_factor

    valid_points1[:, 0] += y_offset
    valid_points1[:, 1] += x_offset

    valid_points1[:, [0, 1]] = valid_points1[:, [1, 0]]

    pnp_inlier_thresh = 8
    pnp_required_ransac_conf = 0.99
    pnp_ransac_iter = 400

    try:
        pose_est_success, rvec, tvec, inliers = cv2.solvePnPRansac(np.array(points2_3D).astype(np.float64),
                                                                   valid_points1, cam_K,
                                                                   distCoeffs=None, iterationsCount=pnp_ransac_iter,
                                                                   reprojectionError=pnp_inlier_thresh,
                                                                   confidence=pnp_required_ransac_conf,
                                                                   flags=cv2.SOLVEPNP_ITERATIVE,
                                                                   )

    except:
        print("Solving PnP failed!")
        pose_est_success = False
        return None, None, None, None

    else:
        # Optional LM refinement on inliers.
        if pose_est_success and pnp_refine_lm:
            rvec, tvec = cv2.solvePnPRefineLM(
                objectPoints=np.array(points2_3D).astype(np.float64)[inliers],
                imagePoints=valid_points1[inliers],
                cameraMatrix=cam_K,
                distCoeffs=None,
                rvec=rvec,
                tvec=tvec,
            )

        R_est = cv2.Rodrigues(rvec)[0]
        quality = 0.0
        if pose_est_success:
            quality = float(len(inliers))

        t_est = np.squeeze(tvec) * scale_factor

    return R_est, t_est, quality, inliers


def poselib_correspondences(points1, points2, y_offset, x_offset, img_uv, cam_K, norm_factor, scale_factor,
                            resize_factor=1.0, img_size=None, distortion=None):
    # filter valid points
    valid_points1 = []
    valid_points2 = []
    for point1, point2 in zip(points1, points2):
        if np.any(img_uv[round(point2[0]), round(point2[1])] != [0, 0, 0]):
            valid_points1.append(point1)
            valid_points2.append(point2)

    # Check if enough correspondences for PnPRansac
    if len(valid_points1) < 4:
        return None, None

    points2_3D = transform_2D_3D(valid_points2, img_uv, norm_factor)

    valid_points1 = np.array(valid_points1).astype(np.float64) / resize_factor

    valid_points1[:, 0] += y_offset
    valid_points1[:, 1] += x_offset

    valid_points1[:, [0, 1]] = valid_points1[:, [1, 0]]

    try:
        confidence = 0.9999
        iterationsCount = 10_000
        reprojectionError = 5
        colmap_intrinsics = opencv_to_colmap_intrinsics(cam_K)
        fx = colmap_intrinsics[0, 0]
        fy = colmap_intrinsics[1, 1]
        cx = colmap_intrinsics[0, 2]
        cy = colmap_intrinsics[1, 2]
        width = img_size[0] if img_size is not None else int(cx * 2)
        height = img_size[1] if img_size is not None else int(cy * 2)

        if distortion is None:
            camera = {'model': 'PINHOLE', 'width': width, 'height': height, 'params': [fx, fy, cx, cy]}
        else:
            camera = {'model': 'OPENCV', 'width': width, 'height': height,
                      'params': [fx, fy, cx, cy] + distortion}

        pts2D = np.copy(valid_points1)
        pts2D[:, 0] += 0.5
        pts2D[:, 1] += 0.5
        pose, _ = poselib.estimate_absolute_pose(pts2D, np.array(points2_3D).astype(np.float64), camera,
                                                 {'max_reproj_error': reprojectionError,
                                                  'max_iterations': iterationsCount,
                                                  'success_prob': confidence}, {})

        Rt = pose.Rt
        R_est = Rt[:, :3]
        tvec = Rt[:, 3]

    except:
        print("Solving PnP failed!")
        return None, None

    t_est = np.squeeze(tvec) * scale_factor

    return R_est, t_est


def load_test_list_and_cnos_detections(
        root_dir, dataset_name, cnos_file, max_det_per_object_id=None
):
    """
    We use a sorting techniques which has been done in MegaPose (thanks Mederic Fourmy for sharing!)
    Idea: when there is no detection at object level, we use the detections at image level
    """
    # load test list
    test_list = load_json(root_dir + dataset_name + "/test_targets_bop19.json")

    # load cnos detections
    cnos_dets_path = cnos_file
    all_cnos_dets = load_json(cnos_dets_path)

    # sort by image_id
    all_cnos_dets_per_image = group_by_image_level(all_cnos_dets, image_key="image_id")

    selected_detections = []
    for idx, test in tqdm(enumerate(test_list)):
        test_object_id = test["obj_id"]
        scene_id, im_id = test["scene_id"], test["im_id"]
        image_key = f"{scene_id:06d}_{im_id:06d}"

        # get the detections for the current image
        if image_key in all_cnos_dets_per_image:
            cnos_dets_per_image = all_cnos_dets_per_image[image_key]
            dets = [
                det
                for det in cnos_dets_per_image
                if (det["category_id"] == test_object_id)
            ]
            if len(dets) == 0:  # done in MegaPose
                dets = copy.deepcopy(cnos_dets_per_image)
                for det in dets:
                    det["category_id"] = test_object_id

            assert len(dets) > 0

            # sort the detections by score descending
            dets = sorted(
                dets,
                key=lambda x: x["score"],
                reverse=True,
            )
            # keep only the top detections
            if max_det_per_object_id is not None:
                num_instances = max(max_det_per_object_id, test["inst_count"])
            else:
                num_instances = test["inst_count"]
            dets = dets[:num_instances]
            selected_detections.append(dets)
        else:
            print(f"No detection for {image_key}")

    print(f"Detections: {len(test_list)} test samples!")
    assert len(selected_detections) == len(test_list)
    selected_detections = group_by_image_level(
        selected_detections, image_key="image_id"
    )
    test_list = group_by_image_level(test_list, image_key="im_id")
    return test_list, selected_detections


def group_by_image_level(data, image_key="im_id"):
    # group the detections by scene_id and im_id
    data_per_image = {}
    for det in data:
        if isinstance(det, dict):
            dets = [det]
        else:
            dets = det
        for det in dets:
            scene_id, im_id = int(det["scene_id"]), int(det[image_key])
            key = f"{scene_id:06d}_{im_id:06d}"
            if key not in data_per_image:
                data_per_image[key] = []
            data_per_image[key].append(det)
    return data_per_image


def load_json(path, keys_to_int=False):
    """Loads content of a JSON file.

    :param path: Path to the JSON file.
    :return: Content of the loaded JSON file.
    """

    # Keys to integers.
    def convert_keys_to_int(x):
        return {int(k) if k.lstrip("-").isdigit() else k: v for k, v in x.items()}

    with open(path, "r") as f:
        if keys_to_int:
            content = json.load(f, object_hook=lambda x: convert_keys_to_int(x))
        else:
            content = json.load(f)

    return content


def opencv_to_colmap_intrinsics(K):
    """
    Modify camera intrinsics to follow a different convention.
    Coordinates of the center of the top-left pixels are by default:
    - (0.5, 0.5) in Colmap
    - (0,0) in OpenCV
    """
    K = K.copy()
    K[0, 2] += 0.5
    K[1, 2] += 0.5
    return K
