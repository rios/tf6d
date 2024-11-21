import blenderproc
import bpy
import bmesh
import math
import numpy as np
import os


def get_camera_positions(num_views):
    """
    Generate camera positions around an object in a spherical pattern
    to cover the full sphere with `num_views` evenly spaced views.
    """

    # Calculate angle step based on number of views
    angle_step = 360 / num_views

    # Generate camera positions
    positions = []
    for i in range(num_views):
        azimuth = i * angle_step
        elevation = 25  # Fixed elevation angle, adjust as needed

        rad_az = math.radians(azimuth)
        rad_el = math.radians(elevation)

        x = math.cos(rad_el) * math.sin(rad_az)
        y = math.cos(rad_el) * math.cos(rad_az)
        z = math.sin(rad_el)

        positions.append((x, y, z))

    return positions


def normalize(vec):
    return vec / (np.linalg.norm(vec, axis=-1, keepdims=True))


def look_at(cam_location, point):
    # Camera points in positive z direction
    forward = point - cam_location
    forward = normalize(forward)

    tmp = np.array([0.0, 0.0, -1.0])
    # Print warning when camera location is parallel to tmp
    norm = min(
        np.linalg.norm(cam_location - tmp, axis=-1),
        np.linalg.norm(cam_location + tmp, axis=-1),
    )
    if norm < 1e-3:
        print("Warning: camera location is parallel to tmp")
        tmp = np.array([0.0, -1.0, 0.0])

    right = np.cross(tmp, forward)
    right = normalize(right)

    up = np.cross(forward, right)
    up = normalize(up)

    mat = np.stack((right, up, forward, cam_location), axis=-1)

    hom_vec = np.array([[0.0, 0.0, 0.0, 1.0]])

    if len(mat.shape) > 2:
        hom_vec = np.tile(hom_vec, [mat.shape[0], 1, 1])

    mat = np.concatenate((mat, hom_vec), axis=-2)
    return mat


def convert_location_to_rotation(locations):
    obj_poses = np.zeros((len(locations), 4, 4))
    for idx, pt in enumerate(locations):
        obj_poses[idx] = look_at(pt, np.array([0, 0, 0]))
    return obj_poses


def inverse_transform(poses):
    new_poses = np.zeros_like(poses)
    for idx_pose in range(len(poses)):
        rot = poses[idx_pose, :3, :3]
        t = poses[idx_pose, :3, 3]
        rot = np.transpose(rot)
        t = -np.matmul(rot, t)
        new_poses[idx_pose][3][3] = 1
        new_poses[idx_pose][:3, :3] = rot
        new_poses[idx_pose][:3, 3] = t
    return new_poses


if __name__ == "__main__":
    save_dir = "./predefined_poses"

    # Check if the directory exists
    if not os.path.exists(save_dir):
        # If not, create it
        os.makedirs(save_dir)

    num_views = 800  # Total number of views to render
    position_icosphere = np.asarray(get_camera_positions(num_views))
    cam_poses = convert_location_to_rotation(position_icosphere)
    cam_poses[:, :3, 3] *= 1000.  # Adjust scaling if needed
    print("cam_poses shape:")
    print(cam_poses.shape)
    np.save(f"{save_dir}/cam_poses_deng.npy", cam_poses)
    obj_poses = inverse_transform(cam_poses)
    print("obj_poses shape:")
    print(obj_poses.shape)
    np.save(f"{save_dir}/obj_poses_wdeng.npy", obj_poses)

    print("Output saved to: " + save_dir)
