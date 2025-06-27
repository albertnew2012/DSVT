import os
import warnings
from matplotlib.text import Annotation
import numpy as np
from multiprocessing import Pool, cpu_count
import time
import pcl
import open3d as o3d
from open3d.t.geometry import TriangleMesh
import json
import pickle
import torch
import re
import matplotlib.pylab as plt
from scipy.spatial import cKDTree
from tqdm import tqdm
from collections import Counter
import time
VISUALIZATION = True


def load_bin_file(file_path, save_into_pcd=False):
    """
    Load a .bin file containing point cloud data.

    Args:
        file_path (str): Path to the .bin file.

    Returns:
        numpy.ndarray: Loaded point cloud data.
    """
    try:
        # Assuming each point is stored as 4 floats (x, y, z, intensity)
        points = np.fromfile(file_path, dtype=np.float32).reshape(-1, 4)
    except Exception as e:
        print(f"Error loading file {file_path}: {e}")
        return None
    pcd_file = os.path.splitext(file_path)[0] + ".pcd"
    if save_into_pcd:
        pcd = pcl.PointCloud_PointXYZI(points)
        pcl.save(pcd, pcd_file)
    return points


def load_json_file(file_path):
    """
    Load a .bin file containing point cloud data.

    Args:
        file_path (str): Path to the .bin file.

    Returns:
        numpy.ndarray: Loaded point cloud data.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The file '{file_path}' does not exist.")
    with open(file_path, 'r') as file:
        d = json.load(file)
    return d


def range_filter(points, point_cloud_range):
    """
    Filters points in a point cloud to keep only those within the specified range.

    Parameters:
        points (numpy.ndarray): An Nx3 array where each row is a point (x, y, z).
        point_cloud_range (list or tuple): A 6-element list [x_min, y_min, z_min, x_max, y_max, z_max].

    Returns:
        numpy.ndarray: Filtered points within the specified range.
    """
    # Extract range boundaries
    x_min, y_min, z_min, x_max, y_max, z_max = point_cloud_range

    # Apply range filter
    mask = (
        (points[:, 0] >= x_min) & (points[:, 0] <= x_max) &
        (points[:, 1] >= y_min) & (points[:, 1] <= y_max) &
        (points[:, 2] >= z_min) & (points[:, 2] <= z_max)
    )

    # Return the filtered points
    return points[mask]


def filter_bounding_boxes(point_cloud, bounding_boxes, threshold):
    """
    Filters bounding boxes that contain fewer points than the threshold.

    Args:
        point_cloud (o3d.geometry.PointCloud): The input point cloud.
        bounding_boxes (list of np.ndarray): List of bounding boxes, each defined as [x, y, z, w, l, h, yaw].
        threshold (int): Minimum number of points required in a bounding box.

    Returns:
        filtered_boxes (list of np.ndarray): List of bounding boxes that meet the threshold.
    """
    points = np.asarray(
        point_cloud.points)  # Convert point cloud to numpy array
    filtered_boxes = []
    filtered_boxe_indices = []

    for i, box in enumerate(bounding_boxes):
        x, y, z, w, l, h, yaw = box  # Bounding box parameters

        # Create a rotation matrix for the bounding box
        cos_yaw, sin_yaw = np.cos(yaw), np.sin(yaw)
        rotation_matrix = np.array([
            [cos_yaw, -sin_yaw, 0],
            [sin_yaw,  cos_yaw, 0],
            [0,        0,       1]
        ])

        # Translate points relative to the bounding box center
        centered_points = points[:, :3] - np.array([x, y, z])

        # Rotate points into the bounding box frame
        rotated_points = centered_points @ rotation_matrix.T

        # Check if points are inside the bounding box (axis-aligned after rotation)
        mask = (
            (np.abs(rotated_points[:, 0]) <= w / 2) &
            (np.abs(rotated_points[:, 1]) <= l / 2) &
            (np.abs(rotated_points[:, 2]) <= h / 2)
        )

        num_points_in_box = np.sum(mask)

        # Keep the box if it contains enough points
        if num_points_in_box >= threshold:
            filtered_boxes.append(box)
            filtered_boxe_indices.append(i)

    return [filtered_boxes, filtered_boxe_indices]


def filter_bounding_boxes_fast(point_cloud, bounding_boxes, threshold):
    """
    Fast filtering of bounding boxes by using a KD-Tree to count points efficiently.

    Args:
        point_cloud (o3d.geometry.PointCloud): The input point cloud.
        bounding_boxes (list of np.ndarray): List of bounding boxes, each defined as [x, y, z, w, l, h, yaw].
        threshold (int): Minimum number of points required in a bounding box.

    Returns:
        filtered_boxes (list of np.ndarray): Bounding boxes with enough points.
    """
    points = np.asarray(
        point_cloud.points)  # Convert point cloud to numpy array
    # Build a KD-Tree for fast nearest neighbor search
    kdtree = cKDTree(points)
    filtered_boxes = []
    filtered_bbox_indices = []

    for i, box in enumerate(bounding_boxes):
        x, y, z, w, l, h, yaw, *_ = box  # Bounding box parameters

        # Get bounding box corner points (axis-aligned before rotation)
        half_w, half_l, half_h = w / 2, l / 2, h / 2
        min_bound = np.array([x - half_w, y - half_l, z - half_h])
        max_bound = np.array([x + half_w, y + half_l, z + half_h])

        # Query KD-Tree to get points inside the AABB (Axis-Aligned Bounding Box)
        inside_idx = kdtree.query_ball_point([x, y, z], max(w, l, h) / 2)
        candidate_points = points[inside_idx]

        # Rotate points into bounding box local frame
        cos_yaw, sin_yaw = np.cos(yaw), np.sin(yaw)
        rotation_matrix = np.array([
            [cos_yaw, -sin_yaw, 0],
            [sin_yaw,  cos_yaw, 0],
            [0,        0,       1]
        ])
        centered_points = candidate_points - np.array([x, y, z])
        rotated_points = centered_points @ rotation_matrix.T

        # Check if points are within the box after rotation
        mask = (
            (np.abs(rotated_points[:, 0]) <= half_w) &
            (np.abs(rotated_points[:, 1]) <= half_l) &
            (np.abs(rotated_points[:, 2]) <= half_h)
        )

        num_points_in_box = np.sum(mask)

        # Keep the box if it contains enough points
        if num_points_in_box >= threshold:
            filtered_boxes.append(box)
            filtered_bbox_indices.append(i)

    return [filtered_boxes, filtered_bbox_indices]


# Initialize the visualizer
if VISUALIZATION:
    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window(window_name="Point Cloud Visualization with Ground Plane",
                      width=2560, height=1440, visible=True)


# Variable to control frame update
frame_ready = [False]  # Use a list to modify it inside the callback


def next_frame_callback(vis):
    """
    Callback function to advance to the next frame on key press.
    """
    frame_ready[0] = True  # Signal to proceed to the next frame


# Register the spacebar key (' ') to proceed to the next frame
if VISUALIZATION:
    vis.register_key_callback(ord(" "), next_frame_callback)

if __name__ == "__main__":
    # os.chdir("/mmdetection3d")
    # load pickle file
    # pkl_path = "work_dirs/cpr_24amu_rt_ernest/results_al.pkl"
    pkl_path = "nuscenes3.pkl"
    # pkl_path = "lucid.pkl"

    # pkl_path = "/home/zliu3/Desktop/reclass_dg_2/draft/ct_20250502_130716_manual _cpr_24amu_rt_ernest_3_tree_4.pkl"

    with open(pkl_path, "rb") as file:
        pkl = pickle.load(file)

    num_of_points = [pkl[i]['prediction']['points'].shape[0]
                     for i in range(len(pkl))]
    for i in range(len(pkl)):
        # Path to the directory containing .bin files
        # bin_file = re.sub(r'^.*?/data', 'data', pkl[i]['point_cloud']['velodyne_path'])
        # if not os.path.exists(bin_file):
        #     raise FileNotFoundError(f"pcd file '{bin_file}' does not exist.")

        # Load and merge files with multiprocessing
        # points = load_bin_file(bin_file)
        points = pkl[i]['prediction']['points']
        point_cloud_range = [-54, -54, -5.0, 54, 54, 3.0]
        # filtered_points = range_filter(points, point_cloud_range)
        filtered_points = points
         
        # Create an Open3D PointCloud object
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(filtered_points[:, :3])

        if filtered_points[:, -1].max() > 1.0:
            filtered_points[:, -1] /= 255.0
        # Convert intensity to colors using a colormap
        assert 0 <= filtered_points[:, -1].min() and filtered_points[:, -1].max(
        ) <= 1.0, "Found point cloud intensity value outside of [0.0,1.0]"
        # You can use other colormaps like 'viridis' or 'plasma'
        colormap = plt.get_cmap("jet")
        # Extract RGB channels
        colors = colormap(filtered_points[:, -1])[:, :3]
        pcd.colors = o3d.utility.Vector3dVector(colors)

        # Define the Y-axis rotation matrix
        # Define the rotation angle in degrees
        angle_degrees = 0.00018
        # Convert degrees to radians
        angle_radian = np.radians(angle_degrees)
        rotation_matrix = pcd.get_rotation_matrix_from_axis_angle(
            [0, angle_radian, 0])  # [0, theta, 0] for Y-axis rotation

        # Apply the rotation with origin as the center
        # Center of rotation is the origin
        pcd.rotate(rotation_matrix, center=(0, 0, 0))

        # Add coordinate frame (axis display)
        coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
            size=2, origin=[0, 0, 0])

        # Extract dimensions from point_cloud_range
        x_min, y_min, z_min, x_max, y_max, z_max = point_cloud_range
        plane_width = x_max - x_min
        plane_height = y_max - y_min

        # Create a ground plane using a thin box
        plane = o3d.geometry.TriangleMesh.create_box(
            width=plane_width, height=plane_height, depth=0.01)

        # Position the plane at the center of the `point_cloud_range`
        # plane.translate([x_min, y_min, z_min])  # Align to the bottom-left corner of the range
        # Shift to the center of the range
        plane.translate([-plane_width/2, -plane_height / 2, -5.75])

        # Paint the plane light gray
        plane.paint_uniform_color([0.9, 0.9, 0.9])
        if VISUALIZATION:
            # Clear old geometries from the visualizer
            vis.clear_geometries()
            # Clear old geometries fr
            vis.add_geometry(coordinate_frame)
            vis.add_geometry(plane)
            vis.add_geometry(pcd)
            render_option = vis.get_render_option()
            render_option.point_size = 4.0  # Adjust the value as needed

        # ###################################################################################################
        # # annotation
        # ###################################################################################################
        # annotations = pkl[i]['annos']
        # # sanity check
        # shapes = [annotations[key].shape[0] for key in annotations.keys()]
        # if not all(shape == shapes[0] for shape in shapes):
        #     raise Exception("pkl file is corrupted")

        # # List to store bounding boxes
        # bounding_boxes = []

        # # Process each bounding box in annotation["objects"]
        # for j in range(shapes[0]):
        #     # Extract bounding box parameters
        #     center = annotations['location'][j]
        #     size = annotations['dimensions'][j]
        #     center[-1] += size[2]/2
        #     # yaw, pitch, roll = annotations['rotation_y'][j], 0, 0
        #     # yaw, pitch, roll = 0, 0, 0
        #     yaw, pitch, roll = annotations['alpha'][j], annotations['rotation_y'][j], 0

        #     # Create a rotation matrix from yaw, pitch, roll
        #     # The rotation is applied around the the box center.
        #     rotation_matrix = o3d.geometry.get_rotation_matrix_from_xyz([
        #                                                                 0, pitch, 0])

        #     # Create an oriented bounding box
        #     obb = o3d.geometry.OrientedBoundingBox(
        #         center, rotation_matrix, size)

        #     # # Step 3: Apply additional pitch correction (around origin)
        #     # pitch_correction_matrix = o3d.geometry.get_rotation_matrix_from_xyz(
        #     #     [roll, pitch, 0])
        #     # obb.rotate(pitch_correction_matrix, center=(
        #     #     0, 0, 0))  # Rotate around origin

        #     obb.color = (0, 1, 0)  # Red color for bounding boxes

        #     # Add to the list of bounding boxes
        #     # bounding_boxes.append(obb)
        #     vis.add_geometry(obb)

        ###################################################################################################
        # annotation
        ###################################################################################################
        # annotations = pkl[i]['gt_bboxes_3d']
        # # sanity check
        # # shapes = [annotations[key].shape[0] for key in annotations.keys()]
        # # if not all(shape == shapes[0] for shape in shapes):
        # #     raise Exception("pkl file is corrupted")

        # # List to store bounding boxes
        # bounding_boxes = []

        # # Process each bounding box in annotation["objects"]
        # for j in range(annotations.shape[0]):
        #     # Extract bounding box parameters
        #     center = annotations[j][:3]
        #     size = annotations[j][3:6]
        #     center[-1] += size[2]/2
        #     # yaw, pitch, roll = annotations['rotation_y'][j], 0, 0
        #     # yaw, pitch, roll = 0, 0, 0
        #     roll, pitch, yaw = 0, 0, annotations[j][6]
        #     # yaw, pitch, roll = annotations['alpha'][j], annotations['rotation_y'][j], 0

        #     # Create a rotation matrix from yaw, pitch, roll
        #     # The rotation is applied around the the box center.
        #     rotation_matrix = o3d.geometry.get_rotation_matrix_from_xyz([roll, pitch, yaw])

        #     # Create an oriented bounding box
        #     obb = o3d.geometry.OrientedBoundingBox(
        #         center, rotation_matrix, size)

        #     # # Step 3: Apply additional pitch correction (around origin)
        #     # pitch_correction_matrix = o3d.geometry.get_rotation_matrix_from_xyz(
        #     #     [roll, pitch, 0])
        #     # obb.rotate(pitch_correction_matrix, center=(
        #     #     0, 0, 0))  # Rotate around origin

        #     obb.color = (0, 1, 0)  # Red color for bounding boxes

        #     # Add to the list of bounding boxes
        #     # bounding_boxes.append(obb)
        #     vis.add_geometry(obb)

        ###################################################################################################
        # prediction
        ###################################################################################################
        preds = pkl[i]['prediction']
        # sanity check
        shapes = [preds[key].shape[0]
                  for key in preds.keys() if key != 'points']
        if not all(shape == shapes[0] for shape in shapes):
            raise Exception(
                "'boxes_3d', 'labels_3d', 'scores_3d' have different shapes!")

        bboxes = preds['boxes_3d'].numpy()
        threshold = 5  # Minimum number of points required per box
        # filtered_bboxes, filtered_bbox_indices = filter_bounding_boxes(pcd, bboxes, threshold)
        filtered_bboxes, filtered_bbox_indices = filter_bounding_boxes_fast(
            pcd, bboxes, threshold)
        # List to store bounding boxes
        pred_bounding_boxes = []

        print(f"======================frame: {i}======================")
        # Process each bounding box in annotation["objects"]
        for k in range(shapes[0]):
            # if k not in filtered_bbox_indices:
            #     continue

            # Extract bounding box parameters
            score = preds['scores_3d'][k].item()
            if score < 0.35:
                continue 
            # if score < 0.50 and preds['labels_3d'][k]==0  or score<0.42 and preds['labels_3d'][k]==1:
            #     continue
            # if preds['labels_3d'][k]!=2:
            #     continue
            if preds['labels_3d'][k] == 0 and score < 0.5:
                continue
            if preds['labels_3d'][k] == 1 and score < 0.5:
                continue
            if preds['labels_3d'][k] == 2 and score < 0.35:
                continue
            
            print(
                f"object type: {preds['labels_3d'][k]}, score:{preds['scores_3d'][k].item()}")
            center = preds['boxes_3d'][k][:3].numpy()
            size = preds['boxes_3d'][k][3:6].numpy()
            center[-1] += size[2]/2
            # yaw, pitch, roll = annotations['rotation_y'][k], 0, 0
            # yaw, pitch, roll = 0, 0, 0
            roll, pitch, yaw = 0, 0, preds['boxes_3d'][k][6].item()
            # yaw, pitch, roll = preds['boxes_3d'][k][6].item(), 0, 0

            # Create a rotation matrix from yaw, pitch, roll
            # The rotation is applied around the the box center.
            rotation_matrix = o3d.geometry.get_rotation_matrix_from_xyz(
                [roll, pitch, yaw])

            # Create an oriented bounding box
            obb = o3d.geometry.OrientedBoundingBox(
                center, rotation_matrix, size)

            # # Step 3: Apply additional pitch correction (around origin)
            # pitch_correction_matrix = o3d.geometry.get_rotation_matrix_from_xyz(
            #     [roll, pitch, 0])
            # obb.rotate(pitch_correction_matrix, center=(
            #     0, 0, 0))  # Rotate around origin
            # ['Car', 'Truck']

            if preds['labels_3d'][k] == 0:  # 'Car' in red
                obb.color = (1, 0, 0)
            elif preds['labels_3d'][k] == 1:  # 'Truck' in blue
                obb.color = (0, 1, 0)
            elif preds['labels_3d'][k] == 2:  # 'Motorcycle' in yellow
                obb.color = (0.502, 0.000, 0.502)  # (1, 1, 0)
            elif preds['labels_3d'][k] == 3:  # 'Pedestrian' in green
                obb.color = (0, 1, 1)
            else:
                obb.color = (0, 0, 0)
            # Add to the list of bounding boxes
            # bounding_boxes.append(obb)

            label_mesh = o3d.t.geometry.TriangleMesh.create_text(
                f"{score:0.2f}", depth=2).to_legacy()
            label_mesh.paint_uniform_color(obb.color)

            # Apply transformations to make the text mesh parallel to the yz plane
            # label_mesh.transform([[0, 0, 0.1, center[0]],
            #                       [0, 0.1, 0, center[1]],
            #                       [0.1, 0, 0, center[2]],
            #                       [0, 0, 0, 1]])

            # label_mesh.transform([[0,    0,    0.1, center[0]],
            #                       [-0.1, 0,    0,   center[1]],
            #                       [0,    0.1,  0,   center[2]],
            #                       [0,    0,    0,   1]])

            label_mesh.transform([[0.1, 0,   0,   center[0]],
                                [0,   0.1, 0,   center[1]],
                                [0,   0,   0.1, center[2]],
                                [0,   0,   0,   1]])


            label_mesh = label_mesh.subdivide_midpoint(number_of_iterations=1)

            if VISUALIZATION:
                vis.add_geometry(obb)
                vis.add_geometry(label_mesh)

            # Set the line width for bounding boxes
            render_option = vis.get_render_option()
            render_option.line_width = 10.0  # Adjust the value as needed

        # # Visualize
        # o3d.visualization.draw_geometries([pcd, coordinate_frame, plane] + bounding_boxes,
        #                                 window_name="Point Cloud Visualization with Ground Plane",
        #                                 width=2560,
        #                                 height=1440,
        #                                 point_show_normal=False)
        if VISUALIZATION:
            # Access the ViewControl object
            view_control = vis.get_view_control()

            # Set the zoom level
            # Smaller values zoom out, larger values zoom in (default is around 0.7)
            view_control.set_zoom(0.25)  # Example: reduc
            # Set camera parameters for finer control
            # Set the point the camera looks at
            view_control.set_lookat([0, 0, 0])
            # Set the upward direction of the camera
            view_control.set_up([1, 0, 0])
            # view_control.set_front([0, 0, 1])  # Set the direction the camera faces
            # Set the direction the camera faces
            view_control.set_front([0, 0, 1])
            render_option = vis.get_render_option()
            # Set line thickness (default is usually 1.0)
            render_option.line_width = 10.0
            # Render the frame and wait for key press
            frame_ready[0] = False  # Reset the frame_ready flag
            while not frame_ready[0]:
                vis.poll_events()
                vis.update_renderer()
            # vis.poll_events()
            # vis.update_renderer()
            # time.sleep(0.3)
    # Close the visualizer after the loop
    if VISUALIZATION:
        vis.destroy_window()
    print("visulized one frame!")
    # print(data)

    # theta = np.radians(angle_degrees)  # Convert degrees to radians

    # # Define the Y-axis rotation matrix
    # rotation_matrix = np.array([q
    #     [np.cos(theta), 0, np.sin(theta)],
    #     [0, 1, 0],
    #     [-np.sin(theta), 0, np.cos(theta)]
    # ])
