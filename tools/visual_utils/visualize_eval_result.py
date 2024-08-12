import open3d as o3d
import numpy as np
import os
import pickle
import argparse

n = 0

colors = [
    (1, 0, 0),     # Red
    (0, 0, 1),     # Blue
    (0, 0, 0),     # Black
    (1, 1, 0),     # Yellow
    (0, 1, 0),     # Green
    (1, 0, 1),     # Magenta
    (0, 1, 1),     # Cyan
    (0.5, 0, 0),   # Dark Red
    (0, 0.5, 0),   # Dark Green
    (0, 0, 0.5),   # Dark Blue
    (0.5, 0.5, 0),  # Olive
    (0.5, 0, 0.5),  # Purple
]


def create_bounding_box(center, extent, yaw):
    # Create an oriented bounding box
    bbox = o3d.geometry.OrientedBoundingBox()
    bbox.center = center
    bbox.extent = extent
    # Apply rotation
    R = bbox.get_rotation_matrix_from_xyz((0, 0, yaw))
    bbox.rotate(R, center=center)
    return bbox


def visualize(points: np.array, boxes: np.array, labels=None):
    # create pcd object
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points[:, :3])
    geometries = [pcd]
    for i, box in enumerate(boxes):
        center = box[:3]
        extent = box[3:6]
        yaw = box[6]  # assuming yaw is provided in the boxes array
        bbox = create_bounding_box(center, extent, yaw)
        if labels is not None:
            # Set bounding box color based on label
            bbox.color = colors[labels[i]-1]
        geometries.append(bbox)
    # Visualize point cloud
    o3d.visualization.draw_geometries(geometries)


def load_pcd_bin_file(file_path,save_bin=True):
    # Load the binary point cloud file
    with open(file_path, 'rb') as f:
        # By default, each point is represented by 5 floats (x, y, z, intensity, ring)
        points = np.fromfile(f, dtype=np.float32).reshape(-1, 5)

    # save points[:,:4] into .bin file for DSVT-AI-TRT
    if save_bin:
        global n
        n = n + 1
        file_name = str(n).zfill(6) + ".bin"
        points[:,:4].tofile("data/bin/" + file_name)
        
    # Extract XYZ coordinates
    xyz = points[:, :3]

    # Create an Open3D point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)

    return pcd


def main(results_path):
    # Open the pickle file in binary read mode
    with open(results_path, 'rb') as file:
        # Load the data from the file
        results = pickle.load(file)

    # Now `results` contains the object that was stored in the pickle file
    # print(results)

    for ret in results:
        pcd_path = f"data/nuscenes/v1.0-trainval/samples/LIDAR_TOP/{ret['frame_id']}.bin"
        if not os.path.exists(pcd_path):
            pcd_path = f"data/nuscenes/v1.0-mini/samples/LIDAR_TOP/{ret['frame_id']}.bin"
        if not os.path.exists(pcd_path):
            raise FileNotFoundError(
                f"The specified file {ret['frame_id']}.bin was not found.")
        # Load the point cloud
        pcd = load_pcd_bin_file(pcd_path)
        
        # Visualize the point cloud
        visualize(np.asarray(pcd.points),
                  ret['boxes_lidar'], ret['pred_labels'])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Visualize bounding boxes on point clouds.")
    parser.add_argument("result_path", type=str,
                        help="Path to the result.pkl file")
    args = parser.parse_args()
    main(args.result_path)
