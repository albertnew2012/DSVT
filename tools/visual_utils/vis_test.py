import open3d as o3d
import numpy as np


def create_bounding_box(center, extent, yaw):
    # Create an oriented bounding box
    bbox = o3d.geometry.OrientedBoundingBox()
    bbox.center = center
    bbox.extent = extent
    # Apply rotation
    R = bbox.get_rotation_matrix_from_xyz((0, 0, yaw))
    bbox.rotate(R, center=center)

    return bbox


colors = [
    (1, 0, 0),     # Red
    (0, 0, 1),     # Blue
    (0, 0, 0),     # Black
    (1, 1, 0),     # Yellow
    (0, 1, 0),     # Green
    (1, 0, 1),     # Magenta
    (0, 1, 1),     # Cyan
]
def visualize(points: np.array, boxes: np.array, labels=None):
    # create pcd object
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points[:, :3])
    geometries = [pcd]
    bounding_boxes = []
    for i, box in enumerate(boxes):
        center = box[:3]
        extent = box[3:6]
        yaw = box[6]  # assuming yaw is provided in the boxes array
        bbox = create_bounding_box(center, extent, yaw)
        if labels is not None:
            bbox.color = colors[labels[i]-1]  # Set bounding box color to red
        geometries.append(bbox)
    # Visualize point cloud
    o3d.visualization.draw_geometries(geometries)
    # Capture and save the screen image

def load_pcd_bin_file(file_path):
    # Load the binary point cloud file
    with open(file_path, 'rb') as f:
        # Each point is represented by 5 floats (x, y, z, intensity, ring)
        points = np.fromfile(f, dtype=np.float32).reshape(-1, 5)
    
    # Extract XYZ coordinates
    xyz = points[:, :3]

    # Create an Open3D point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)

    return pcd

import pickle

# Path to your pickle file
val_path = 'output/cfgs/dsvt_models/dsvt_plain_1f_onestage_nusences_debug/default/eval/epoch_99/val/default/result.pkl'
# Open the pickle file in binary read mode
with open(val_path, 'rb') as file:
    # Load the data from the file
    results = pickle.load(file)

# Now `data` contains the object that was stored in the pickle file
print(results)


for ret in results:
    # label, score, boxes_lidar, pred_labels,frame_id,metadata = ret
    pcd_path = f"data/nuscenes/v1.0-mini/samples/LIDAR_TOP/{ret['frame_id']}.bin"
    # Load the point cloud
    pcd = load_pcd_bin_file(pcd_path)
    # Visualize the point cloud
    # o3d.visualization.draw_geometries([pcd])
    visualize(np.asarray(pcd.points), ret['boxes_lidar'], ret['pred_labels'])