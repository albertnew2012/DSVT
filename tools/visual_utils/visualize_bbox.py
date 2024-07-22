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


def visualize(points: np.array, boxes: np.array):
    # create pcd object
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points[:, :3])
    geometries = [pcd]
    bounding_boxes = []
    for box in boxes:
        center = box[:3]
        extent = box[3:6]
        yaw = box[6]  # assuming yaw is provided in the boxes array
        bbox = create_bounding_box(center, extent, yaw)
        bbox.color = (1, 0, 0)  # Set bounding box color to red
        geometries.append(bbox)
    # Visualize point cloud
    o3d.visualization.draw_geometries(geometries)
    # Capture and save the screen image


data = train_set[0]
visualize(data["points"], data["gt_boxes"])

##############################################################################

import matplotlib.pyplot as plt

# Coordinates of the triangle's vertices
x = [0, 1, 0.5, 0]  # x-coordinates
y = [0, 0, 1, 0]    # y-coordinates

# Create a new figure
plt.figure()

# Plot the triangle
plt.plot(x, y, marker='o')

# Set the aspect ratio of the plot to be equal
plt.gca().set_aspect('equal', adjustable='box')

# Set labels for the axes
plt.xlabel('X-axis')
plt.ylabel('Y-axis')

# Set the title of the plot
plt.title('Triangle Plot')

# Display the plot
plt.show()


