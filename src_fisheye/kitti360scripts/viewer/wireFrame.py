import os
import sys
import open3d as o3d
import numpy as np
from kitti360Viewer3D import Kitti360Viewer3D
import matplotlib.pyplot as plt
from os import path, listdir
# --- Configuration: Car dimensions (in meters) ---
CAR_LENGTH = 4.2   # along x
CAR_WIDTH  = 1.8   # along z
CAR_HEIGHT = 1.6   # along y
SEQUENCE = int(input("Enter the sequence number: "))
SCENES = '{:0>4}'.format(SEQUENCE)
STOP = 5

if 'KITTI360_DATASET' in os.environ:
    kitti360Path = os.environ['KITTI360_DATASET']
else:
    kitti360Path = os.path.join(os.path.dirname(
        os.path.realpath(__file__)), '..', '..')

IMU_GPS_POSES_PATH = r"data_poses/2013_05_28_drive_" + SCENES + r"_sync/poses.txt"
STATIC_PATH = r"data_3d_semantics/train/2013_05_28_drive_" + SCENES + r"_sync/static"
DYNAMIC_PATH = r"data_3d_semantics/train/2013_05_28_drive_" + SCENES + r"_sync/dynamic"

# --- Helper function: Load poses from file ---
def load_poses(filename, max_poses= STOP*295):
    poses = []
    with open(filename, "r") as f:
        for i, line in enumerate(f):
            if i >= max_poses:
                break
            if not line.strip():
                continue
            parts = line.split()
            frame = int(parts[0])
            nums = list(map(float, parts[1:]))
            if len(nums) != 12:
                continue
            # Reshape to 3x4 and convert to 4x4 matrix
            pose3x4 = np.array(nums).reshape(3, 4)
            pose = np.vstack((pose3x4, [0, 0, 0, 1]))
            poses.append((frame, pose))
    return poses

# --- Helper function: Create a car box centered at the origin ---
def create_car_box(car_length, car_height, car_width):
    # Open3D creates a box with one corner at the origin; we create one and then translate it so the center is at (0,0,0).
    box = o3d.geometry.TriangleMesh.create_box(width=car_length, height=car_width, depth=car_height)
    box.translate(np.array([-car_length/2, -car_height/2, -car_width/2]))
    return box

# Ensure the script can find the KITTI360 modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Sequence number (update as needed)


# Initialize the viewer
viewer = Kitti360Viewer3D(seq=SEQUENCE)

# Load all PLY files (static + dynamic point clouds)
viewer.loadWindows(colorType='semantic', stop=10)

# Load 3D bounding boxes as wireframes
viewer.loadBoundingBoxWireframes()

poses = load_poses(os.path.join(kitti360Path, IMU_GPS_POSES_PATH))

# Extract frame numbers for colormap normalization
frames = np.array([frame for frame, _ in poses])
min_frame, max_frame = frames.min(), frames.max()
frames_norm = (frames - min_frame) / (max_frame - min_frame)

# Use a colormap: blue for early frames, red for later ones (jet typically goes from blue to red)
cmap = plt.get_cmap("jet")

car_boxes = []  # list to store the transformed car boxes

for i, ((frame, pose), norm) in enumerate(zip(poses, frames_norm)):
    if i % 10 != 0:  # Only visualize every 10th pose
        continue
    car_box = create_car_box(CAR_LENGTH, CAR_HEIGHT, CAR_WIDTH)
    car_box.transform(pose)
    color = cmap(norm)[:3]
    car_box.paint_uniform_color(color)
    car_boxes.append(car_box)

# Create Open3D visualization
geometries = list(viewer.pointClouds.values()) + viewer.bboxes + viewer.lineSets + car_boxes

# Create Open3D visualizer
vis = o3d.visualization.Visualizer()
vis.create_window()

# Add all objects to the visualizer
for geom in geometries:
    vis.add_geometry(geom)

# Set the camera viewpoint
# ctr = vis.get_view_control()
# ctr.convert_from_pinhole_camera_parameters(camera_pose)

# Run the Open3D visualizer
vis.run()
vis.destroy_window()

