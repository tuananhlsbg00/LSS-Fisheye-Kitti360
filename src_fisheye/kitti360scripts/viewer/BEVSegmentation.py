import os
import sys
import numpy as np
import cv2
import matplotlib.pyplot as plt

from kitti360scripts.helpers.labels import name2label, id2label, kittiId2label
from kitti360scripts.helpers.project import CameraFisheye

# Import the Annotation3D class and helper functions from the KITTI360 codebase.
from kitti360scripts.helpers.annotation import Annotation3D, local2global, global2local

# ---------------------------
# Configuration
# ---------------------------    

# BEV configuration: 20mx20m area, with resolution 200x200 (0.1 m per pixel)
bev_size = 20.0  # in meters
bev_resolution = 200
bev_min = -bev_size / 2.0  # -10 m
bev_max = bev_size / 2.0  # 10 m

# We want the BEV plane to be 0.9m below the IMU origin.
# If the IMU coordinate system's origin is at the sensor and the ground is 0.9 m below,
# then transforming world -> IMU may include that shift.
# Here, we define a translation transformation that moves points down by 0.9 m.
# When converting from world to IMU, we will use the inverse of the pose from poses.txt,
# which is the IMU_to_world matrix. (Assuming that matrix already accounts for sensor mounting,
# this additional translation will bring the BEV plane to z=0.)
# For this example, we assume that after inverting the pose, we further translate by -0.9 in z.
T_additional = np.eye(4)
T_additional[0, 3] =  -.81  # Shift in X (backward)
T_additional[1, 3] =  -.32  # Shift in Y (left)
T_additional[2, 3] = -0.9   # Shift in Z (down)


# ---------------------------
# Helper Functions
# ---------------------------

def load_poses(filename, start=0, max_poses=300):
    """
    Load poses from a file. Each line should have 13 numbers: a frame index
    and 12 numbers representing a 3x4 IMU_to_world matrix.
    Returns a list of (frame, T_imu_to_world) tuples.
    """
    poses = np.loadtxt(filename, dtype=np.float32, skiprows=start, max_rows=max_poses )
    return poses

def assign_color(globalId):
    """
    Assign the correct color from the KITTI 360 labels.
    """
    semanticId, instanceId = global2local(globalId)

    if semanticId in id2label:
        return id2label[semanticId].color  # RGB from labels.py
    elif instanceId > 0:
        return (255, 0, 0)  # Assign red to unique instances (adjust if needed)
    else:
        return (96, 96, 96)  # Default color for "stuff" objects

def transform_points(points, T):
    """
    Transform an array of 3D points (N x 3) using a 4x4 transformation matrix T.
    """
    N = points.shape[0]
    points_hom = np.hstack((points, np.ones((N, 1))))
    points_trans = (T @ points_hom.T).T
    return points_trans[:, :3]


def world_to_bev_indices(points_xy, bev_min, bev_max, resolution):
    """
    Convert 2D coordinates (in meters) to pixel indices on a BEV grid.
    """
    scale = resolution / (bev_max - bev_min)  # pixels per meter
    indices = ((points_xy - bev_min) * scale).astype(np.int32)
    return indices


def fill_polygon(seg_map, polygon, color=None):
    """
    Fills a polygon on the segmentation map with the given RGB color.
    - seg_map: (H, W, 3) NumPy array (RGB image)
    - polygon: List of (x, y) points
    - color: (R, G, B) tuple
    """
    # Convert polygon points to an integer NumPy array for OpenCV
    pts = np.array(polygon, dtype=np.int32).reshape((-1, 1, 2))

    # Ensure color is a NumPy array of type uint8
    if color:
        color = np.array(color, dtype=np.uint8)
        # Fill polygon with the RGB color
        cv2.fillPoly(seg_map, [pts], color.tolist())  # Convert to list for OpenCV
    else:
        cv2.fillPoly(seg_map, [pts], 1.0)



def get_bottom_face(vertices_imu):
    """
    Given a bounding box's full set of 3D vertices (N x 3) in the IMU coordinate frame,
    return the 4 corners that form the 'bottom face' (lowest Z).

    For a standard box with 8 corners, this yields the rectangular footprint.
    If the mesh has more than 8 vertices, we still pick the 4 with the lowest Z.
    """
    # Sort all vertices by their Z-coordinate (lowest to highest).
    sorted_by_z = vertices_imu[vertices_imu[:, 2].argsort()]

    # Take the first 4 entries => the ones with the lowest Z.
    bottom4 = sorted_by_z[4:]

    # OPTIONAL: Reorder these 4 points in a clockwise or counter-clockwise manner
    # to avoid crossing edges when we fill the polygon.
    # We'll do this by computing the centroid and sorting by angle around it.
    center = bottom4.mean(axis=0)
    angles = np.arctan2(bottom4[:, 1] - center[1], bottom4[:, 0] - center[0])
    reorder = angles.argsort()
    bottom4 = bottom4[reorder]

    return bottom4

def draw_ego_vehicle(bev_map):
    """
    Draws the ego vehicle at the center of the BEV map with a green box and arrow.
    - bev_map: The RGB segmentation image (H, W, 3).
    """
    H, W, _ = bev_map.shape  # Get BEV image dimensions
    cx, cy = W // 2, H // 2  # Center of the BEV map

    # Define vehicle size in pixels (approximate)
    car_w, car_h = 30, 16  # Width and Length in pixels (adjust as needed)

    # Define car rectangle (centered)
    car_box = np.array([
        [cx - car_w // 2, cy - car_h // 2],  # Bottom-left
        [cx + car_w // 2, cy - car_h // 2],  # Bottom-right
        [cx + car_w // 2, cy + car_h // 2],  # Top-right
        [cx - car_w // 2, cy + car_h // 2]   # Top-left
    ], dtype=np.int32)

    # Draw filled green rectangle
    cv2.fillPoly(bev_map, [car_box], (0, 255, 0))

    # Define arrowhead (pointing right)
    arrow_tip = (cx + car_w // 2 + 10, cy)  # Tip of arrow (right)
    arrow_top = (cx + car_w // 2, cy - car_h // 2)  # Top base
    arrow_bottom = (cx + car_w // 2, cy + car_h // 2)  # Bottom base
    arrow = np.array([arrow_tip, arrow_top, arrow_bottom], dtype=np.int32)

    # Draw the arrow in green
    cv2.fillPoly(bev_map, [arrow], (0, 255, 0))


def draw_fisheye_coverage(
        bev_map,
        bev_min,
        bev_max,
        bev_resolution,
        camera,
        color=(25, 25, 50)
):
    v0 = camera.fi['projection_parameters']['v0']
    u0 = camera.fi['projection_parameters']['u0']
    distorted_point = np.array([[[u0 - 643, v0],
                                 [u0 + 643, v0],
                                 [u0, v0]]], dtype=np.float32)
    Z = np.array([[[500],
                          [500],
                          [  0]]], dtype=np.float32)

    undistorted_point = np.squeeze(camera.image2cam_cv2(distorted_point, Z))

    # 4b) Transform coverage_cam from camera->IMU using camToPose
    #     camToPose is 4x4
    T_camToPose = camera.camToPose  # camera->IMU
    coverage_imu = []
    for pt in undistorted_point:
        pt_hom = np.array([pt[0], pt[1], pt[2], 1.0])
        pt_imu = T_camToPose @ pt_hom
        coverage_imu.append([pt_imu[0], pt_imu[1]])  # just x,y

    coverage_imu = np.array(coverage_imu)

    # 4c) Convert coverage to BEV
    coverage_indices = world_to_bev_indices(coverage_imu, bev_min, bev_max, bev_resolution)
    fill_polygon(bev_map, coverage_indices, color=color)  # dim gray wedge

# ---------------------------
# Main Script
# ---------------------------

def main():
    # Set paths (adjust as needed)
    if 'KITTI360_DATASET' in os.environ:
        kitti360Path = os.environ['KITTI360_DATASET']
    else:
        raise RuntimeError("Please set KITTI360_DATASET in your environment.")

    sequence = '2013_05_28_drive_0003_sync'

    # Load 3D bounding box annotations
    label3DBboxPath = os.path.join(kitti360Path, 'data_3d_bboxes')
    poses_file = os.path.join(kitti360Path, 'data_poses', sequence, 'poses.txt')
    print("Loading 3D bounding box annotations ...")
    annotation3D = Annotation3D(label3DBboxPath, sequence)

    print("Loading poses ...")
    poses = load_poses(poses_file, max_poses=None)
    print(f"Loaded {len(poses)} poses.")

    cam_02 = CameraFisheye(kitti360Path, sequence, cam_id=2)
    cam_03 = CameraFisheye(kitti360Path, sequence, cam_id=3)

    for pose in poses:

        frame, T_imu_to_world = int(pose[0]), pose[1:].reshape(1,12)
        pad = np.zeros((1,4))
        pad[:,-1] = 1
        T_imu_to_world = np.hstack((T_imu_to_world, pad))

        T_imu_to_world = T_imu_to_world.reshape((4, 4))
        T_world_to_imu = np.linalg.inv(T_imu_to_world)
        T_total = T_additional @ T_world_to_imu  # Transformation from world to BEV (ego-car)

        bev_map = np.zeros((bev_resolution, bev_resolution, 3), dtype=np.uint8)

        draw_ego_vehicle(bev_map)

        draw_fisheye_coverage(bev_map=bev_map, bev_min=bev_min,
                              bev_max=bev_max, bev_resolution=bev_resolution,
                              camera=cam_02,path=kitti360Path, seq=sequence)

        draw_fisheye_coverage(bev_map=bev_map, bev_min=bev_min,
                              bev_max=bev_max, bev_resolution=bev_resolution,
                              camera=cam_03,path=kitti360Path, seq=sequence)

        for globalId, bbox_dict in annotation3D.objects.items():
            if len(bbox_dict) != 1:
                continue
            obj = list(bbox_dict.values())[0]

            # Transform vertices to IMU coordinates
            vertices_world = obj.vertices
            vertices_imu = transform_points(vertices_world, T_total)

            # Extract only the bottom face (4 points)
            bottom4 = get_bottom_face(vertices_imu)
            bottom4_xy = bottom4[:, :2]  # (4,2) shape

            # Ensure bounding box is within the BEV area
            if (bottom4_xy[:, 0].max() < bev_min or bottom4_xy[:, 0].min() > bev_max or
                bottom4_xy[:, 1].max() < bev_min or bottom4_xy[:, 1].min() > bev_max):
                continue

            # Convert (x,y) to pixel indices
            indices = world_to_bev_indices(bottom4_xy, bev_min, bev_max, bev_resolution)
            # Fill the polygon using 4 bottom face points
            label_val = globalId % 256
            # Get the correct RGB color
            r,g,b = assign_color(globalId)  # Fetch color from labels.py
            color = (b, g, r)
            # Fill the polygon with the assigned color
            fill_polygon(bev_map, indices, color)

            if  (bottom4_xy[:, 1].max() < 0):

                #from ego pose to cam coordinate
                vertices_cam = (np.linalg.inv(cam_02.camToPose) @ np.hstack((vertices_imu, np.ones((8,1)))).T)[:3,:]
                # print(vertices_cam, vertices_cam.shape)
                #from cam to image coordinate
                Px, Py, _ = cam_02.cam2image(vertices_cam)

                vertices_img = np.vstack((Px, Py)).T
                #unproject back to cam coordinate
                vertices_unprojected = np.squeeze(cam_02.image2cam_cv2(vertices_img, Z = vertices_cam.T[:,2]))
                for unprojected_point in vertices_unprojected:
                    unprojected_point = np.hstack((unprojected_point, [1]))
                    unprojected = np.vstack(([0,0,0,1], unprojected_point))
                    #rig transform to ego coordinate
                    unprojected_imu = (cam_02.camToPose @ unprojected.T).T
                    unprojected_bottom4 = get_bottom_face(unprojected_imu)
                    unprojected_bottom4_xy = unprojected_bottom4[:, :2]
                    indices = world_to_bev_indices(unprojected_bottom4_xy, bev_min, bev_max, bev_resolution)
                    cv2.line(bev_map, indices[0], indices[1], [255,255,255],1)

            if  (bottom4_xy[:, 1].min() > 0):

                #from ego pose to cam coordinate
                vertices_cam = (np.linalg.inv(cam_03.camToPose) @ np.hstack((vertices_imu, np.ones((8,1)))).T)[:3,:]
                # print(vertices_cam, vertices_cam.shape)
                #from cam to image coordinate
                Px, Py, _ = cam_03.cam2image(vertices_cam)

                vertices_img = np.vstack((Px, Py)).T
                #unproject back to cam coordinate
                vertices_unprojected = np.squeeze(cam_03.image2cam_cv2(vertices_img, Z = vertices_cam.T[:,2]))
                for unprojected_point in vertices_unprojected:
                    unprojected_point = np.hstack((unprojected_point, [1]))
                    unprojected = np.vstack(([0,0,0,1], unprojected_point))
                    #rig transform to ego coordinate
                    unprojected_imu = (cam_03.camToPose @ unprojected.T).T
                    unprojected_bottom4 = get_bottom_face(unprojected_imu)
                    unprojected_bottom4_xy = unprojected_bottom4[:, :2]
                    indices = world_to_bev_indices(unprojected_bottom4_xy, bev_min, bev_max, bev_resolution)
                    cv2.line(bev_map, indices[0], indices[1], [255,255,255],1)



        # Save segmentation map
        output_filename = f"labels/bev_seg_{frame:06d}.jpg"
        bev_map = cv2.rotate(bev_map, cv2.ROTATE_90_COUNTERCLOCKWISE)
        W, H, _ = bev_map.shape
        bev_map = cv2.resize(bev_map, (W*3, H*3))
        cv2.imwrite(output_filename, bev_map)
        print(f"Saved BEV segmentation for frame {frame} as {output_filename}")


if __name__ == "__main__":
    main()
