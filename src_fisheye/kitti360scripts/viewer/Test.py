import os
import sys
import open3d as o3d
import numpy as np
from kitti360Viewer3D import Kitti360Viewer3D

from kitti360scripts.helpers.project import CameraFisheye
# --- Configuration: Car dimensions (in meters) ---
CAR_LENGTH = 4.2   # along x
CAR_WIDTH  = 1.8   # along z
CAR_HEIGHT = 1.6   # along y
SEQUENCE =1 #int(input("Enter the sequence number: "))
SCENES = '{:0>4}'.format(SEQUENCE)
STOP = 5

if 'KITTI360_DATASET' in os.environ:
    kitti360Path = os.environ['KITTI360_DATASET']
else:
    print('error')

IMU_GPS_POSES_PATH = r"/data_poses/2013_05_28_drive_" + SCENES + r"_sync/poses.txt"
STATIC_PATH = r"/data_3d_semantics/train/2013_05_28_drive_" + SCENES + r"_sync/static"
DYNAMIC_PATH = r"/data_3d_semantics/train/2013_05_28_drive_" + SCENES + r"_sync/dynamic"
print(kitti360Path)
print(os.path.join(kitti360Path, IMU_GPS_POSES_PATH))

point = [[41.62605246, -0.77198424, -1.9628973 ],
         [41.62605246, -0.77198424, -1.9628973 ],
         [42.23029722, -0.78488249, -1.78551892]]

point = np.array(point, dtype=np.float32)
point[1,:] = point[1,:]*2
sequence = '2013_05_28_drive_0000_sync'
camera = CameraFisheye(kitti360Path, seq=sequence, cam_id=2)
# distorted_point = camera.cam2image_ocv2(point)

v0 = camera.fi['projection_parameters']['v0']
u0 = camera.fi['projection_parameters']['u0']
distorted_point = np.array([[[u0 - 700, v0],
                                    [u0 + 700, v0],
                                    [      u0, v0]]], dtype=np.float32)
Z = np.array([[[500],
                      [500],
                      [  0]]], dtype=np.float32)

undistorted_point = camera.image2cam_cv2(distorted_point, Z)
undistorted_point = np.squeeze(undistorted_point)
print('****************** distorted_point ***************\n',
      distorted_point,
      '\n****************** undistorted_point***************\n',
      undistorted_point, undistorted_point.shape)

